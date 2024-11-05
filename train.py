import os
import sys
import numpy as np
from datetime import datetime
import argparse
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
from text2grasp import Text2Grasp, get_loss
from pytorch_utils import BNMomentumScheduler
from dataset.text2grasp_dataset import Text2GraspDataset, collate_fn


int_ = 111
random.seed(int_)
np.random.seed(int_)
torch.manual_seed(int_)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='', help='Dataset root')
parser.add_argument('--camera', default='kinect', help='Camera split [realsense/kinect]')
parser.add_argument('--checkpoint_path', help='Model checkpoint path', default=None)
parser.add_argument('--log_dir', default='', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--max_epoch', type=int, default=30, help='Epoch to run [default: 18]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 2]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--bn_decay_step', type=int, default=1, help='Period of BN decay (in epochs) [default: 2]')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
parser.add_argument('--lr_decay_steps', default='8,12,16',
                    help='When to decay the learning rate (in epochs) [default: 8,12,16]')
parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
cfgs = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
EPOCH_CNT = 0
LR_DECAY_STEPS = [int(x) for x in cfgs.lr_decay_steps.split(',')]
LR_DECAY_RATES = [float(x) for x in cfgs.lr_decay_rates.split(',')]
assert (len(LR_DECAY_STEPS) == len(LR_DECAY_RATES))
DEFAULT_CHECKPOINT_PATH = os.path.join(cfgs.log_dir, 'checkpoint.tar')
CHECKPOINT_PATH = cfgs.checkpoint_path if cfgs.checkpoint_path is not None \
    else DEFAULT_CHECKPOINT_PATH

if not os.path.exists(cfgs.log_dir):
    os.makedirs(cfgs.log_dir)

LOG_FOUT = open(os.path.join(cfgs.log_dir, 'log_train.txt'), 'a')
LOG_FOUT.write(str(cfgs) + '\n')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


net = Text2Grasp(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
                cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

optimizer = optim.Adam(net.parameters(), lr=cfgs.learning_rate, weight_decay=cfgs.weight_decay)

it = -1
start_epoch = 0
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    log_string("-> loaded checkpoint %s (epoch: %d)" % (CHECKPOINT_PATH, start_epoch))

BN_MOMENTUM_INIT = 0.5
BN_MOMENTUM_MAX = 0.001
bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * cfgs.bn_decay_rate ** (int(it / cfgs.bn_decay_step)), BN_MOMENTUM_MAX)
bnm_scheduler = BNMomentumScheduler(net, bn_lambda=bn_lbmd, last_epoch=start_epoch - 1)


def get_current_lr(epoch):
    lr = cfgs.learning_rate
    for i, lr_decay_epoch in enumerate(LR_DECAY_STEPS):
        if epoch >= lr_decay_epoch:
            lr *= LR_DECAY_RATES[i]
    return lr


def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


TRAIN_WRITER = SummaryWriter(os.path.join(cfgs.log_dir, 'train'))
TEST_WRITER = SummaryWriter(os.path.join(cfgs.log_dir, 'test'))


def train_ep(scene_start, scene_end):
    TRAIN_DATASET = Text2GraspDataset(cfgs.dataset_root, scene_start, scene_end, camera=cfgs.camera, split='train',
                                     num_points=cfgs.num_point, remove_outlier=True, augment=True)
    TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=cfgs.batch_size, shuffle=True,
                                  num_workers=4, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)
    print(len(TRAIN_DATALOADER))
    stat_dict = {}
    adjust_learning_rate(optimizer, EPOCH_CNT)
    bnm_scheduler.step()

    net.train()
    train_dataset = {
        'point_clouds': np.ndarray,
        'cloud_colors': np.ndarray,
        'objectness_label': np.ndarray,
        'object_poses_list': list,
        'grasp_points_list': list,
        'grasp_offsets_list': list,
        'grasp_labels_list': list,
        'grasp_tolerance_list': list,

    }
    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        object_poses = batch_data_label['object_poses_list']
        if not object_poses or any(object_poses[num] == 0 for num in range(len(object_poses))):
            continue
        else:
            for key in batch_data_label:
                train_dataset[key] = batch_data_label[key]
                if 'list' in key:
                    for i in range(len(batch_data_label[key])):
                        for j in range(len(batch_data_label[key][i])):
                            train_dataset[key][i][j] = batch_data_label[key][i][j].to(device)
                else:
                    train_dataset[key] = batch_data_label[key].to(device)

            # Forward pass
            end_points = net(train_dataset)

        loss, end_points = get_loss(end_points)
        loss.backward()
        if (batch_idx + 1) % 1 == 0:
            optimizer.step()
            optimizer.zero_grad()

        for key in end_points:
            if 'loss' in key or 'acc' in key or 'prec' in key or 'recall' in key or 'count' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_interval = 2
        if (batch_idx + 1) % batch_interval == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx + 1))
            for key in sorted(stat_dict.keys()):
                TRAIN_WRITER.add_scalar(key, stat_dict[key] / batch_interval,
                                        (EPOCH_CNT * len(TRAIN_DATALOADER) + batch_idx) * cfgs.batch_size)
                log_string('mean %s: %f' % (key, stat_dict[key] / batch_interval))
                stat_dict[key] = 0


# -------------------------------------------------------------------------


def eval_ep(scene_start, scene_end):
    TEST_DATASET = Text2GraspDataset(cfgs.dataset_root, scene_start, scene_end, camera=cfgs.camera,
                                     split='test_seen', num_points=cfgs.num_point, remove_outlier=True,
                                     augment=False)

    TRAIN_DATALOADER = DataLoader(TEST_DATASET, batch_size=cfgs.batch_size, shuffle=False,
                                  num_workers=4, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)
    stat_dict = {}

    net.eval()
    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        if batch_idx % 10 == 0:
            print('Eval batch: %d' % (batch_idx))
        for key in batch_data_label:
            if 'list' in key:
                for i in range(len(batch_data_label[key])):
                    for j in range(len(batch_data_label[key][i])):
                        batch_data_label[key][i][j] = batch_data_label[key][i][j].to(device)
            else:
                batch_data_label[key] = batch_data_label[key].to(device)

        with torch.no_grad():
            end_points = net(batch_data_label)

        loss, end_points = get_loss(end_points)

        for key in end_points:
            if 'loss' in key or 'acc' in key or 'prec' in key or 'recall' in key or 'count' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

    for key in sorted(stat_dict.keys()):
        TEST_WRITER.add_scalar(key, stat_dict[key] / float(batch_idx + 1),
                               (EPOCH_CNT + 1) * len(TEST_DATASET) * cfgs.batch_size)
        log_string('eval mean %s: %f' % (key, stat_dict[key] / (float(batch_idx + 1))))

    mean_loss = stat_dict['loss/overall_loss'] / float(batch_idx + 1)
    return mean_loss

if __name__ == '__main__':
    for epoch in range(start_epoch, cfgs.max_epoch):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % (epoch))
        log_string('Current learning rate: %f' % (get_current_lr(epoch)))
        log_string('Current BN decay momentum: %f' % (bnm_scheduler.lmbd(bnm_scheduler.last_epoch)))
        log_string(str(datetime.now()))

        for scene_train in [[90, 100]]:
            scene_start, scene_end = scene_train
            np.random.seed()
            train_ep(scene_start, scene_end)

        save_dict = {'epoch': epoch + 1,
                         'scene': scene_start,
                         'optimizer_state_dict': optimizer.state_dict(),

                         }

        try:
            save_dict['model_state_dict'] = net.module.state_dict()
        except:
            save_dict['model_state_dict'] = net.state_dict()
            torch.save(save_dict, os.path.join(cfgs.log_dir, 'checkpoint.tar'))
