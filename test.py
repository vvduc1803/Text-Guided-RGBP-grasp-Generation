""" Testing for GraspNet baseline model. """

import os
import sys
import numpy as np
import argparse
import time

import torch
from torch.utils.data import DataLoader
from graspnetAPI.graspnetAPI import GraspGroup, GraspNetEval

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from text2grasp import Text2Grasp, pred_decode
from dataset.text2grasp_dataset import Text2GraspDataset, collate_fn
from collision_detector import ModelFreeCollisionDetector

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='', help='Datasetroot')
parser.add_argument('--checkpoint_path', default='', help='Model checkpoint path')
parser.add_argument('--dump_dir', default='', help='Dump dir to save outputs')
parser.add_argument('--camera', default='kinect', help='Camera split [realsense/kinect]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during inference [default: 1]')
parser.add_argument('--collision_thresh', type=float, default=0.01,
                    help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01,
                    help='Voxel Size to process point clouds before collision detection [default: 0.01]')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers used in evaluation [default: 30]')
cfgs = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
if not os.path.exists(cfgs.dump_dir):
    os.mkdir(cfgs.dump_dir)


# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass

start_scene, end_scene = 90,91
# Create Dataset and Dataloader
TEST_DATASET = Text2GraspDataset(cfgs.dataset_root, start_scene, end_scene, test_split=False, split='test',
                               camera=cfgs.camera, num_points=cfgs.num_point, remove_outlier=True, augment=False,
                               load_label=False)


SCENE_LIST = TEST_DATASET.scene_list()

TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=cfgs.batch_size, shuffle=False,
                             num_workers=4, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)

# Init the model
net = Text2Grasp(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
               cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
# Load checkpoint
checkpoint = torch.load(cfgs.checkpoint_path)
net.load_state_dict(checkpoint['model_state_dict'])
start_epoch = checkpoint['epoch']
print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))


# ------------------------------------------------------------------------- GLOBAL CONFIG END

def inference():
    batch_interval = 100
    stat_dict = {}  # collect statistics
    # set model to eval mode (for bn and dp)
    net.eval()
    tic = time.time()
    for batch_idx, batch_data in enumerate(TEST_DATALOADER):
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(device)
            else:
                batch_data[key] = batch_data[key].to(device)

        # Forward pass
        with torch.no_grad():
            end_points = net(batch_data)
            grasp_preds = pred_decode(end_points)

        # Dump results for evaluation
        for i in range(cfgs.batch_size):
            data_idx = batch_idx * cfgs.batch_size + i
            preds = grasp_preds[i].detach().cpu().numpy()
            # print(preds)
            gg = GraspGroup(preds)
            obj_id, index_text, sceneId, img_num = TEST_DATASET.super_paths[data_idx].split('_')
            # collision detection
            # if cfgs.collision_thresh > 0:
            #     cloud, _ = TEST_DATASET.get_data(data_idx, sceneId, img_num, return_raw_cloud=True)
            #     mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
            #     collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
            #     print(np.sum(collision_mask))
            #     gg = gg[~collision_mask]

            # save grasps
            save_dir = os.path.join(cfgs.dump_dir, f'scene_{sceneId}', cfgs.camera, obj_id, index_text)
            save_path = os.path.join(save_dir, img_num + '.npy')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            gg.save_npy(save_path)

        if batch_idx % batch_interval == 0:
            toc = time.time()
            print('Eval batch: %d, time: %fs' % (batch_idx, (toc - tic) / batch_interval))
            tic = time.time()


def evaluate():
    root = cfgs.dump_dir
    list_scenes = os.listdir(root)
    dic = {'0.2': [],'0.4': [],'0.6': [],'0.8': [],'1.0': [],'1.2': [],'all': [],}
    for scene in list_scenes:
        if scene.startswith('scene_'):
            object_root = os.path.join(root, scene, cfgs.camera)
            object_ids = os.listdir(object_root)
            for object_id in object_ids:
                img_root = os.path.join(object_root, object_id, '0')
                img_nums = [i.split('.')[0] for i in os.listdir(img_root)]

                for img_num in img_nums:

                    ge = GraspNetEval(root=cfgs.dataset_root, camera=cfgs.camera, split='test')

                    values = ge.eval_seen(cfgs.dump_dir, scene_ids=scene, object_id=object_id, img_num=img_num, proc=cfgs.num_workers)
                    if values != None:
                        ap, ap_02, ap_04, ap_06, ap_08, ap_10, ap_12 = values
                        if ap > 0.1:

                            dic['0.2'].append(ap_02)
                            dic['0.4'].append(ap_04)
                            dic['0.6'].append(ap_06)
                            dic['0.8'].append(ap_08)
                            dic['1.0'].append(ap_10)
                            dic['1.2'].append(ap_12)
                            dic['all'].append(ap)

            for key in dic.keys():
                print(f'Results {key}:{sum(dic[key]) / len(dic[key])}')


if __name__ == '__main__':
    evaluate()
    # inference()
