""" Demo to show prediction results.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np

import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image
import spacy  # for tokenizer
import json
import torch
from graspnetAPI.graspnetAPI import GraspGroup
import time

spacy_eng = spacy.load("en_core_web_sm")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import random
from text2grasp import Text2Grasp, pred_decode
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default='', help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.1, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
cfgs = parser.parse_args()

class Vocabulary:
    def __init__(self, freq_threshold=5):
        # Initialize 2 dictionary: index to string and string to index
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}

        # Threshold for add word to dictionary
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, data_path):
        self.freq_threshold = 1
        frequencies = {}
        idx = 4
        num_object = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 28, 33, 35,
                      36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 69,
                      71, 72, 75]
        descriptions_path = open(os.path.join(data_path), 'r')
        descriptions_file = json.load(descriptions_path)

        for i, obj_id in enumerate(descriptions_file.keys()):
            sentence_list = descriptions_file[str(num_object[i])]
            for word in self.tokenizer_eng(sentence_list['name_object']):
                if word not in frequencies:
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

        self.create_vocab('/home/grasp/PycharmProjects/pythonProject/text_grasp/data/text_data/vocab.json')

    def read_vocab(self, file_name):
        """
        Load created vocabulary file and replace the 'index to string' and 'string to index' dictionary
        """
        vocab_path = open(file_name, 'r')
        vocab = json.load(vocab_path)
        new_itos = {int(key): value for key, value in vocab['itos'].items()}

        self.itos = new_itos
        self.stoi = vocab['stoi']

    def create_vocab(self, file_name):
        # create json object from dictionary
        vocab = json.dumps({'itos': self.itos,
                            'stoi': self.stoi})

        # open file for writing, "w"
        f = open(file_name, "w")

        # write json object to file
        f.write(vocab)

        # close file
        f.close()

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

def get_text_data(descript):
    vocab = Vocabulary(1)
    vocab.read_vocab('/home/grasp/Downloads/data/text_data/vocab.json')
    numericalized_descript = [vocab.stoi["<SOS>"]]
    numericalized_descript += vocab.numericalize(descript)
    numericalized_descript.append(vocab.stoi["<EOS>"])

    padding = [vocab.stoi["<PAD>"] for _ in range(50 - len(numericalized_descript))]
    numericalized_descript.extend(padding)

    return torch.tensor(numericalized_descript)

def get_net():
    # Init the model
    net = Text2Grasp(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net

def get_and_process_data(data_dir):
    # load data
    color = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
    workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
    meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
    intrinsic = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']
    # generate cloud
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # get valid points
    mask = (workspace_mask & (depth > 0))
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # sample points
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # convert data
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    # o3d.io.write_point_cloud('obj4.ply', cloud)
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled
    cloud_sampled = torch.from_numpy(np.concatenate((cloud_sampled.squeeze(0), color_sampled), axis=1))
    numericalized_descript = get_text_data("I need a quick snack to tide me over until dinner. Do we have any fruit that's filling and easy to eat?"
)
    # numericalized_descript = get_text_data("This screw is loose. Do we have a tool with a rotating handle that can help me tighten it?")
    end_points['text_input'] = torch.tensor(numericalized_descript).to(device).unsqueeze(0)
    end_points['point_clouds'] = cloud_sampled.to(device).unsqueeze(0)
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud

def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        start_time = time.time()
        end_points = net(end_points)
        print("--- %s seconds ---" % (time.time() - start_time))
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(np.array(gg_array))
    return gg

def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
    return gg

def vis_grasps(gg, cloud):
    gg.nms()
    gg.sort_by_score()
    gg = gg
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])

def demo(data_dir):
    net = get_net()
    end_points, cloud = get_and_process_data(data_dir)
    gg = get_grasps(net, end_points)
    if cfgs.collision_thresh > 0:
        gg = collision_detection(gg, np.array(cloud.points))
    vis_grasps(gg, cloud)


if __name__=='__main__':
    data_dir = ''
    demo(data_dir)

