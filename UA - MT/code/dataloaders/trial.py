import os
import argparse
import torch
# from networks.modified_vnet import VNet
# from ..test_util import test_single_case

import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm

np.bool = np.bool_

def test_all_case(image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True, test_save_path=None, preproc_fn=None):
    # total_metric = 0.0
    for image_path in tqdm(image_list):
        id = image_path.split('/')[-1]
        h5f = h5py.File(image_path, 'r')
        print(image_path, id, h5f)
        image = h5f['image'][:]
        label = h5f['label'][:]
        print(image.shape, label.shape, image, label)
        # if preproc_fn is not None:
        #     image = preproc_fn(image)
        # prediction, score_map = test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)

    #     if np.sum(prediction)==0:
    #         single_metric = (0,0,0,0)
    #     else:
    #         single_metric = calculate_metric_percase(prediction, label[:])
    #     total_metric += np.asarray(single_metric)

    #     if save_result:
    #         nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path + id + "_pred.nii.gz")
    #         nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + id + "_img.nii.gz")
    #         nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + id + "_gt.nii.gz")
    # avg_metric = total_metric / len(image_list)
    # print('average metric is {}'.format(avg_metric))

    return None

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/2018LA_Seg_Training Set/', help='Name of Experiment')
# parser.add_argument('--model', type=str,  default='vnet_supervisedonly_dp', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
# snapshot_path = "../model/"+FLAGS.model+"/"
# test_save_path = "../model/prediction/"+FLAGS.model+"_post/"
# if not os.path.exists(test_save_path):
#     os.makedirs(test_save_path)

num_classes = 2

with open(FLAGS.root_path + '/../test.list', 'r') as f:
    image_list = f.readlines()
image_list = [FLAGS.root_path +item.replace('\n', '')+"/mri_norm2.h5" for item in image_list]


def test_calculate_metric(epoch_num):
    # net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=False).cuda()
    # save_mode_path = os.path.join(snapshot_path, 'iter_' + str(epoch_num) + '.pth')
    # net.load_state_dict(torch.load(save_mode_path))
    # print("init weight from {}".format(save_mode_path))
    # net.eval()

    avg_metric = test_all_case(image_list, num_classes=num_classes,
                               patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                               save_result=True)

    return avg_metric


if __name__ == '__main__':
    # for iteration in range(1000, 6001, 1000):
    metric = test_calculate_metric(6000)
    print(metric)