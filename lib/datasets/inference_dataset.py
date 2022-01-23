# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np

import torch
from torch.nn import functional as F
from PIL import Image

from .base_dataset import BaseDataset


class ADE20K(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=150,
                 multi_scale=True,
                 flip=True,
                 ignore_label=-1,
                 base_size=520,
                 crop_size=(520, 520),
                 downsample_rate=1,
                 scale_factor=11,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        super(ADE20K, self).__init__(ignore_label, base_size,
                                  crop_size, downsample_rate, scale_factor, mean, std)

        self.root = root
        self.num_classes = num_classes
        self.list_path = list_path
        self.class_weights = None

        self.multi_scale = multi_scale
        self.flip = flip
        self.img_list = [line.strip().split() for line in open(root+list_path)]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

    def read_files(self):
        files = []
        for item in self.img_list:
            image_path, label_path = item
            name = os.path.splitext(os.path.basename(label_path))[0]
            sample = {
                'img': image_path,
                'label': label_path,
                'name': name
            }
            files.append(sample)
        return files

    def resize_image(self, image, label, size):
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        # image_path = os.path.join(self.root, 'ade20k', item['img'])
        # label_path = os.path.join(self.root, 'ade20k', item['label'])
        image_path = os.path.join(self.root, item['img'])
        label_path = os.path.join(self.root, item['label'])
        image = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR
        )
        try:
            size = image.shape
        except:
            return np.zeros((3,1080,1920)), 0, np.array([3,1080,1920]), 'default'
        image = self.input_transform(image)
        image = image.transpose((2, 0, 1))

        return image.copy(), 0, np.array(size), name