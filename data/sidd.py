import os
import glob

import numpy as np
import torch
import torch.utils.data as data
from torchvision.transforms.functional import to_tensor
from torchvision import datasets

from data.common import augment, is_image_file, load_img
from data.patchdata import PatchData
from utils.wavelet import swt2d

class SIDD1(PatchData):
    def __init__(self, args, name='sidd1', mode='train', benchmark=False):
        super(SIDD1, self).__init__(
            args, name=name, mode=mode, benchmark=benchmark
        )

    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        names_lr = sorted(
            glob.glob(os.path.join(self.dir_lr, '*' + self.ext[1]))
        )

        return names_hr, names_lr

    def _set_filesystem(self, data_dir):
        super(SIDD1, self)._set_filesystem(data_dir)

        self.dir_hr = os.path.join(self.apath, 'HIGH')
        self.dir_lr = os.path.join(self.apath, 'LOW')
        self.ext = ('.tiff', '.tiff')

class SIDD2(PatchData):
    def __init__(self, args, name='sidd2', mode='train', benchmark=False):
        super(SIDD2, self).__init__(
            args, name=name, mode=mode, benchmark=benchmark
        )

    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        names_lr = sorted(
            glob.glob(os.path.join(self.dir_lr, '*' + self.ext[1]))
        )

        return names_hr, names_lr

    def _set_filesystem(self, data_dir):
        super(SIDD2, self)._set_filesystem(data_dir)

        self.dir_hr = os.path.join(self.apath, 'HIGH')
        self.dir_lr = os.path.join(self.apath, 'LOW')
        self.ext = ('.png', '.png')
