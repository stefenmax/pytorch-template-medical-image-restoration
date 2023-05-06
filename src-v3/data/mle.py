import os
from data import srdata

import numpy as np
from data import common
from scipy.io import loadmat
import pickle

class MLE(srdata.SRData):
    def __init__(self, args, name='MLE', train=True, benchmark=False):
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]
        self.begin, self.end = list(map(lambda x: int(x), data_range))
        super(MLE, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr, names_edge, names_lr  = super(DIV2K, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_edge = names_edge[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        return names_hr, names_edge, names_lr, 

    def _set_filesystem(self, dir_data):
        super(AAPM, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'High_dose')
        self.dir_edge = os.path.join(self.apath, 'img_edge')
        self.dir_lr = os.path.join(self.apath, 'Low_dose')
        if self.input_large: self.dir_lr += 'L'
        if self.args.using_npy:
            self.ext = ('.npy', '.npy')
        if self.args.using_mat:
            self.ext = ('.mat', '.mat')
        
    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        lr, hr = common.set_channel(lr, hr, n_channels=self.args.n_colors)
        lr, hr = self.get_patch(lr, hr)
        lr_tensor, hr_tensor = common.np2Tensor(lr, hr, rgb_range=self.args.rgb_range)
        return lr_tensor, hr_tensor, filename

    
    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[self.idx_scale][idx]
        filename, _ = os.path.splitext(os.path.basename(f_hr))
        if self.args.ext == 'img' and self.args.using_npy:
            hr = np.load(f_hr)
            lr = np.load(f_lr)
        elif self.args.ext == 'img' and self.args.using_mat:
            lr = loadmat(f_lr)
            lr = lr['label']  # Should change it according to your own dataset
            hr = loadmat(f_hr)
            hr = hr['label']
        elif self.args.ext.find('sep') >= 0:
            with open(f_hr, 'rb') as _f:
                hr = pickle.load(_f)
            with open(f_lr, 'rb') as _f:
                lr = pickle.load(_f)

        return lr, hr, filename