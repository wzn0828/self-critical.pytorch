from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import os
import numpy as np
import random
import torch
import skimage
import skimage.io
import scipy.misc

from torchvision import transforms as trn

preprocess = trn.Compose([
    # trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

from misc.resnet_utils import myResnet
import misc.resnet


class DataLoaderTest():

    def __init__(self, opt):
        self.opt = opt
        self.input_fc_dir = opt.input_fc_dir
        self.input_att_dir = opt.input_att_dir
        self.batch_size = opt.batch_size
        self.seq_per_img = 1

        # load the json file which contains additional information about the dataset
        print('DataLoaderTest loading fc features from folder: ', self.input_fc_dir)
        print('DataLoaderTest loading att features from folder: ', self.input_att_dir)

        self.fc_files = []
        self.att_files = []
        self.ids = []

        # read in all the filenames from the folder
        print('listing all imageids in directory ' + self.input_fc_dir)

        n = 1
        for root, dirs, files in os.walk(self.input_fc_dir, topdown=False):
            for file in files:
                fullpath = os.path.join(self.input_fc_dir, file)
                self.fc_files.append(fullpath)
                att_fullpath = os.path.join(self.input_att_dir, file[:-1]+'z')
                self.att_files.append(att_fullpath)
                self.ids.append(str(n))  # just order them sequentially
                n = n + 1

        self.N = len(self.fc_files)
        print('DataLoaderRaw found ', self.N, ' images')

        self.iterator = 0

    def get_batch(self, batch_size=None):
        batch_size = batch_size or self.batch_size

        fc_batch = []
        att_batch = []
        max_index = self.N
        wrapped = False
        infos = []

        for i in range(batch_size):
            ri = self.iterator
            ri_next = ri + 1
            if ri_next >= max_index:
                ri_next = 0
                wrapped = True
                # wrap back around
            self.iterator = ri_next

            att_feat = np.load(self.att_files[ri])['feat']
            att_feat = att_feat.reshape(-1, att_feat.shape[-1])
            fc_feat = np.load(self.fc_files[ri])

            fc_batch.append(fc_feat)
            att_batch.append(att_feat)

            info_struct = {}
            info_struct['id'] = self.ids[ri]
            info_struct['att_file_path'] = self.att_files[ri]
            info_struct['fc_file_path'] = self.fc_files[ri]
            infos.append(info_struct)

        fc_batch, att_batch, infos = zip(*sorted(zip(fc_batch, att_batch, infos), key=lambda x: 0, reverse=True))

        data = {}
        data['fc_feats'] = np.stack(reduce(lambda x, y: x + y, [[_] for _ in fc_batch]))
        # merge att_feats
        max_att_len = max([_.shape[0] for _ in att_batch])
        data['att_feats'] = np.zeros([len(att_batch), max_att_len, att_batch[0].shape[1]], dtype='float32')
        for i in range(len(att_batch)):
            data['att_feats'][i:(i + 1), :att_batch[i].shape[0]] = att_batch[i]
        data['att_masks'] = np.zeros(data['att_feats'].shape[:2], dtype='float32')
        for i in range(len(att_batch)):
            data['att_masks'][i:(i + 1), :att_batch[i].shape[0]] = 1
        # set att_masks to None if attention features have same length
        if data['att_masks'].sum() == data['att_masks'].size:
            data['att_masks'] = None

        data['bounds'] = {'it_pos_now': self.iterator, 'it_max': self.N, 'wrapped': wrapped}
        data['infos'] = infos

        return data

    def reset_iterator(self, split):
        self.iterator = 0

    def get_vocab_size(self):
        return len(self.ix_to_word)

    def get_vocab(self):
        return self.ix_to_word


