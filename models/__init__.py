from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy

import numpy as np
import misc.utils as utils
import torch

from .ShowTellModel import ShowTellModel
from .FCModel import FCModel
from .OldModel import ShowAttendTellModel, AllImgModel
# from .Att2inModel import Att2inModel
from .AttModel import *

def setup(opt):
    
    if opt.caption_model == 'fc':
        model = FCModel(opt)
    if opt.caption_model == 'show_tell':
        model = ShowTellModel(opt)
    # Att2in model in self-critical
    elif opt.caption_model == 'att2in':
        model = Att2inModel(opt)
    # Att2in model with two-layer MLP img embedding and word embedding
    elif opt.caption_model == 'att2in2':
        model = Att2in2Model(opt)
    elif opt.caption_model == 'att2all2':
        model = Att2all2Model(opt)
    # Adaptive Attention model from Knowing when to look
    elif opt.caption_model == 'adaatt':
        model = AdaAttModel(opt)
    # Adaptive Attention with maxout lstm
    elif opt.caption_model == 'adaattmo':
        model = AdaAttMOModel(opt)
    # Top-down attention model
    elif opt.caption_model == 'topdown':
        model = TopDownModel(opt)
    # StackAtt
    elif opt.caption_model == 'stackatt':
        model = StackAttModel(opt)
    # DenseAtt
    elif opt.caption_model == 'denseatt':
        model = DenseAttModel(opt)
    # original paper model
    elif opt.caption_model == 'topdown_original':
        model = TopDownOriginalModel(opt)
    # topdown_up_cat_weighted_hidden
    elif opt.caption_model == 'topdown_up_cat_weighted_hidden_3':
        model = TopDownUpCatWeightedHiddenModel_3(opt)
    # topdown_up_cat_weighted_hidden
    elif opt.caption_model == 'testmodel':
        model = TestModel(opt)
    # bottomup
    elif opt.caption_model == 'bottomup':
        model = BottomUpModel(opt)

    else:
        raise Exception("Caption model not supported: {}".format(opt.caption_model))

    # check compatibility if training is continued from previously saved model
    if vars(opt).get('start_from', None) is not None:
        # check if all necessary files exist 
        assert os.path.isdir(opt.start_from)," %s must be a a path" % opt.start_from
        if opt.load_best:
            assert os.path.isfile(os.path.join(opt.start_from,
                                               "infos_" + opt.id + "-best.pkl")), "infos_-best.pkl file does not exist in path %s" % opt.start_from
            state_dict = torch.load(os.path.join(opt.start_from, 'model-best.pth'))
        else:
            assert os.path.isfile(os.path.join(opt.start_from,
                                               "infos_" + opt.id + ".pkl")), "infos.pkl file does not exist in path %s" % opt.start_from
            state_dict = torch.load(os.path.join(opt.start_from, 'model.pth'))

        if opt.att_normalize_method == '6-1-0':
            del state_dict['core.lang_lstm.weight_hh']
            del state_dict['core.lang_lstm.bias_hh']
            del state_dict['core.lang_lstm.weight_ih']
            del state_dict['core.lang_lstm.bias_ih']
            del state_dict['core.h2_affine.weight']
            del state_dict['core.h2_affine.bias']
            del state_dict['logit.weight']
            del state_dict['logit.bias']

        model.load_state_dict(state_dict)

    return model