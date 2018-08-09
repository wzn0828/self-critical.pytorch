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
    # Top-down-sentinal attention model s
    elif opt.caption_model == 'topdown_sentinal':
        model = TopDownSentinalModel(opt)
    elif opt.caption_model == 'topdown_sentinal_affine':
        model = TopDownSentinalAffineModel(opt)
    elif opt.caption_model == 'topdown_sentinal_affine2':
        model = TopDownSentinalAffine2Model(opt)

    # Top-down-cat-sentinal model s
    elif opt.caption_model == 'topdown_cat_sentinal':
        model = TopDownCatSentinalModel(opt)
    elif opt.caption_model == 'topdown_cat_sentinal_affine':
        model = TopDownCatSentinalAffineModel(opt)
    elif opt.caption_model == 'topdown_cat_sentinal_affine2':
        model = TopDownCatSentinalAffine2Model(opt)

    # Top-down-recurrent-sentinal model s
    elif opt.caption_model == 'topdown_recurrent_sentinal':
        model = TopDownRecurrentSentinalModel(opt)
    elif opt.caption_model == 'topdown_recurrent_sentinal_affine':
        model = TopDownRecurrentSentinalAffineModel(opt)
    elif opt.caption_model == 'topdown_recurrent_sentinal_affine2':
        model = TopDownRecurrentSentinalAffine2Model(opt)

    # top-down-average-sentinal model
    elif opt.caption_model == 'topdown_average_sentinal':
        model = TopDownAverageSentinalModel(opt)

    # top-down-weighted-sentinal model
    elif opt.caption_model == 'topdown_weighted_sentinal':
        model = TopDownWeightedSentinalModel(opt)

    elif opt.caption_model == 'topdown_weighted_sentinal_tanh_att':
        model = TopDownWeightedSentinalModel_tanh_att(opt)

    # top-down-weighted2-sentinal model
    elif opt.caption_model == 'topdown_weighted2_sentinal':
        model = TopDownWeighted2SentinalModel(opt)

    # Top-down-cat-recurrent-sentinal model s
    elif opt.caption_model == 'topdown_cat_recurrent_sentinal':
        model = TopDownCatRecurrentSentinalModel(opt)
    elif opt.caption_model == 'topdown_cat_recurrent_sentinal_affine':
        model = TopDownCatRecurrentSentinalAffineModel(opt)
    elif opt.caption_model == 'topdown_cat_recurrent_sentinal_affine2':
        model = TopDownCatRecurrentSentinalAffine2Model(opt)

    # original paper model
    elif opt.caption_model == 'topdown_original':
        model = TopDownOriginalModel(opt)

    elif opt.caption_model == 'topdown_original_weighted_sentinel':
        model = TopDownOriginalWeightedSentinelModel(opt)

    # top-down-cat-weighted-sentinal model
    elif opt.caption_model == 'topdown_cat_weighted_sentinal':
        model = TopDownCatWeightedSentinalModel(opt)


    # topdown_cat_weighted_sentinal_base_attfeat
    elif opt.caption_model == 'topdown_cat_weighted_sentinal_base_attfeat':
        model = TopDownCatWeightedSentinalBaseAttModel(opt)


    # topdown_up_cat_weighted_sentinal
    elif opt.caption_model == 'topdown_up_cat_weighted_sentinal':
        model = TopDownUpCatWeightedSentinalModel(opt)


    # topdown_up_add_weighted_sentinal
    elif opt.caption_model == 'topdown_up_add_weighted_sentinal':
        model = TopDownUpAddWeightedSentinalModel(opt)


    # topdown_up_cat_weighted_sentinal
    elif opt.caption_model == 'topdown_up_cat_weighted_sentinal_base_attfeat':
        model = TopDownUpCatWeightedSentinalBaseAttModel(opt)


    # topdown_up_cat_weighted_sentinal
    elif opt.caption_model == 'topdown_up_add_weighted_sentinal_base_attfeat':
        model = TopDownUpAddWeightedSentinalBaseAttModel(opt)


    # base the original paper model, add a affine to att_feature
    elif opt.caption_model == 'topdown_original_2':
        model = TopDownOriginal2Model(opt)

    elif opt.caption_model == 'topdown_recurrent_hidden':
        model = TopDownRecurrentHiddenModel(opt)


    # top-down-weighted-hidden model
    elif opt.caption_model == 'topdown_weighted_hidden':
        model = TopDownWeightedHiddenModel(opt)

    # top-down-cat-weighted-hidden model
    elif opt.caption_model == 'topdown_cat_weighted_hidden':
        model = TopDownCatWeightedHiddenModel(opt)

    # topdown_up_cat_weighted_hidden
    elif opt.caption_model == 'topdown_up_cat_weighted_hidden':
        model = TopDownUpCatWeightedHiddenModel(opt)


    # topdown_up_cat_weighted_hidden
    elif opt.caption_model == 'topdown_up_cat_weighted_hidden_1':
        model = TopDownUpCatWeightedHiddenModel_1(opt)
    # topdown_up_cat_weighted_hidden
    elif opt.caption_model == 'topdown_up_cat_weighted_hidden_2':
        model = TopDownUpCatWeightedHiddenModel_2(opt)
    # topdown_up_cat_weighted_hidden
    elif opt.caption_model == 'topdown_up_cat_weighted_hidden_3':
        model = TopDownUpCatWeightedHiddenModel_3(opt)
    # topdown_up_cat_weighted_hidden
    elif opt.caption_model == 'topdown_up_cat_weighted_hidden_4':
        model = TopDownUpCatWeightedHiddenModel_4(opt)


    # topdown_up_cat_average_hidden
    elif opt.caption_model == 'topdown_up_cat_average_hidden':
        model = TopDownUpCatAverageHiddenModel(opt)



    # topdown_2layer_up_cat_weighted_hidden
    elif opt.caption_model == 'topdown_2layer_up_cat_weighted_hidden':
        model = TopDown2LayerUpCatWeightedHiddenModel(opt)


    # topdown_attlayer_up_cat_weighted_hidden
    elif opt.caption_model == 'topdown_attlayer_up_cat_weighted_hidden':
        model = TopDownAttLayerUpCatWeightedHiddenModel(opt)

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
            model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model-best.pth')))
        else:
            assert os.path.isfile(os.path.join(opt.start_from,
                                               "infos_" + opt.id + ".pkl")), "infos.pkl file does not exist in path %s" % opt.start_from
            model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model.pth')))

    return model