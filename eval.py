from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
from dataloaderraw import *
import eval_utils
import argparse
import misc.utils as utils
import torch

# Input arguments and options
from test import DataLoaderTest

parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--model', type=str, default='',
                help='path to model to evaluate')
parser.add_argument('--cnn_model', type=str,  default='resnet101',
                help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str, default='',
                help='path to infos to evaluate')
# Basic options
parser.add_argument('--batch_size', type=int, default=0,
                help='if > 0 then overrule, otherwise load from checkpoint.')
parser.add_argument('--num_images', type=int, default=-1,
                help='how many images to use when periodically evaluating the loss? (-1 = all)')
parser.add_argument('--language_eval', type=int, default=0,
                help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
parser.add_argument('--dump_images', type=int, default=1,
                help='Dump images into vis/imgs folder for vis? (1=yes,0=no)')
parser.add_argument('--dump_json', type=int, default=1,
                help='Dump json with predictions into vis folder? (1=yes,0=no)')
parser.add_argument('--dump_path', type=int, default=0,
                help='Write image paths along with predictions into vis json? (1=yes,0=no)')
parser.add_argument('--ciderd', type=bool, default=False,
                help='whether produce cider score for each generated caption. If TURE, the split must in the MS COCO validation dataset')
# Sampling options
parser.add_argument('--sample_max', type=int, default=1,
                help='1 = sample argmax words. 0 = sample from distributions.')
parser.add_argument('--max_ppl', type=int, default=0,
                help='beam search by max perplexity or max probability.')
parser.add_argument('--beam_size', type=int, default=2,
                help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
parser.add_argument('--group_size', type=int, default=1,
                help='used for diverse beam search. if group_size is 1, then it\'s normal beam search')
parser.add_argument('--diversity_lambda', type=float, default=0.5,
                help='used for diverse beam search. Usually from 0.2 to 0.8. Higher value of lambda produces a more diverse list')
parser.add_argument('--temperature', type=float, default=1.0,
                help='temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
parser.add_argument('--decoding_constraint', type=int, default=0,
                help='If 1, not allowing same word in a row')
# For evaluation on a folder of images:
parser.add_argument('--image_folder', type=str, default='', 
                help='If this is nonempty then will predict on the images in this folder path')
parser.add_argument('--image_root', type=str, default='', 
                help='In case the image paths have to be preprended with a root path to an image folder')
# For evaluation on MSCOCO images from some split:
parser.add_argument('--input_fc_dir', type=str, default='',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_att_dir', type=str, default='',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_box_dir', type=str, default='',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_label_h5', type=str, default='',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_json', type=str, default='', 
                help='path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
parser.add_argument('--split', type=str, default='test', 
                help='if running on MSCOCO images, which split to use: val|test|train')
parser.add_argument('--coco_json', type=str, default='', 
                help='if nonempty then use this file in DataLoaderRaw (see docs there). Used only in MSCOCO test evaluation, where we have a specific json file of only test set images.')
# misc
parser.add_argument('--id', type=str, default='', 
                help='an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')
parser.add_argument('--verbose_beam', type=int, default=1, 
                help='if we need to print out all beam search beams.')
parser.add_argument('--verbose_loss', type=int, default=0, 
                help='if we need to calculate loss.')

opt = parser.parse_args()

# ----- for my local set ----- #
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
opt.dump_images = 0
opt.num_images = -1
opt.model = 'Experiments/Visualized/BUTD-XE_Noh2pre/model.pth'
opt.infos_path = 'Experiments/Visualized/BUTD-XE_Noh2pre/infos_BUTD-XE_Noh2pre.pkl'
opt.language_eval = 1
opt.beam_size = 1
opt.batch_size = 64

opt.input_feature = 'fc'
opt.pre = 'h'
opt.output_attention = False
opt.save_att_statics = True

opt.att_statics_numfeat_path = '/home/wzn/PycharmProjects/self-critical.pytorch/data/MSCOCO/train_att_statics_numboxes.npy'
opt.att_statics_weights_l2_path = '/home/wzn/PycharmProjects/self-critical.pytorch/data/MSCOCO/train_att_statics_l2weight_e3.npy'

opt.dim_att_statics_numfeat_path = '/home/wzn/PycharmProjects/self-critical.pytorch/data/MSCOCO/dim_train_att_statics_numboxes.npy'
opt.dim_att_statics_weights_l2_path = '/home/wzn/PycharmProjects/self-critical.pytorch/data/MSCOCO/dim_train_att_statics_l2weight_e3.npy'

opt.specified_id=[]
opt.dataset='coco'
opt.annfile = '/home/wzn/PycharmProjects/self-critical.pytorch/coco-caption/annotations/captions_val2014.json'
opt.input_json = 'data/MSCOCO/cocotalk.json'
opt.input_label_h5 = '/home/wzn/PycharmProjects/self-critical.pytorch/data/MSCOCO/cocotalk_label.h5'
opt.input_fc_dir = '/home/wzn/Datasets/ImageCaption/MSCOCO/detection_features/trainval_20-100/trainval_fc'
opt.input_att_dir = '/home/wzn/Datasets/ImageCaption/MSCOCO/detection_features/trainval_20-100/trainval_att'

# opt.print_lang_weights = True
opt.testOrnot = False

# opt.specified_id = [88652, 46775]
opt.split = 'train'   # 'specified'

cache_path = os.path.join('vis', str(opt.num_images))

# Load infos
with open(opt.infos_path) as f:
    infos = cPickle.load(f)

# ----- for my local set ----- #
opt.input_box_dir = infos['opt'].input_box_dir
# opt.input_label_h5 = infos['opt'].input_label_h5
# ----- for my local set ----- #


# override and collect parameters
if len(opt.input_fc_dir) == 0:
    opt.input_fc_dir = infos['opt'].input_fc_dir
    opt.input_att_dir = infos['opt'].input_att_dir
    opt.input_box_dir = infos['opt'].input_box_dir
    opt.input_label_h5 = infos['opt'].input_label_h5
if len(opt.input_json) == 0:
    opt.input_json = infos['opt'].input_json
if opt.batch_size == 0:
    opt.batch_size = infos['opt'].batch_size
if len(opt.id) == 0:
    opt.id = infos['opt'].id
ignore = ["id", "batch_size", "beam_size", "start_from", "language_eval", 'input_label_h5',"input_fc_dir", "input_att_dir", "input_json",'input_box_dir']
for k in vars(infos['opt']).keys():
    if k not in ignore:
        if k in vars(opt):
            assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
        else:
            vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

vocab = infos['vocab'] # ix -> word mapping

# Setup the model
model = models.setup(opt)
model.load_state_dict(torch.load(opt.model))
model.cuda()
# model = torch.nn.DataParallel(model)
model.eval()
crit = utils.LanguageModelCriterion()

# Create the Data Loader instance
if len(opt.image_folder) == 0:
    if opt.testOrnot:
        loader = DataLoaderTest(opt)
    else:
        loader = DataLoader(opt)
else:
    loader = DataLoaderRaw({'folder_path': opt.image_folder,
                            'coco_json': opt.coco_json,
                            'batch_size': opt.batch_size,
                            'cnn_model': opt.cnn_model})
# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
loader.ix_to_word = infos['vocab']

assert opt.annfile is not None and len(opt.annfile) > 0
# Set sample options
loss, split_predictions, lang_stats = eval_utils.eval_split(model, crit, loader,
    vars(opt))

if opt.save_att_statics:
    # json.dump(model.core.att_statics_numfeat, open(opt.att_statics_numfeat_path, 'w'))
    # json.dump(model.core.att_statics_weights_l2, open(opt.att_statics_weights_l2_path, 'w'))

    np.save(model.core.att_statics_numfeat_path, model.core.att_statics_numfeat)
    np.save(model.core.att_statics_weights_l2_path, model.core.att_statics_weights_l2)

    # json.dump(model.core.dim_att_statics_numboxes, open(opt.dim_att_statics_numfeat_path, 'w'))
    # json.dump(model.core.dim_att_statics_l2weight, open(opt.dim_att_statics_weights_l2_path, 'w'))

    np.save(model.core.dim_att_statics_numfeat_path, model.core.dim_att_statics_numboxes)
    np.save(model.core.dim_att_statics_weights_l2_path, model.core.dim_att_statics_l2weight)


print('loss: ', loss)
if lang_stats:
  print(lang_stats)

if opt.dump_json == 1:
    # dump the json
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
    if opt.split == None:
        split = 'test'
    elif opt.split == 'raw_val':
        split = 'val'
    else:
        split = opt.split
    cache_path = os.path.join(cache_path, 'captions_%s2014_LanguageAttention_results'%split + '.json')
    json.dump(split_predictions, open(cache_path, 'w'))
