"""
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import argparse
import importlib
import os
from os.path import join as ospj
import shutil
import warnings

from util import Logger

_DATASET_NAMES = ('CUB', 'ILSVRC')
_ARCHITECTURE_NAMES = ('vgg_MNPO', 'vgg_MNPOs')
_METHOD_NAMES = ('cam', 'adl', 'acol', 'spg', 'has', 'cutmix')
_SPLITS = ('train', 'val', 'test')


def box_v2_metric(args):
    if args.box_v2_metric:
        args.multi_contour_eval = True
        args.multi_iou_eval = True
    else:
        args.multi_contour_eval = False
        args.multi_iou_eval = False
        warnings.warn("MaxBoxAcc metric is deprecated.")
        warnings.warn("Use MaxBoxAccV2 by setting args.box_v2_metric to True.")


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_architecture_type(wsol_method):
    if wsol_method in ('has', 'cutmix'):
        architecture_type = 'cam'
    else:
        architecture_type = wsol_method
    return architecture_type

def configure_data_paths(args):
    if args.dataset_name == 'ILSVRC':
        train = val = test = '/opt/data/private/dataset/ILSVRC2015/Data/CLS-LOC/'
    elif args.dataset_name == 'CUB':
        train = val = test = '/opt/data/private/dataset/CUB_200_2011/images/'
    data_paths = {'train':train, 'val':val, 'test':test}
    return data_paths


def configure_log_folder(args):
    log_folder = ospj('output', args.experiment_name)
    
    if os.path.exists(log_folder) is False:
        os.makedirs(log_folder)
    return log_folder

def configure_log(args):
    log_file_name = ospj(args.log_folder, 'log.log')
    Logger(log_file_name)


def configure_reporter(args):
    reporter = importlib.import_module('util').Reporter
    reporter_log_root = ospj(args.log_folder, 'reports')
    if not os.path.isdir(reporter_log_root):
        os.makedirs(reporter_log_root)
    return reporter, reporter_log_root

def configure_pretrained_path(args):
    pretrained_path = None
    return pretrained_path


def check_dependency(args):
    if args.dataset_name == 'CUB':
        if args.num_val_sample_per_class >= 6:
            raise ValueError("num-val-sample must be <= 5 for CUB.")
    if args.dataset_name == 'OpenImages':
        if args.num_val_sample_per_class >= 26:
            raise ValueError("num-val-sample must be <= 25 for OpenImages.")

def get_configs():
    parser = argparse.ArgumentParser()

    # Util
    parser.add_argument('--seed', type=int)
    parser.add_argument('--experiment_name', type=str, default='test_case')
    parser.add_argument('--workers', default=10, type=int,
                        help='number of data loading workers (default: 4)')

    # Data
    parser.add_argument('--dataset_name', type=str, default='ILSVRC',
                        choices=_DATASET_NAMES)
    
    parser.add_argument('--metadata_root', type=str, default='metadata/')
    parser.add_argument('--proxy_training_set', type=str2bool, nargs='?',
                        const=True, default=False,
                        help='Efficient hyper_parameter search with a proxy '
                             'training set.')
    parser.add_argument('--num_val_sample_per_class', type=int, default=0,
                        help='Number of full_supervision validation sample per '
                             'class. 0 means "use all available samples".')

    # Setting
    parser.add_argument('--architecture', default='vgg_MNPOs',
                        choices=_ARCHITECTURE_NAMES,
                        help='model architecture: ' +
                             ' | '.join(_ARCHITECTURE_NAMES) +
                             ' (default: resnet50)')
    parser.add_argument('--isgrad', type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument("--loc_metric_list", default=['gt-known','top-1','top-5'], type=str)#
    parser.add_argument('--epochs', default=10, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--pretrained', type=str2bool, nargs='?',
                        const=True, default=True,
                        help='Use pre_trained model.')
    parser.add_argument('--resize_size', type=int, default=256,
                        help='input resize size')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='input crop size')
    parser.add_argument('--iou_threshold', type=int, default=50)
    parser.add_argument('--eval_checkpoint_type', type=str, default='best',
                        choices=('best', 'last'))
    parser.add_argument('--box_v2_metric', type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument('--main_layer', type=str, nargs='?',
                        const=True, default='conv5_2')
    parser.add_argument('--branch_layer', type=str, nargs='?',
                        const=True, default='branch_fea')

    # Common hyperparameters
    parser.add_argument('--batch_size', default=100, type=int,
                        help='Mini-batch size (default: 256), this is the total'
                             'batch size of all GPUs on the current node when'
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', default=0.0008, type=float,
                        help='initial learning rate', dest='lr')
    parser.add_argument('--lr_decay_frequency', type=int, default=4,
                        help='How frequently do we decay the learning rate?')
    parser.add_argument('--lr_classifier_ratio', type=float, default=15,
                        help='Multiplicative factor on the classifier layer.')#？？？
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--large_feature_map', type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument('--istencrop', type=str2bool, nargs='?',
                        const=True, default=True)

    # Method-specific hyperparameters
    parser.add_argument('--wsol_method', type=str, default='cam',
                        choices=_METHOD_NAMES)
    parser.add_argument('--has_grid_size', type=int, default=4)
    parser.add_argument('--has_drop_rate', type=float, default=0.5)
    parser.add_argument('--acol_threshold', type=float, default=0.7)
    parser.add_argument('--spg_threshold_1h', type=float, default=0.7,
                        help='SPG threshold')
    parser.add_argument('--spg_threshold_1l', type=float, default=0.01,
                        help='SPG threshold')
    parser.add_argument('--spg_threshold_2h', type=float, default=0.5,
                        help='SPG threshold')
    parser.add_argument('--spg_threshold_2l', type=float, default=0.05,
                        help='SPG threshold')
    parser.add_argument('--spg_threshold_3h', type=float, default=0.7,
                        help='SPG threshold')
    parser.add_argument('--spg_threshold_3l', type=float, default=0.1,
                        help='SPG threshold')
    parser.add_argument('--adl_drop_rate', type=float, default=0.75,
                        help='ADL dropout rate')
    parser.add_argument('--adl_threshold', type=float, default=0.9,
                        help='ADL gamma, threshold ratio '
                             'to maximum value of attention map')
    parser.add_argument('--cutmix_beta', type=float, default=1.0,
                        help='CutMix beta')
    parser.add_argument('--cutmix_prob', type=float, default=1.0,
                        help='CutMix Mixing Probability')

    args = parser.parse_args()

    check_dependency(args)
    args.log_folder = configure_log_folder(args)
    configure_log(args)
    box_v2_metric(args)

    args.architecture_type = get_architecture_type(args.wsol_method)
    args.data_paths = configure_data_paths(args)
    args.metadata_root = ospj(args.metadata_root, args.dataset_name)
    args.reporter, args.reporter_log_root = configure_reporter(args)
    args.pretrained_path = configure_pretrained_path(args)
    args.spg_thresholds = ((args.spg_threshold_1h, args.spg_threshold_1l),
                           (args.spg_threshold_2h, args.spg_threshold_2l),
                           (args.spg_threshold_3h, args.spg_threshold_3l))

    return args
