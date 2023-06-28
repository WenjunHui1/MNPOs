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

import os
import pickle
import random
import torch
import torch.nn as nn
import torch.optim
import cv2
import numpy as np

from config import get_configs
from data_loaders import get_data_loader, configure_metadata
from util import string_contains_any
import wsol
import wsol.method
from PIL import Image
from torchvision import transforms
import time
from inference import CAMComputer
from tqdm import *
from collections import OrderedDict

import torch.nn.functional as F

_IMAGENET_MEAN = [0.485, .456, .406]
_IMAGENET_STDDEV = [.229, .224, .225]
_RESIZE_LENGTH = 224

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "aux" not in name)/1e6

def normalize_scoremap(cam):
    """
    Args:
        cam: numpy.ndarray(size=(H, W), dtype=np.float)
    Returns:
        numpy.ndarray(size=(H, W), dtype=np.float) between 0 and 1.
        If input array is constant, a zero-array is returned.
    """
    if np.isnan(cam).any():
        return np.zeros_like(cam)
    if cam.min() == cam.max():
        return np.zeros_like(cam)
    cam -= cam.min()
    cam /= cam.max()
    return cam

#设置随机种子.该代码设置的seed为None
def set_random_seed(seed):
    if seed is None:
        return
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

#性能表
class PerformanceMeter(object):
    def __init__(self, split, higher_is_better=True):
        self.best_function = max if higher_is_better else min
                                                              

        self.current_value = None
        self.best_value = None
        self.best_epoch = None
        self.value_per_epoch =  [] \
            if split == 'val' else [-np.inf if higher_is_better else np.inf]#？？？

    def update(self, new_value):
        self.value_per_epoch.append(new_value)
        self.current_value = self.value_per_epoch[-1]
        self.best_value = self.best_function(self.value_per_epoch)
        self.best_epoch = self.value_per_epoch.index(self.best_value)


class Trainer(object):
    _CHECKPOINT_NAME_TEMPLATE = '{}_checkpoint.pth.tar'
    _SPLITS = ('train', 'val')
    _EVAL_METRICS = ['loss', 'classification', 'localization']
    _BEST_CRITERION_METRIC = 'localization'
    _NUM_CLASSES_MAPPING = {
        "CUB": 200,
        "ILSVRC": 1000,
    }
    _FEATURE_PARAM_LAYER_PATTERNS = {       
        'vgg': ['features.'],
        'vgg_2branch': ['branch1_layer'],
    }

    def __init__(self):
        self.args = get_configs()

        set_random_seed(self.args.seed)
        print(self.args)
        self.performance_meters = self._set_performance_meters()
        self.reporter = self.args.reporter
        self.model = self._set_model()
        self.cross_entropy_loss = nn.CrossEntropyLoss().cuda()
        self.optimizer = self._set_optimizer() 
        self.loaders = get_data_loader(         
            data_roots=self.args.data_paths,
            metadata_root=self.args.metadata_root,
            batch_size=self.args.batch_size,
            workers=self.args.workers,
            resize_size=self.args.resize_size,
            crop_size=self.args.crop_size,
            proxy_training_set=self.args.proxy_training_set,
            num_val_sample_per_class=self.args.num_val_sample_per_class)

    
    def _set_performance_meters(self):
        self._EVAL_METRICS += ['{}_loc'.format(metric)
                               for metric in self.args.loc_metric_list]

        eval_dict = {                                                    
            split: {
                metric: PerformanceMeter(split,
                                         higher_is_better=False
                                         if metric in ('loss','localization','gt-known_loc','top-1_loc','top-5_loc')  else True)
                for metric in self._EVAL_METRICS 
            }
            for split in self._SPLITS
        }
        return eval_dict

    
    def _set_model(self):
        GPUID = '0'

        os.environ["CUDA_VISIBLE_DEVICES"] = GPUID
        num_classes = self._NUM_CLASSES_MAPPING[self.args.dataset_name]
        print("Loading model {}".format(self.args.architecture))
        model = wsol.__dict__[self.args.architecture](                 
            dataset_name=self.args.dataset_name,
            architecture_type=self.args.architecture_type,
            pretrained=self.args.pretrained,
            num_classes=num_classes,
            large_feature_map=self.args.large_feature_map,
            pretrained_path=self.args.pretrained_path,
            adl_drop_rate=self.args.adl_drop_rate,
            adl_drop_threshold=self.args.adl_threshold,
            acol_drop_threshold=self.args.acol_threshold,
            main_layer = self.args.main_layer,
            branch_layer = self.args.branch_layer)          
        model = model.cuda()
        print('Total Params = %.2fMB' % count_parameters_in_MB(model))
        return model

    def _set_optimizer(self):
        param_features = []
        param_classifiers = []

        def param_features_substring_list(architecture):
            for key in self._FEATURE_PARAM_LAYER_PATTERNS:
                if architecture.startswith(key):
                    return self._FEATURE_PARAM_LAYER_PATTERNS[key]
            raise KeyError("Fail to recognize the architecture {}"
                           .format(self.args.architecture))

        #dict consists of the revised layer,set different lr
        for name, parameter in self.model.named_parameters():
            if string_contains_any(
                    name,
                    param_features_substring_list(self.args.architecture)):
                if self.args.architecture in ('vgg16'):
                    param_features.append(parameter)
                elif self.args.architecture in ('vgg_2branch'):
                    param_classifiers.append(parameter)
            else:
                if self.args.architecture in ('vgg16'):
                    param_classifiers.append(parameter)
                elif self.args.architecture in ('vgg_2branch'):
                    param_features.append(parameter)

        optimizer = torch.optim.SGD([                
            {'params': param_features, 'lr': self.args.lr},
            {'params': param_classifiers,
             'lr': self.args.lr * self.args.lr_classifier_ratio}],
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
            nesterov=True)
        return optimizer
        
    def _wsol_training(self, images, target):
        output_dict = self.model(images, target)
        logits = output_dict['logits1']
        loss =  self.cross_entropy_loss(output_dict['logits1'], target) + \
                self.cross_entropy_loss(output_dict['logits2'], target)
        return logits, loss

    def train(self, split):
        torch.cuda.empty_cache()
        self.model.train()

        loader = self.loaders[split]

        total_loss = 0.0
        num_correct = 0
        num_images = 0

        for images, targets,_ in tqdm(loader):
            images = images.cuda()
            targets = targets.cuda()
            
            logits, loss = self._wsol_training(images, targets)
            logits = logits.argmax(dim=1)

            total_loss += float(loss.item()) * images.size(0)
            num_correct += (logits == targets).sum().item()
            num_images += float(images.size(0))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        loss_average = total_loss / float(num_images)
        classification_acc = num_correct / float(num_images) * 100
        print('loss.requires_grad:',loss.requires_grad)

        self.performance_meters[split]['classification'].update(
            classification_acc)
        self.performance_meters[split]['loss'].update(loss_average)

        return dict(classification_acc=classification_acc,
                    loss=loss_average)

    def print_performances(self):
        for split in self._SPLITS:
            for metric in self._EVAL_METRICS:
                current_performance = self.performance_meters[split][metric].current_value
                if current_performance is not None:
                    print("Split {}, metric {}, current value: {}".format(
                        split, metric, current_performance))
                    if split != 'test':
                        print("Split {}, metric {}, best value: {}".format(
                            split, metric,
                            self.performance_meters[split][metric].best_value))
                        print("Split {}, metric {}, best epoch: {}".format(
                            split, metric,
                            self.performance_meters[split][metric].best_epoch))

    def save_performances(self):
        log_path = os.path.join(self.args.log_folder, 'performance_log.pickle')
        with open(log_path, 'wb') as f:
            pickle.dump(self.performance_meters, f)

    def _compute_accuarcy(self, loader):
        num_correct = 0
       
        num_images = 0
        for images, targets, image_ids in tqdm(loader):
            targets = targets.cuda()
            
            images = images.cuda()
            output_dict = self.model(images, targets)
            
            pred = output_dict['logits1'].argmax(dim=1)
            

            num_correct += (pred == targets).sum().item()
            
            num_images += images.size(0)

        classification_accuracy = num_correct / float(num_images) * 100
        
        return classification_accuracy

    def evaluate(self, epoch, split):
       
        torch.cuda.empty_cache()
        print("Evaluate epoch {}, split {}".format(epoch, split))
        self.model.eval()
       
        cam_computer = CAMComputer(model=self.model,
                                        loader_cls=self.loaders['test'],
                                        loader_loc=self.loaders[split],
                                        metadata_root=os.path.join(self.args.metadata_root, split),
                                        iou_threshold=self.args.iou_threshold,
                                        dataset_name=self.args.dataset_name,
                                        split=split,
                                        loc_metric_list=self.args.loc_metric_list,
                                        isgrad=self.args.isgrad,
                                        istencrop=self.args.istencrop)

        cam_performance = cam_computer.compute_and_evaluate_cams()
        loc_score = cam_performance[self.args.loc_metric_list.index('gt-known')]
        self.performance_meters[split]['localization'].update(loc_score)#定位得分更新
            
        for idx, loc_metric in enumerate(self.args.loc_metric_list):
                self.performance_meters[split]['{}_loc'.format(loc_metric)].update(cam_performance[idx])
       

    def _torch_save_model(self, filename, epoch):
        torch.save({'architecture': self.args.architecture,
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()},
                   os.path.join(self.args.log_folder, filename))

    def save_checkpoint(self, epoch, split):
        if (self.performance_meters[split][self._BEST_CRITERION_METRIC]
                .best_epoch) == epoch:
            self._torch_save_model(
                self._CHECKPOINT_NAME_TEMPLATE.format('best'), epoch)
        if self.args.epochs == epoch:
            self._torch_save_model(
                self._CHECKPOINT_NAME_TEMPLATE.format('last'), epoch)
        self._torch_save_model(self._CHECKPOINT_NAME_TEMPLATE.format('epoch' + str(epoch)), epoch)

    def report_train(self, train_performance, epoch, split='train'):
        reporter_instance = self.reporter(self.args.reporter_log_root, epoch)
        reporter_instance.add(
            key='{split}/classification'.format(split=split),
            val=train_performance['classification_acc'])
        reporter_instance.add(
            key='{split}/loss'.format(split=split),
            val=train_performance['loss'])
        reporter_instance.write()

    #报告性能
    def report(self, epoch, split):
        reporter_instance = self.reporter(self.args.reporter_log_root, epoch)
        for metric in self._EVAL_METRICS:
            reporter_instance.add(
                key='{split}/{metric}'
                    .format(split=split, metric=metric),
                val=self.performance_meters[split][metric].current_value)
            reporter_instance.add(
                key='{split}/{metric}_best'
                    .format(split=split, metric=metric),
                val=self.performance_meters[split][metric].best_value)
        reporter_instance.write()

    def adjust_learning_rate(self, epoch):
        if self.args.dataset_name == "ILSVRC":
            if epoch != 0 and epoch % self.args.lr_decay_frequency == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.1
        else:
            if epoch == 75 or epoch == 100:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.1
        for param_group in self.optimizer.param_groups:
            print("learning rate is: {}".format(param_group['lr']))
        

    def load_checkpoint(self, checkpoint_type):
        if checkpoint_type not in ('best', 'last'):
            raise ValueError("checkpoint_type must be either best or last.")

        checkpoint_path = './ILSVRC_vgg_2branch/checkpoint.pth.tar'
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("Loading the best model from: '{}'".format(checkpoint_path))
        else:
            raise IOError("No checkpoint {}.".format(checkpoint_path))

    def make_dirs(self,path):
        if os.path.exists(path) is False:
            os.makedirs(path)

def main():
    trainer = Trainer()
    
    print(trainer.model)
    print("===========================================================")
    print("Start inference ...")
    trainer.load_checkpoint(checkpoint_type=trainer.args.eval_checkpoint_type)
    trainer.evaluate(epoch=0, split='val')
    trainer.print_performances()
    trainer.report(epoch=0, split='val')
    trainer.save_checkpoint(epoch=0, split='val')




if __name__ == '__main__':
    main()