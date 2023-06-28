#multi里面的inference.py

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

import cv2
import numpy as np
import os
from evaluation import BoxEvaluator
from evaluation import configure_metadata
from util import t2n
from tqdm import *
import torch
from torch.nn import functional as F
from PIL import Image


_IMAGENET_MEAN = [0.485, .456, .406]
_IMAGENET_STDDEV = [.229, .224, .225]
_RESIZE_LENGTH = 224


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

class CAMComputer(object):
    def __init__(self, model, loader_cls, loader_loc, metadata_root,
                 iou_threshold, dataset_name, split,
                 loc_metric_list, isgrad, istencrop):
        self.model = model
        self.model.eval()
        self.loader_cls = loader_cls
        self.loader_loc = loader_loc
        self.loc_metric_list = loc_metric_list
        self.iou_threshold = iou_threshold
        self.isgrad = isgrad
        self.istencrop = istencrop

        metadata = configure_metadata(metadata_root)
        cam_threshold_list = list(np.arange(0.05, 0.35, 0.01))

        self.evaluator = BoxEvaluator(metadata=metadata,
                                      dataset_name=dataset_name,
                                      split=split,
                                      cam_threshold_list=cam_threshold_list,
                                      iou_threshold=self.iou_threshold,
                                      loc_metric_list=self.loc_metric_list)
        
    def compute_and_evaluate_cams(self):
        print("Computing and evaluating cams.")
        image_sizes = {}

        if self.isgrad:
            print("it is MNPOs.")
        else:
            print("it is MNPO.")
        for (images_cls, targets, image_ids),(images_loc, _, _) in zip(tqdm(self.loader_cls), self.loader_loc):
            image_size = images_loc.shape[2:]
            
            images_loc = images_loc.cuda()
            targets = targets.cuda()
            if self.isgrad:
                output_loc = self.model(images_loc, targets, return_cam=not self.isgrad, return_dgl=self.isgrad)
            else:
                output_loc = self.model(images_loc, targets, return_cam=not self.isgrad) 
            
            cams1 = t2n(output_loc['cams1'])
            cams2 = t2n(output_loc['cams2'])

            if self.istencrop:
                images_cls = images_cls.cuda()
                bs, ncrops, c, h, w = images_cls.shape
                images_cls = images_cls.view(-1, c, h, w)
                output_cls = self.model(images_cls, targets)
                logits = output_cls['logits1'].view(bs, ncrops, -1).mean(1)
            else:
                logits = output_loc['logits1'].cuda()
            i = 0
            for cam1, cam2, image_id, target in zip(cams1, cams2, image_ids, targets):
                if self.isgrad:
                    cam_resized = cv2.resize(cam1, image_size,interpolation=cv2.INTER_CUBIC)
                else:
                    cam1_resized = torch.tensor(cv2.resize(cam1, image_size,interpolation=cv2.INTER_CUBIC))
                    cam2_resized = torch.tensor(cv2.resize(cam2, image_size,interpolation=cv2.INTER_CUBIC)) 
                    
                    cam_resized = np.asarray(torch.max(cam1_resized, cam2_resized))
                cam_normalized = normalize_scoremap(cam_resized)
                
                self.evaluator.accumulate(cam_normalized, image_id, target, logits[i])
                i += 1
            
        return self.evaluator.compute()
        
    
