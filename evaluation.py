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
import cv2
import numpy as np
import os
import torch.utils.data as torchdata

from config import str2bool
from data_loaders import configure_metadata
from data_loaders import get_image_ids
from data_loaders import get_bounding_boxes
from data_loaders import get_image_sizes
from data_loaders import get_mask_paths
from util import check_scoremap_validity
from util import check_box_convention
from util import t2n
import torch

_IMAGENET_MEAN = [0.485, .456, .406]
_IMAGENET_STDDEV = [.229, .224, .225]
_RESIZE_LENGTH = 224
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0


def calculate_multiple_iou(box_a, box_b):
    
    """
    Args:
        box_a: numpy.ndarray(dtype=np.int, shape=(num_a, 4))
            x0y0x1y1 convention.
        box_b: numpy.ndarray(dtype=np.int, shape=(num_b, 4))
            x0y0x1y1 convention.
    Returns:
        ious: numpy.ndarray(dtype=np.int, shape(num_a, num_b))
    """
    num_a = box_a.shape[0]
    num_b = box_b.shape[0]
    check_box_convention(box_a, 'x0y0x1y1')
    check_box_convention(box_b, 'x0y0x1y1')

    # num_a x 4 -> num_a x num_b x 4应该是将box_a和box_b整成相同维度
    box_a = np.tile(box_a, num_b)
    box_a = np.expand_dims(box_a, axis=1).reshape((num_a, num_b, -1))

    # num_b x 4 -> num_b x num_a x 4
    box_b = np.tile(box_b, num_a)
    box_b = np.expand_dims(box_b, axis=1).reshape((num_b, num_a, -1))

    # num_b x num_a x 4 -> num_a x num_b x 4
    box_b = np.transpose(box_b, (1, 0, 2))

    # num_a x num_b
    min_x = np.maximum(box_a[:, :, 0], box_b[:, :, 0])
    min_y = np.maximum(box_a[:, :, 1], box_b[:, :, 1])
    max_x = np.minimum(box_a[:, :, 2], box_b[:, :, 2])
    max_y = np.minimum(box_a[:, :, 3], box_b[:, :, 3])

    # num_a x num_b
    area_intersect = (np.maximum(0, max_x - min_x + 1)
                      * np.maximum(0, max_y - min_y + 1))#∩
    area_a = ((box_a[:, :, 2] - box_a[:, :, 0] + 1) *
              (box_a[:, :, 3] - box_a[:, :, 1] + 1))
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0] + 1) *
              (box_b[:, :, 3] - box_b[:, :, 1] + 1))

    denominator = area_a + area_b - area_intersect#∪
    degenerate_indices = np.where(denominator <= 0)
    denominator[degenerate_indices] = 1

    ious = area_intersect / denominator
    ious[degenerate_indices] = 0
    return ious

def resize_bbox(box, image_size, resize_size):
    """
    Args:
        box: iterable (ints) of length 4 (x0, y0, x1, y1)
        image_size: iterable (ints) of length 2 (width, height)
        resize_size: iterable (ints) of length 2 (width, height)

    Returns:
         new_box: iterable (ints) of length 4 (x0, y0, x1, y1)
    """
    check_box_convention(np.array(box), 'x0y0x1y1')
    box_x0, box_y0, box_x1, box_y1 = map(float, box)
    image_w, image_h = map(float, image_size)
    new_image_w, new_image_h = map(float, resize_size)

    newbox_x0 = box_x0 * new_image_w / image_w
    newbox_y0 = box_y0 * new_image_h / image_h
    newbox_x1 = box_x1 * new_image_w / image_w
    newbox_y1 = box_y1 * new_image_h / image_h
    
    '''yy = int((256 - 224) / 2)
    xx = int((256 - 224) / 2)
    newbox_x0 = newbox_x0 - xx
    
    newbox_x1 = newbox_x1 - xx
    newbox_y0 = newbox_y0 - yy
    newbox_y1 = newbox_y1 - yy
    if newbox_x0<0  : newbox_x0 = 0
    if newbox_x1>223: newbox_x1 = 223
    if newbox_x1<0         : newbox_x1 = 0
    if newbox_y0<0         : newbox_y0 = 0
    if newbox_y1>223: newbox_y1 = 223
    if newbox_y1<0         : newbox_y1 = 0 '''          
    
    return int(newbox_x0), int(newbox_y0), int(newbox_x1), int(newbox_y1)


def compute_bboxes_from_scoremaps(scoremap, scoremap_threshold_list):
    """
    Args:
        scoremap: numpy.ndarray(dtype=np.float32, size=(H, W)) between 0 and 1
        scoremap_threshold_list: iterable
        multi_contour_eval: flag for multi-contour evaluation

    Returns:
        estimated_boxes_at_each_thr: list of estimated boxes (list of np.array)
            at each cam threshold
        number_of_box_list: list of the number of boxes at each cam threshold
    """
    check_scoremap_validity(scoremap)
    height, width = scoremap.shape
    scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)

    def scoremap2bbox(threshold):
        _, thr_gray_heatmap = cv2.threshold(         
            src=scoremap_image,
            thresh=int(threshold * np.max(scoremap_image)),
            maxval=255,
            type=cv2.THRESH_BINARY)
        contours = cv2.findContours(                 
            image=thr_gray_heatmap,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

        if len(contours) == 0:
            return np.asarray([[0, 0, 0, 0]]), 1

       
        contours = [max(contours, key=cv2.contourArea)]

        estimated_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x0, y0, x1, y1 = x, y, x + w, y + h
            x1 = min(x1, width - 1)
            y1 = min(y1, height - 1)
            estimated_boxes.append([x0, y0, x1, y1])

        return np.asarray(estimated_boxes), len(contours)

    estimated_boxes_at_each_thr = []
    number_of_box_list = []
    for threshold in scoremap_threshold_list:
        boxes, number_of_box = scoremap2bbox(threshold)
        estimated_boxes_at_each_thr.append(boxes)
        number_of_box_list.append(number_of_box)

    return estimated_boxes_at_each_thr, number_of_box_list


#定位评估
class LocalizationEvaluator(object):
    """ Abstract class for localization evaluation over score maps.

    The class is designed to operate in a for loop (e.g. batch-wise cam
    score map computation). At initialization, __init__ registers paths to
    annotations and data containers for evaluation. At each iteration,
    each score map is passed to the accumulate() method along with its image_id.
    After the for loop is finalized, compute() is called to compute the final
    localization performance.
    """

    def __init__(self, metadata, dataset_name, split, cam_threshold_list, iou_threshold, loc_metric_list):
        self.metadata = metadata
        self.cam_threshold_list = cam_threshold_list
        self.loc_metric_list = loc_metric_list
        self.iou_threshold = iou_threshold
        self.dataset_name = dataset_name

    def accumulate(self, scoremap, image_id, preds, target, num_image):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError

class BoxEvaluator(LocalizationEvaluator):
    def __init__(self, **kwargs):
        super(BoxEvaluator, self).__init__(**kwargs)

        self.image_ids = get_image_ids(metadata=self.metadata)
        self.resize_length = _RESIZE_LENGTH
        self.cnt = 0
        self.num_correct = {loc_metric: np.zeros(len(self.cam_threshold_list))
                            for loc_metric in self.loc_metric_list}
        self.original_bboxes = get_bounding_boxes(self.metadata)
        self.image_sizes = get_image_sizes(self.metadata)
        self.gt_bboxes = self._load_resized_boxes(self.original_bboxes)

    def _load_resized_boxes(self, original_bboxes):
        resized_bbox = {image_id: [
            resize_bbox(bbox, self.image_sizes[image_id],
                        (self.resize_length, self.resize_length))
            for bbox in original_bboxes[image_id]]
            for image_id in self.image_ids}
        return resized_bbox

    def compute_correct(self, input, k, target):
        pre = 0
        pred = (torch.topk(input, k))[1]
        pre += (pred==target).sum().item()
        return pre

    def accumulate(self, scoremap, image_id, target=None, pred=None):
        """
        From a score map, a box is inferred (compute_bboxes_from_scoremaps).
        The box is compared against GT boxes. Count a scoremap as a correct
        prediction if the IOU against at least one box is greater than a certain
        threshold (_IOU_THRESHOLD).

        Args:
            scoremap: numpy.ndarray(size=(H, W), dtype=np.float)
            image_id: string.
        """
        boxes_at_thresholds, number_of_box_list = compute_bboxes_from_scoremaps(
            scoremap=scoremap,
            scoremap_threshold_list=self.cam_threshold_list,)

        boxes_at_thresholds = np.concatenate(boxes_at_thresholds, axis=0)

        multiple_iou = calculate_multiple_iou(
            np.array(boxes_at_thresholds),
            np.array(self.gt_bboxes[image_id]))

        for loc_metric in self.loc_metric_list:
            if loc_metric == 'gt-known':
                sliced_multiple_iou = []
                idx = 0
                for nr_box in number_of_box_list:
                    sliced_multiple_iou.append(max(multiple_iou.max(1)[idx:idx + nr_box]))
                    idx += nr_box
                correct_threshold_indices = np.where(np.asarray(sliced_multiple_iou) >= (self.iou_threshold/100))[0]#记录大于iou_threshold的位置
                self.num_correct[loc_metric][correct_threshold_indices] += 1
                
            elif loc_metric == 'top-1':
                sliced_multiple_iou = []
                idx = 0
                pred1 = self.compute_correct(pred, 1, target)
                for nr_box in number_of_box_list:
                    if pred1 == 0:
                        sliced_multiple_iou.append(0)
                    elif pred1 == 1:
                        sliced_multiple_iou.append(max(multiple_iou.max(1)[idx:idx + nr_box]))
                    else:
                        exit()
                    idx += nr_box
                correct_threshold_indices = np.where(np.asarray(sliced_multiple_iou) >= (self.iou_threshold/100))[0]
                self.num_correct[loc_metric][correct_threshold_indices] += 1
                
            elif loc_metric == 'top-5':
                idx = 0
                sliced_multiple_iou = []
                pred5 = self.compute_correct(pred, 5, target)
                for nr_box in number_of_box_list:
                    if pred5 == 0:
                        sliced_multiple_iou.append(0)
                    elif pred5 == 1:
                        sliced_multiple_iou.append(max(multiple_iou.max(1)[idx:idx + nr_box]))
                    else:
                        exit()
                    idx += nr_box
                correct_threshold_indices = np.where(np.asarray(sliced_multiple_iou) >= (self.iou_threshold/100))[0]
                self.num_correct[loc_metric][correct_threshold_indices] += 1
                
            else:
                print('loc_metric error')
                exit()
        self.cnt += 1

    def compute(self):
        """
        Returns:
            max_localization_accuracy: float. The ratio of images where the
               box prediction is correct. The best scoremap threshold is taken
               for the final performance.
        """
        max_box_acc_err = []
        thresh = []
        for loc_metric in self.loc_metric_list:
            
            
            localization_accuracies = 100 - (self.num_correct[loc_metric] * 100. / \
                                      float(self.cnt))
            max_box_acc_err.append(localization_accuracies.min())

        return max_box_acc_err