#it is the code for version3
"""
Original code: https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.model_zoo import load_url
from collections import OrderedDict
from .method import AcolBase
from .method import ADL
from .method import spg
from .method.util import normalize_tensor
from .util import remove_layer
from .util import replace_layer
from .util import initialize_weights

__all__ = ['vgg16']

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'
}

configs_dict = {
    'cam': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512,
                  512, 'M', 512, 512, 512],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512,
                  512, 512, 512, 512],
    },
}

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, base_width=64):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.))
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width,3,stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)


        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Branch1_Module(nn.Module):
    def __init__(self, block, in_fea=[16, 32, 64, 128, 256, 512, 1024]):
        super(Branch1_Module, self).__init__()
        self.inplanes = 512        

        self.conv1 = nn.Sequential(
           nn.Conv2d(in_fea[5], in_fea[2], kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
           nn.BatchNorm2d(in_fea[2]),
           nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
           nn.Conv2d(in_fea[5], in_fea[2], kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
           nn.BatchNorm2d(in_fea[2]),
           nn.ReLU(inplace=True)
        )     
        
        self.layer1 = self._make_layer(block, 224, 1, stride=2)
        
        self.inplanes = 960
        self.layer2 = self._make_layer(block, 240, 1, stride=1)
        
    def forward(self, x1, x2, x3):
        x1 = self.layer1(x1)
        x2 = self.conv1(x2)
        x12_ = torch.cat([x1, x2], dim=1)
        x12 = self.layer2(x12_)
        x3 = self.conv2(x3)
        branch_fea = torch.cat([x12, x3], dim=1)
        return x12_, branch_fea
        
    def _make_layer(self, block, planes, blocks, stride):
        layers = self._layer(block, planes, blocks, stride)
        return nn.Sequential(*layers)

    def _layer(self, block, planes, blocks, stride):
        downsample = get_downsampling_layer(self.inplanes, block, planes,
                                            stride)

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return layers      

class VggCam(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(VggCam, self).__init__()
        
        self.conv1 = nn.Sequential(*features[:2])
        self.conv2 = nn.Sequential(*features[2:4])
        
        self.maxpool = nn.Sequential(*features[4:5])
        
        self.conv3 = nn.Sequential(*features[5:7])
        self.conv4 = nn.Sequential(*features[7:9])
        
        self.conv5 = nn.Sequential(*features[10:12])
        self.conv6_= nn.Sequential(*features[12:14])
        self.conv7 = nn.Sequential(*features[14:16])
        
        self.conv8 = nn.Sequential(*features[17:19])
        self.conv9 = nn.Sequential(*features[19:21])
        
        self.conv10 = nn.Sequential(*features[21:23])
        
        self.conv11 = nn.Sequential(*features[24:26])
        self.conv12 = nn.Sequential(*features[26:28])
        self.conv13 = nn.Sequential(*features[28:])

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)
        self.branch1_layer = Branch1_Module(block=Bottleneck)
        
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False, return_dgl=False):
     
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool(x)
        
        x = self.conv5(x)
        x = self.conv6_(x)
        x = self.conv7(x)
        x = self.maxpool(x)
        
        x = self.conv8(x)
        x1 = self.conv9(x)
        x = self.conv10(x1)
        x = self.maxpool(x)
        
        x2 = self.conv11(x)
        x = self.conv12(x2)
        x3 = self.conv13(x) 

        x = self.conv6(x3)
        x = self.relu(x)
        
        x32, branch2_fea = self.branch1_layer(x1, x2, x3)
        
        pre_logit1 = self.avgpool(x)
        pre_logit1 = pre_logit1.view(pre_logit1.size(0), -1)
        logits1 = self.fc(pre_logit1)
        
        pre_logit2 = self.avgpool(branch2_fea)
        pre_logit2 = pre_logit2.view(pre_logit2.size(0), -1)
        logits2 = self.fc(pre_logit2)
        
        
        if return_cam:
            feature_map = x.detach().clone()
            cam_weights = self.fc.weight[labels]
            cam_weights = torch.where(cam_weights > 0, cam_weights * 10, cam_weights * 5)
            cams1 = (cam_weights.view(*feature_map.shape[:2], 1, 1) *
                    feature_map).mean(1, keepdim=False)
                   
            feature_map2 = branch2_fea.detach().clone()
            cams2 = (cam_weights.view(*feature_map2.shape[:2], 1, 1) *
                    feature_map2).mean(1, keepdim=False)
                    
            
            return {'cams1':cams1, 'cams2':cams2, 'logits1': logits1, 'logits2': logits2} 
        else:
            return {'logits1': logits1, 'logits2': logits2}


def adjust_pretrained_model(pretrained_model, current_model):
    def _get_keys(obj, split):
        keys = []
        values = []
        iterator = obj.items() if split == 'pretrained' else obj
        for key, value in iterator:
            keys.append(key)
            values.append(key)
        return keys, values

    def _align_keys(obj, key1, key2):
        length = len(key1)
        pretrained_model_adjust = {}

        for i in range(length):
            pretrained_model_adjust[key2[i]] = obj[key1[i]]
        return pretrained_model_adjust

    pretrained_keys, pretrained_values = _get_keys(pretrained_model, 'pretrained')
    current_keys, current_values = _get_keys(current_model.named_parameters(), 'model')

    pretrained_model_adjust = _align_keys(pretrained_model, pretrained_keys, current_keys)    

    return pretrained_model_adjust

def load_pretrained_model(model, architecture_type, path=None, **kwargs):
    
    print("Loading pretrain path ...")
    state_dict = load_url(model_urls['vgg16'], progress=True)
    
    model.load_state_dict(state_dict, strict=False)
    
    return model
def make_layers(cfg, **kwargs):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M1':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        elif v == 'M2':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        elif v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'A':
            layers += [
                ADL(kwargs['adl_drop_rate'], kwargs['adl_drop_threshold'])]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def get_downsampling_layer(inplanes, block, planes, stride):
    outplanes = planes * block.expansion
    if stride == 1 and inplanes == outplanes:
        return
    else:
        return nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1, stride, bias=False),
            nn.BatchNorm2d(outplanes),
        )

def vgg_MNPO(architecture_type, pretrained=False, pretrained_path=None,
          **kwargs):
    config_key = '14x14'
    layers = make_layers(configs_dict[architecture_type][config_key], **kwargs)
    model = VggCam(layers, **kwargs)
    if pretrained:
        model = load_pretrained_model(model, architecture_type,
                                      path=pretrained_path, **kwargs)
    return model