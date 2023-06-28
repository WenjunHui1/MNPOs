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

class softmax_cross_entropy_loss_F(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, target):
        if not target.is_same_size(input):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(
                target.size(), input.size()))
        ctx.save_for_backward(input, target)
        input = F.softmax(input, dim=1)

        loss = - target * torch.log(input)
        loss = torch.sum(loss, 1)
        loss = torch.unsqueeze(loss, 1)
        return torch.sum(loss)

    @staticmethod
    def backward(ctx, grad_output):
        input, target = ctx.saved_tensors
        return F.softmax(input, dim=1) - target, None

class softmax_cross_entropy_loss(nn.Module):
    def __init__(self):
        super(softmax_cross_entropy_loss, self).__init__()
    def forward (self, input, target):
        if not target.is_same_size(input):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(
                target.size(), input.size()))

        return softmax_cross_entropy_loss_F.apply(input, target)

class FixedBatchNorm(nn.BatchNorm2d):
    def forward(self, input):
        return F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias,
                            training=False, eps=self.eps)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, base_width=64):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.))
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)#因为BatchNorm层可以去均值化，所以卷积层的bias都是False
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
            identity = self.downsample(x)#当连接的维度不同时，使用1*1的卷积核将低维转成高维，然后才能进行相加

        out += identity#实现H(x)=F(x)+x或H(x)=F(x)+Wx
        out = self.relu(out)

        return out


class Branch1_Module(nn.Module):
    def __init__(self, block, in_fea=[16, 32, 64, 128, 256, 512, 1024]):#56x56x256#14x14x2048
        super(Branch1_Module, self).__init__()
        self.inplanes = 512   

        self.isgrad_dgl = True
        self.branch_grads_list = {}       

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
        
    def forward(self, x1, x2, x3, grads_branch_dgl, return_dgl=True):#[32, 512, 28, 28]) torch.Size([32, 512, 14, 14]) torch.Size([32, 512, 14, 14]
        self.grads_branch_dgl = grads_branch_dgl
        x1 = self.layer1(x1)#14x14x(512+256+128)
        
        x2 = self.conv1(x2)#14x14x64
        x12_ = torch.cat([x1, x2], dim=1)#14x14x(512+256+128)
        if return_dgl:
            x12_.register_hook(self.save_grad('branch_concat1_'))
            self.branch_grads_list['branch_concat1_'] = x12_
            
        x12 = self.layer2(x12_)#14x14x(512+256+128)
        if return_dgl:
            x12.register_hook(self.save_grad('branch_concat12'))
            self.branch_grads_list['branch_concat12'] = x12
            
        x3 = self.conv2(x3)#14x14x(64)
        branch_fea = torch.cat([x12, x3], dim=1)
        if return_dgl:
            branch_fea.register_hook(self.save_grad('branch_fea'))
            self.branch_grads_list['branch_fea'] = branch_fea
        return branch_fea, self.branch_grads_list, self.grads_branch_dgl
        
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
        
        
    def save_grad(self, name):
        def hook(grad):
            self.grads_branch_dgl[name] = grad
        return hook

    def change_grad(self, labels):
        def hook(grad):
            print('===========   I change it')
            grad = grad*labels
        return hook

        

class VggCam(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(VggCam, self).__init__()
        #64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512,512, 'M', 512, 512, 512
        #self.features = features
        self.cls_loss_dgl = softmax_cross_entropy_loss()
        self.WA = True
        self.isgrad_dgl = True
        self.dgl_main_layer = kwargs['main_layer']
        self.dgl_branch_layer = kwargs['branch_layer']
        self.layers_grads_list = {}
        self.grads_main_dgl = {}
        
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
        #self.branch2 = Branch2_Module(block=Bottleneck)
        initialize_weights(self.modules(), init_mode='he')
        
    def save_grad(self, name):
        def hook(grad):
            self.grads_main_dgl[name] = grad
        return hook

    def change_grad(self, labels):
        def hook(grad):
            grad = grad * labels
        return hook

    def forward(self, x, labels=None, return_cam=False, return_dgl=False):
     #64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512,512, 'M', 512, 512, 512
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool(x)
        
        x = self.conv5(x)
        x = self.conv6_(x)
        if self.isgrad_dgl:
            x.register_hook(self.save_grad('conv3_2'))
            self.layers_grads_list['conv3_2'] = x
        
        x = self.conv7(x)
        if self.isgrad_dgl:
            x.register_hook(self.save_grad('conv3_3'))
            self.layers_grads_list['conv3_3'] = x
        
        x = self.maxpool(x)#[32, 256, 28, 28]
        if self.isgrad_dgl:
            x.register_hook(self.save_grad('pool3'))
            self.layers_grads_list['pool3'] = x
        
        x = self.conv8(x)
        if self.isgrad_dgl:
            x.register_hook(self.save_grad('conv4_1'))
            self.layers_grads_list['conv4_1'] = x
        
        x1 = self.conv9(x)
        if self.isgrad_dgl:
            x1.register_hook(self.save_grad('conv4_2'))
            self.layers_grads_list['conv4_2'] = x1
        
        x = self.conv10(x1)
        if self.isgrad_dgl:
            x.register_hook(self.save_grad('conv4_3'))
            self.layers_grads_list['conv4_3'] = x
        
        x = self.maxpool(x)#[32, 512, 14, 14]
        if self.isgrad_dgl:
            x.register_hook(self.save_grad('pool4'))
            self.layers_grads_list['pool4'] = x
        
        x2 = self.conv11(x)
        if self.isgrad_dgl:
            x2.register_hook(self.save_grad('conv5_1'))
            self.layers_grads_list['conv5_1'] = x2
        
        x = self.conv12(x2)
        if self.isgrad_dgl:
            x.register_hook(self.save_grad('conv5_2'))
            self.layers_grads_list['conv5_2'] = x
            
        x3 = self.conv13(x) #[32, 512, 14, 14]
        if self.isgrad_dgl:
            x3.register_hook(self.save_grad('conv5_3'))
            self.layers_grads_list['conv5_3'] = x3

        x = self.conv6(x3)#28x28x1024
        x = self.relu(x)
        if self.isgrad_dgl:
            x.register_hook(self.save_grad('conv5_4'))
            self.layers_grads_list['conv5_4'] = x
            
        branch_grads_list = self.layers_grads_list
        grads_branch_dgl = self.grads_main_dgl
        
        branch2_fea, branch_grads_list, grads_branch_dgl = self.branch1_layer(x1, x2, x3, grads_branch_dgl)#56x56
           
        pre_logit1 = self.avgpool(x)
        pre_logit1 = pre_logit1.view(pre_logit1.size(0), -1)
        logits1 = self.fc(pre_logit1)
        if self.isgrad_dgl:
            logits1.register_hook(self.save_grad('logits'))
            self.layers_grads_list['logits'] = logits1
            
        pre_logit2 = self.avgpool(branch2_fea)
        pre_logit2 = pre_logit2.view(pre_logit2.size(0), -1)
        logits2 = self.fc(pre_logit2)
     

        if return_cam:
            feature_map = x.detach().clone()
            cam_weights = self.fc.weight[labels]
            cam_weights = torch.where(cam_weights > 0, cam_weights  * 8, cam_weights * 4)
            cams1 = (cam_weights.view(*feature_map.shape[:2], 1, 1) *
                    feature_map).mean(1, keepdim=False)
                    
            feature_map2 = branch2_fea.detach().clone()
            cams2 = (cam_weights.view(*feature_map2.shape[:2], 1, 1) *
                    feature_map2).mean(1, keepdim=False)
                    
            return {'cams1':cams1, 'cams2':cams2, 'logits1': logits1, 'logits2': logits2} 
        if return_dgl:
            with torch.enable_grad():
                #单纯的只是对中间层利用梯度进行加权和会比最后一层要好吗？特征图还需要加强吗？
                batch_size = x.shape[0]
                
                target_layer_name = self.dgl_branch_layer
                sn = branch_grads_list[target_layer_name]
                lc = logits2[range(batch_size), labels].sum()
                self.zero_grad()
                lc.backward(retain_graph=True)
                g_lc2_wrt_sn_main = grads_branch_dgl[self.dgl_main_layer].clone()  
                grads_branch_dgl.clear()

                self.zero_grad()
                weighted_one_hot = torch.zeros([batch_size, 1000]).cuda().float()
                weighted_one_hot[range(batch_size), labels] = 10
                loss = self.cls_loss_dgl(logits2, weighted_one_hot)
                loss.backward(retain_graph=True)
                cls_main_loss_grad = grads_branch_dgl[self.dgl_main_layer].clone()

                target_layer_name = self.dgl_main_layer
                sn = self.layers_grads_list[target_layer_name]#[32, 1000, 28, 28] 
                lc = logits1[range(batch_size), labels].sum()#[32] 
                self.zero_grad()
                lc.backward(retain_graph=True)
                g_lc2_wrt_sn = self.grads_main_dgl[target_layer_name].clone() + g_lc2_wrt_sn_main#[32, 1000, 28, 28])求取梯度作为权重
                grad_weight1 = torch.sum(g_lc2_wrt_sn, dim=[2, 3]) 
                grad_weight1 = torch.where(grad_weight1 > 0, grad_weight1 * 8, grad_weight1 * 4)
                self.grads_main_dgl.clear()
                
                self.zero_grad()
                weighted_one_hot = torch.zeros([batch_size, 1000]).cuda().float()
                weighted_one_hot[range(batch_size), labels] = 10
                loss = self.cls_loss_dgl(logits1, weighted_one_hot)#分类损失
                loss.backward(retain_graph=True)
                cls_loss_grad = self.grads_main_dgl[target_layer_name].clone() + cls_main_loss_grad
                grad_norm = cls_loss_grad / cls_loss_grad.view(batch_size, -1).norm(dim=1).view(batch_size, 1, 1, 1)#归一化梯度
                sn_norm = sn / sn.view(batch_size, -1).norm(dim=1).view(batch_size, 1, 1, 1)#归一化特征图
                An = sn_norm - grad_norm # 增强图
                feature_map1 = An.clone().detach()
                cams1 = (grad_weight1.view(*feature_map1.shape[:2], 1, 1) * feature_map1).mean(1, keepdim=False)
                
                cam_weights = self.fc.weight[labels] 
                feature_map2 = branch2_fea.clone().detach()
                cams2 = (cam_weights.view(*feature_map2.shape[:2], 1, 1) * feature_map2).mean(1, keepdim=False)
            
            return {'logits1': logits1, 'logits2': logits2, 'cams1':cams1, 'cams2':cams2}
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

def vgg_MNPOs(architecture_type, pretrained=False, pretrained_path=None,
          **kwargs):
    config_key = '14x14'
    layers = make_layers(configs_dict[architecture_type][config_key], **kwargs)
    model = VggCam(layers, **kwargs)
    if pretrained:
        model = load_pretrained_model(model, architecture_type,
                                      path=pretrained_path, **kwargs)
    return model
