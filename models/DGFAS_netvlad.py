import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock
import sys
import numpy as np
from torch.autograd import Variable
import random
import os
import math
import torch.nn.functional as F
from utils.utils import to_categorical_ml
def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

def real_select(x):
    batch_size = x.shape[0]
    soft_assign_real_1 = x.narrow(0, 0, batch_size // 6)
    soft_assign_real_2 = x.narrow(0, batch_size // 3, batch_size // 6)
    soft_assign_real_3 = x.narrow(0, (batch_size // 3 )*2, batch_size // 6)
    soft_assign_real = torch.cat([soft_assign_real_1, soft_assign_real_2, soft_assign_real_3], dim=0)  # 30 * 40
    return soft_assign_real

class Feature_Generator_MADDG(nn.Module):
    def __init__(self):
        super(Feature_Generator_MADDG, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(128)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.conv1_3 = nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(196)
        self.relu1_3 = nn.ReLU(inplace=True)
        self.conv1_4 = nn.Conv2d(196, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_4 = nn.BatchNorm2d(128)
        self.relu1_4 = nn.ReLU(inplace=True)
        self.maxpool1_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1_5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_5 = nn.BatchNorm2d(128)
        self.relu1_5 = nn.ReLU(inplace=True)
        self.conv1_6 = nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_6 =  nn.BatchNorm2d(196)
        self.relu1_6 = nn.ReLU(inplace=True)
        self.conv1_7 = nn.Conv2d(196, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_7 = nn.BatchNorm2d(128)
        self.relu1_7 = nn.ReLU(inplace=True)
        self.maxpool1_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1_8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_8 = nn.BatchNorm2d(128)
        self.relu1_8 = nn.ReLU(inplace=True)
        self.conv1_9 = nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_9 = nn.BatchNorm2d(196)
        self.relu1_9 = nn.ReLU(inplace=True)
        self.conv1_10 = nn.Conv2d(196, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_10 = nn.BatchNorm2d(128)
        self.relu1_10 = nn.ReLU(inplace=True)
        self.maxpool1_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.conv1_1(x)
        out = self.bn1_1(out)
        out = self.relu1_1(out)
        out = self.conv1_2(out)
        out = self.bn1_2(out)
        out = self.relu1_2(out)
        out = self.conv1_3(out)
        out = self.bn1_3(out)
        out = self.relu1_3(out)
        out = self.conv1_4(out)
        out = self.bn1_4(out)
        out = self.relu1_4(out)
        pool_out1 = self.maxpool1_1(out)

        out = self.conv1_5(pool_out1)
        out = self.bn1_5(out)
        out = self.relu1_5(out)
        out = self.conv1_6(out)
        out = self.bn1_6(out)
        out = self.relu1_6(out)
        out = self.conv1_7(out)
        out = self.bn1_7(out)
        out = self.relu1_7(out)
        pool_out2 = self.maxpool1_2(out)

        out = self.conv1_8(pool_out2)
        out = self.bn1_8(out)
        out = self.relu1_8(out)
        out = self.conv1_9(out)
        out = self.bn1_9(out)
        out = self.relu1_9(out)
        out = self.conv1_10(out)
        out = self.bn1_10(out)
        out = self.relu1_10(out)
        pool_out3 = self.maxpool1_3(out)
        return pool_out3

class Feature_Embedder_MADDG(nn.Module):
    def __init__(self, num_cluster, dim, alpha):
        super(Feature_Embedder_MADDG, self).__init__()
        #self.conv3_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3_1 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.pool2_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.pool2_2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        '''
        self.bottleneck_layer_1 = nn.Sequential(
            self.conv3_1,
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            self.pool2_1,
            self.conv3_2,
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            self.pool2_2,
            self.conv3_3,
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        '''
        self.bottleneck_layer_2 = nn.Sequential(
            self.conv3_1,
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #self.pool2_1,
            #self.conv3_2,
            #nn.BatchNorm2d(128),
            #nn.ReLU(inplace=True)
        )
        #self.avg_pool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.netvlad = NetVLAD_cent_adap(num_clusters=num_cluster, dim=dim, alpha=alpha)
    def forward(self, input, norm_flag):
        feature = self.bottleneck_layer_2(input)
        feature, soft_assign, local = self.netvlad(feature)  # [0.5, + 1/4 + 1/8 + 1/8] feature[:, 0.75]
        return feature, soft_assign, local

class Feature_Embedder_MADDG_nonorm_sharesp(nn.Module):
    def __init__(self, num_cluster, dim, alpha, par_dim):
        super(Feature_Embedder_MADDG_nonorm_sharesp, self).__init__()
        #self.conv3_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3_1 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.pool2_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.pool2_2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        '''
        self.bottleneck_layer_1 = nn.Sequential(
            self.conv3_1,
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            self.pool2_1,
            self.conv3_2,
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            self.pool2_2,
            self.conv3_3,
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        '''
        self.bottleneck_layer_2 = nn.Sequential(
            self.conv3_1,
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #self.pool2_1,
            #self.conv3_2,
            #nn.BatchNorm2d(128),
            #nn.ReLU(inplace=True)
        )
        self.par_dim = par_dim
        self.netvlad = NetVLAD_cent_nonorm(num_clusters=num_cluster, dim=dim, alpha=alpha)
        #self.netvlad = NetVLAD_cent_adap(num_clusters=num_cluster, dim=dim, alpha=alpha)
    def forward(self, input, norm_flag):
        feature = self.bottleneck_layer_2(input)
        feature, soft_assign, local = self.netvlad(feature)  # [0.5, + 1/4 + 1/8 + 1/8] feature[:, 0.75]
        feature_share = feature[:, 0: self.par_dim[0]]  # L2 normalize
        feature_sp_benefit = feature[:, self.par_dim[0]:]

        return feature_share, feature_sp_benefit, soft_assign, local

def one_hot(ids, depth):
    z = np.zeros([len(ids), depth])
    z[np.arange(len(ids)), ids] = 1
    return z

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    # change your path
    #model_path = '/mnt/sdb/renyi/jw/SSDG-CVPR2020-master/pretrained_model/resnet18-5c106cde.pth'
    model_path = '/home1/wangjiong/FAS/SSDG-CVPR2020-master/pretrained_model/resnet18-5c106cde.pth'
    if pretrained:
        model.load_state_dict(torch.load(model_path))
        print("loading model: ", model_path)
    return model

class Feature_Generator_ResNet18(nn.Module):
    def __init__(self):
        super(Feature_Generator_ResNet18, self).__init__()
        model_resnet = resnet18(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
    def forward(self, input):
        feature = self.conv1(input)
        feature = self.bn1(feature)
        feature = self.relu(feature)
        feature = self.maxpool(feature)
        feature = self.layer1(feature)
        feature = self.layer2(feature)
        feature = self.layer3(feature)
        return feature

class NetVLAD_cent(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, alpha=50.0,
                 normalize_input=True):

        super(NetVLAD_cent, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        # self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self.centroids = nn.Parameter(nn.init.orthogonal_(self.conv.weight).squeeze())
        self._init_params()

    def _init_params(self):
        '''
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )
        '''
        self.conv.weight = nn.Parameter(
            (self.centroids).permute(1, 0)
        )
        self.conv.bias = nn.Parameter(
            torch.zeros_like(self.centroids.norm(dim=1))
        )
    def forward(self, x):
        N, C = x.shape[:2]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim
        # soft-assignment
        soft_assign_ = torch.matmul(x.view(N, C, -1).permute(0, 2, 1),  F.normalize(self.conv.weight, p=2, dim=0))
        soft_assign = (soft_assign_ + self.conv.bias).permute(0, 2, 1)
        soft_assign = F.softmax(soft_assign * self.alpha, dim=1)   # 60, 32, 64
        x_flatten = x.view(N, C, -1)

        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
                   self.conv.weight.expand(x_flatten.size(-1), -1, -1).permute(2, 1, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad, soft_assign_.permute(0, 2, 1)

class NetVLAD_cent_nonorm(nn.Module):
    def __init__(self, num_clusters=64, dim=128, alpha=50.0,
                 normalize_input=True):
        super(NetVLAD_cent_nonorm, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(nn.init.orthogonal_(self.conv.weight).squeeze())
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (self.centroids).permute(1, 0)
        )
        self.conv.bias = nn.Parameter(
            torch.zeros_like(self.centroids.norm(dim=1))
        )

    def forward(self, x):
        N, C = x.shape[:2]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim
        # soft-assignment
        soft_assign_ = torch.matmul(x.view(N, C, -1).permute(0, 2, 1),  F.normalize(self.conv.weight, p=2, dim=0))
        soft_assign = (soft_assign_ + self.conv.bias).permute(0, 2, 1)
        soft_assign = F.softmax(soft_assign * self.alpha, dim=1)   # 60, 32, 64
        x_flatten = x.view(N, C, -1)
        feature = x.view(N, C, -1).permute(0, 2, 1)

        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.conv.weight.expand(x_flatten.size(-1), -1, -1).permute(2, 1, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization, vlad-60,K,128
        vlad = vlad.view(x.size(0), -1)  # flatten
        # vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        return vlad, soft_assign_.permute(0, 2, 1), feature

class NetVLAD_cent_adap(nn.Module):
    def __init__(self, num_clusters=64, dim=128, alpha=50.0, par_clu=[0, 0, 0, 0],
                 normalize_input=True):

        super(NetVLAD_cent_adap, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.par_clu = par_clu
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        # self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self.centroids = nn.Parameter(nn.init.orthogonal_(self.conv.weight).squeeze())
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (self.centroids).permute(1, 0)
        )
        self.conv.bias = nn.Parameter(
            torch.zeros_like(self.centroids.norm(dim=1))
        )
    def forward(self, x):
        N, C = x.shape[:2]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim
        # soft-assignment
        soft_assign_ = torch.matmul(x.view(N, C, -1).permute(0, 2, 1),  F.normalize(self.conv.weight, p=2, dim=0))
        soft_assign = (soft_assign_ + self.conv.bias).permute(0, 2, 1)
        soft_assign = F.softmax(soft_assign * self.alpha, dim=1)   # 60, 32, 64
        feature = x.view(N, C, -1).permute(0, 2, 1)

        x_flatten = x.view(N, C, -1)
        # calculate residuals to each clusters
        #residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
                   #F.normalize(self.conv.weight, p=2, dim=0).expand(x_flatten.size(-1), -1, -1).permute(2, 1, 0).unsqueeze(0)
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
                   self.conv.weight.expand(x_flatten.size(-1), -1, -1).permute(2, 1, 0).unsqueeze(0)
        # residual = F.normalize(residual, p=2, dim=-2)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad, soft_assign_.permute(0, 2, 1), feature

class Feature_Embedder_ResNet18_nonorm_sharesp(nn.Module):
    def __init__(self, num_cluster, dim, par_dim, alpha, par_clu):
        super(Feature_Embedder_ResNet18_nonorm_sharesp, self).__init__()
        self.inplanes = 256
        self.layer4 = self._make_layer(BasicBlock, dim, 1, stride=1)
        for m in self.layer4:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.par_clu = par_clu
        self.netvlad = NetVLAD_cent_nonorm(num_clusters=num_cluster, dim=dim, alpha=alpha)
        self.par_dim = par_dim
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, input, norm_flag):
        feature = self.layer4(input)
        feature, soft_assign, local = self.netvlad(feature)   # [0.5, + 1/4 + 1/8 + 1/8] feature[:, 0.75]
        feature_share = feature[:, 0: self.par_dim[0]]  # L2 normalize
        feature_sp_benefit = feature[:, self.par_dim[0]:]

        return feature_share, feature_sp_benefit, soft_assign, local

class Feature_Embedder_ResNet18(nn.Module):
    def __init__(self, num_cluster, dim, alpha):
        super(Feature_Embedder_ResNet18, self).__init__()
        self.inplanes = 256
        self.layer4 = self._make_layer(BasicBlock, dim, 1, stride=1)
        for m in self.layer4:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.netvlad = NetVLAD_cent(num_clusters=num_cluster, dim=dim, alpha=alpha)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          #kernel_size=3, stride=stride, bias=False, padding=1),
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, input, norm_flag):
        feature = self.layer4(input)
        feature, soft_assign = self.netvlad(feature)  # [0.5, + 1/4 + 1/8 + 1/8] feature[:, 0.75]
        return feature, soft_assign

class Feature_Embedder_ResNet18_adap(nn.Module):
    def __init__(self, num_cluster, dim, alpha):
        super(Feature_Embedder_ResNet18_adap, self).__init__()
        self.inplanes = 256
        self.layer4 = self._make_layer(BasicBlock, dim, 1, stride=1)
        for m in self.layer4:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.netvlad = NetVLAD_cent_adap(num_clusters=num_cluster, dim=dim, alpha=alpha)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          #kernel_size=3, stride=stride, bias=False, padding=1),
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, input, norm_flag):
        feature = self.layer4(input)
        feature, soft_assign, local_ = self.netvlad(feature)  # [0.5, + 1/4 + 1/8 + 1/8] feature[:, 0.75]
        return feature, soft_assign, local_


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier_layer = nn.Linear(512, 2)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)

    def forward(self, input, norm_flag):
        if(norm_flag):
            self.classifier_layer.weight.data = l2_norm(self.classifier_layer.weight, axis=0)
            classifier_out = self.classifier_layer(input)
        else:
            classifier_out = self.classifier_layer(input)
        return classifier_out

class Classifier_480(nn.Module):
    def __init__(self, dim):
        super(Classifier_480, self).__init__()
        self.classifier_layer = nn.Linear(dim, 2)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)

    def forward(self, input, norm_flag):
        if(norm_flag):
            self.classifier_layer.weight.data = l2_norm(self.classifier_layer.weight, axis=0)
            classifier_out = self.classifier_layer(input)
        else:
            classifier_out = self.classifier_layer(input)
        return classifier_out


class GRL(torch.autograd.Function):
    def __init__(self):
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 4000  # be same to the max_iter of config.py

    def forward(self, input):
        #self.iter_num += 1
        return input * 1.0

    def backward(self, gradOutput):
        self.iter_num += 1
        coeff = np.float(2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter))
                         - (self.high - self.low) + self.low)
        # print(self.iter_num )
        return -coeff * gradOutput

class GRL_itnum(torch.autograd.Function):
    def __init__(self):
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 4000  # be same to the max_iter of config.py

    def forward(self, input, iter_num):
        self.iter_num = iter_num
        return input * 1.0

    def backward(self, gradOutput):
        #self.iter_num += 1
        coeff = np.float(2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter))
                         - (self.high - self.low) + self.low)
        #print(gradOutput.shape)
        return -coeff * gradOutput, Variable(torch.tensor(0))

class GRL_noitnum(torch.autograd.Function):
    def __init__(self, coeff):
        self.coeff = coeff

    def forward(self, input):
        return input * 1.0

    def backward(self, gradOutput):
        return self.coeff * gradOutput

class Discriminator_share(nn.Module):
    def __init__(self, dim, coeff):
        super(Discriminator_share, self).__init__()
        # D = 480
        self.fc1 = nn.Linear(dim, dim)
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0.0)
        self.fc2 = nn.Linear(dim, 3)
        self.fc2.weight.data.normal_(0, 0.3)
        self.fc2.bias.data.fill_(0.0)
        self.ad_net = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(0.5),
            self.fc2
        )
        self.grl_layer = GRL_noitnum(coeff)
        #self.grl_layer = GRL()
    def forward(self, feature):
        adversarial_out = self.ad_net(self.grl_layer(feature))
        #adversarial_out = self.ad_net(GRL_module(feature))
        #adversarial_out = self.ad_net(feature)
        return adversarial_out

class Discriminator_share_grl(nn.Module):
    def __init__(self, dim):
        super(Discriminator_share_grl, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0.0)
        self.fc2 = nn.Linear(dim, 3)
        self.fc2.weight.data.normal_(0, 0.3)
        self.fc2.bias.data.fill_(0.0)
        self.ad_net = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(0.5),
            self.fc2
        )
        # self.grl_layer = GRL_noitnum(coeff)
        self.grl_layer = GRL()
    def forward(self, feature):
        adversarial_out = self.ad_net(self.grl_layer(feature))
        return adversarial_out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0.0)
        self.fc2 = nn.Linear(512, 3)
        self.fc2.weight.data.normal_(0, 0.3)
        self.fc2.bias.data.fill_(0.0)
        self.ad_net = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(0.5),
            self.fc2
        )
        self.grl_layer = GRL()

    def forward(self, feature):
        adversarial_out = self.ad_net(self.grl_layer(feature))
        return adversarial_out

class DG_model_vlad(nn.Module):
    def __init__(self, model, num_cluster, dim, alpha, adap):
        super(DG_model_vlad, self).__init__()
        if(model == 'resnet18'):
            self.backbone = Feature_Generator_ResNet18()
            if adap:
                self.embedder = Feature_Embedder_ResNet18_adap(num_cluster=num_cluster, dim=dim, alpha=alpha)
            else:
                self.embedder = Feature_Embedder_ResNet18(num_cluster=num_cluster, dim=dim, alpha=alpha)
        elif(model == 'maddg'):
            self.backbone = Feature_Generator_MADDG()
            self.embedder = Feature_Embedder_MADDG(num_cluster=num_cluster, dim=dim, alpha=alpha)
        else:
            print('Wrong Name!')
        # self.classifier = Classifier()
        self.classifier = Classifier_480(dim=num_cluster*dim)
        #self.classifier = Classifier_480(dim=num_cluster)
    def forward(self, input, norm_flag):
        feature = self.backbone(input)
        feature, soft_assign, local_ = self.embedder(feature, norm_flag)
        classifier_out = self.classifier(feature, norm_flag)

        return classifier_out, feature, soft_assign, local_

class DG_model_vlad_sharesp(nn.Module):
    def __init__(self, model, num_cluster, dim, par_dim, alpha, par_clu):
        super(DG_model_vlad_sharesp, self).__init__()
        self.total_dim = dim * num_cluster
        if(model == 'resnet18'):
            self.backbone = Feature_Generator_ResNet18()
            self.embedder = Feature_Embedder_ResNet18_nonorm_sharesp(num_cluster=num_cluster, dim=dim, par_dim=par_dim, alpha=alpha, par_clu=par_clu)
        elif(model == 'maddg'):
            self.backbone = Feature_Generator_MADDG()
            self.embedder = Feature_Embedder_MADDG_nonorm_sharesp(num_cluster=num_cluster, dim=dim, alpha=alpha, par_dim=par_dim)
        else:
            print('Wrong Name!')
        self.classifier = Classifier_480(dim=par_dim[0]+par_dim[1])
        # self.classifier = Classifier_480(dim=(par_dim[0]))
    def forward(self, input, norm_flag):
        feature = self.backbone(input)
        feature_share, feature_sp_benefit, soft_assign, local = self.embedder(feature, norm_flag)
        # feature = F.normalize(feature_share, p=2, dim=-1)
        feature = F.normalize(torch.cat((feature_share, feature_sp_benefit), -1), p=2, dim=-1)
        #feature[:, 0:128] = torch.zeros(1, 128).cuda().to(feature.device)
        classifier_out = self.classifier(feature, norm_flag)

        return classifier_out, F.normalize(feature_share, p=2, dim=-1), F.normalize(feature_sp_benefit, p=2, dim=-1), feature, soft_assign, local


if __name__ == '__main__':
    x = Variable(torch.ones(1, 3, 256, 256))
    model = DG_model_vlad()
    y, v = model(x, True)






