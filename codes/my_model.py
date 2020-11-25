import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from new_layers import self_conv, Q_A
import torch.nn.init as init
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, bitW, stride=1):
    "3x3 convolution with padding"
    return self_conv(in_planes, out_planes, bitW, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, num_bases, inplanes, planes, bitW, bitA, stride=1, downsample=None, quantize=True):
        super(BasicBlock, self).__init__()
        self.bitW = bitW
        self.bitA = bitA 
        self.num_bases = num_bases
        self.planes = planes
        self.relu = nn.ReLU()
        self.conv1 = nn.ModuleList([conv3x3(inplanes, planes, bitW, stride) for i in range(num_bases)])       
        self.bn1 = nn.ModuleList([nn.BatchNorm2d(planes) for i in range(num_bases)])
        self.conv2 = nn.ModuleList([conv3x3(planes, planes*self.expansion, bitW) for i in range(num_bases)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(planes) for i in range(num_bases)])
        self.scales = nn.ParameterList([nn.Parameter(torch.rand(1), requires_grad=True) for i in range(num_bases)])
        self.masks = nn.ParameterList([nn.Parameter(torch.rand(1), requires_grad=True) for i in range(num_bases)])
        self.downsample = downsample
        self.filter_gate1 = nn.ParameterList([nn.Parameter(torch.rand(planes*self.expansion), requires_grad=True) for i in range(num_bases)])        
        self.filter_gate2 = nn.ParameterList([nn.Parameter(torch.rand(planes*self.expansion), requires_grad=True) for i in range(num_bases)])          

    def quan_activations(self, x, bitA):
        if bitA == 32:
            return (x)
        else:
            return Q_A.apply(x)


    def forward(self, x):

        final_output = None
        if self.downsample is not None:
            x = self.quan_activations(x, self.bitA)
            residual = self.downsample(x)
        else:
            residual = x
            x = self.quan_activations(x, self.bitA)

        for conv1, conv2, bn1, bn2, scale, feature_gate1, feature_gate2, mask in zip(self.conv1, self.conv2, self.bn1, self.bn2, self.scales, self.filter_gate1, self.filter_gate2, self.masks):
            
            out = conv1(x)
            out = self.relu(out)
            out = bn1(out)
            out += residual

            """
            #filter search
            feature_gate = F.hardtanh((feature_gate1 + 0.5/50)*50, min_val=0, max_val=1).view(1, \
            self.planes, 1, 1).expand(x.shape[0], self.planes, out.shape[2], out.shape[3])
            out = feature_gate*out
            """
            out_new = self.quan_activations(out, self.bitA)

            out_new = conv2(out_new)
            out_new = self.relu(out_new)
            out_new = bn2(out_new)
            
            """
            feature_gate = F.hardtanh((feature_gate2 + 0.5/50)*50, min_val=0, max_val=1).view(1, \
            self.planes, 1, 1).expand(x.shape[0], self.planes, out_new.shape[2], out_new.shape[3])
            out_new = feature_gate*out_new
            """
            out_new += out

            #out_new = out_new * scale
            if final_output is None:
                final_output = out_new*F.hardtanh((mask+0.5), min_val=0, max_val=1)
            else:
                final_output += out_new*F.hardtanh((mask+0.5), min_val=0, max_val=1)

        return final_output/self.num_bases



class downsample_layer(nn.Module):
    def __init__(self, inplanes, planes, bitW, kernel_size=1, stride=1, bias=False):
        super(downsample_layer, self).__init__()
        self.conv = self_conv(inplanes, planes, bitW, kernel_size=kernel_size, stride=stride, bias=False)
        self.batch_norm = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        return x



class ResNet(nn.Module):

    def __init__(self, block, layers, bitW, bitA, base, num_classes=10):
        self.inplanes = 16
        self.num_bases1 = base[0]
        self.num_bases2 = base[1]
        self.num_bases3 = base[2]
        self.bitW = bitW
        self.bitA = bitA        
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, layers[0], self.num_bases1, stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], self.num_bases2, stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], self.num_bases3, stride=2)
        self.fc = nn.Linear(64, num_classes)


    def _make_layer(self, block, planes, blocks, num_bases,  stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = downsample_layer(self.inplanes, planes * block.expansion, self.bitW, 
                          kernel_size=1, stride=stride, bias=False)

        layers = []
        layers.append(block(num_bases[0], self.inplanes, planes, self.bitW, self.bitA, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(num_bases[i], self.inplanes, planes, self.bitW, self.bitA))

        return nn.Sequential(*layers)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out,out.size()[3])
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out


def resnet20(bitW, bitA, pruned, pretrained=False, **kwargs):
    """Constructs a ResNet-20 model.
    Args:
        pruned (bool): If True, use pre-pruned model structure
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pruned:
        base = [[5,3,4], [4,3,4], [4,4,5]]  ### base 4 equivalent
    else:
        base = [[5,5,5], [5,5,5], [5,5,5]]
    model = ResNet(BasicBlock, [3,3,3], bitW, bitA, base, **kwargs)
    print("==========ResNet architecture ===========")
    i = 0
    for layer in base:
        for block in layer:
            print("Block {} is expanded {} times".format(i, block))
            i += 1

    if pretrained:
        if pruned:
            load_dict = torch.load('./weights/20blockPruned_architecture.pth.tar')['state_dict']   ### base 4 equivalent
        else:
            load_dict = torch.load('./weights/20blockPruned_finetuned.pth.tar')['state_dict']
        model_dict = model.state_dict()
        model_keys = model_dict.keys()
        for name, param in load_dict.items():
            if name.replace('module.', '') in model_keys:
                model_dict[name.replace('module.', '')] = param    
        model.load_state_dict(model_dict)  
    return model


def resnet32(bitW, bitA, pruned,  pretrained=False, **kwargs):
    """Constructs a ResNet-32 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained
    """
    base = [[5,5,5],[5,5,5],[5,5,5]]
    model = ResNet(BasicBlock, [5,5,5], bitW, bitA, base,  **kwargs)
    if pretrained:
        load_dict = torch.load('./weights/.pth.tar')['state_dict']
        model_dict = model.state_dict()
        model_keys = model_dict.keys()
        for name, param in load_dict.items():
            if name.replace('module.','') in model_keys:
                model_dict[name.replace('module.','')] = param
        model.load_state_dict(model_dict)
    return model

