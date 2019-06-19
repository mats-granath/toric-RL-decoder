import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from util import conv_to_fully_connected, pad_circular


# neural network CNN with one fully connected layer
class NN_11(nn.Module):

    def __init__(self, system_size, number_of_actions, device):
        super(NN_11, self).__init__()
        self.conv1 = nn.Conv2d(2, 128, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 120, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(120, 111, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(111, 104, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(104, 103, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(103, 90, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(90, 80 , kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(80, 73 , kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(73, 71 , kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(71, 64, kernel_size=3, stride=1)
        output_from_conv = conv_to_fully_connected(system_size, 3, 0, 1)
        self.linear1 = nn.Linear(64*int(output_from_conv)**2, 3)
        self.device = device

    def forward(self, x):
        x = pad_circular(x, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        n_features = np.prod(x.size()[1:])
        x = x.view(-1, n_features)
        x = self.linear1(x)
        return x


class NN_17(nn.Module):

    def __init__(self, system_size, number_of_actions, device):
        super(NN_17, self).__init__()
        self.conv1 = nn.Conv2d(2, 256, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 251, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(251, 250, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(250, 240, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(240, 240, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(240, 235, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(235, 233, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(233, 233, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(233, 229, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(229, 225, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(225, 223, kernel_size=3, stride=1, padding=1)
        self.conv13 = nn.Conv2d(223, 220 , kernel_size=3, stride=1, padding=1)
        self.conv14 = nn.Conv2d(220, 220 , kernel_size=3, stride=1, padding=1)
        self.conv15 = nn.Conv2d(220, 220 , kernel_size=3, stride=1, padding=1)
        self.conv16 = nn.Conv2d(220, 215 , kernel_size=3, stride=1, padding=1)
        self.conv17 = nn.Conv2d(215, 214 , kernel_size=3, stride=1, padding=1)
        self.conv18 = nn.Conv2d(214, 205 , kernel_size=3, stride=1, padding=1)
        self.conv19 = nn.Conv2d(205, 204 , kernel_size=3, stride=1, padding=1)
        self.conv20 = nn.Conv2d(204, 200 , kernel_size=3, stride=1)
        output_from_conv = conv_to_fully_connected(system_size, 3, 0, 1)
        self.linear1 = nn.Linear(200*int(output_from_conv)**2, number_of_actions)
        self.device = device

    def forward(self, x):
        x = pad_circular(x, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = F.relu(self.conv14(x))
        x = F.relu(self.conv15(x))
        x = F.relu(self.conv16(x))
        x = F.relu(self.conv17(x))
        x = F.relu(self.conv18(x))
        x = F.relu(self.conv19(x))
        x = F.relu(self.conv20(x))
        n_features = np.prod(x.size()[1:])
        x = x.view(-1, n_features)
        x = self.linear1(x)
        return x


'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=3):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        #self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])
