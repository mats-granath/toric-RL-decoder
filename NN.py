import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#import torchvision.transforms as T

def conv_to_fully_connected(input_size, filter_size, padding, stride):
    return (input_size - filter_size + 2 * padding)/ stride + 1


def pad_circular(x, pad):
    x = torch.cat([x, x[:,:,:,0:pad]], dim=3)
    x = torch.cat([x, x[:,:, 0:pad,:]], dim=2)
    x = torch.cat([x[:,:,:,-2 * pad:-pad], x], dim=3)
    x = torch.cat([x[:,:, -2 * pad:-pad,:], x], dim=2)
    return x


class NN_0(nn.Module):

    def __init__(self, system_size, number_of_actions, device):
        super(NN_0, self).__init__()
        self.conv1 = nn.Conv2d(2, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.linear1 = nn.Linear(64*system_size*system_size, 512)
        self.linear2 = nn.Linear(512, 3) # 0: x, 1: y, 3: z

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        n_features = np.prod(x.size()[1:])
        x = x.view(-1, n_features)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class NN_7(nn.Module):

    def __init__(self, system_size, number_of_actions, device):
        super(NN_7, self).__init__()
        self.device = device

        sizes_linear = [2*system_size**2, 2028, 1024, 32, 2024, 1056]
        self.linear = [nn.Linear(sizes_linear[i-1], sizes_linear[i]) for i in range(1, len(sizes_linear))]

        self.output = nn.Linear(sizes_linear[-1], number_of_actions)

    def forward(self, x):

        n_features = np.prod(x.size()[1:])
        x = x.view(-1, n_features)

        for layer in self.linear:
            layer.to(self.device)
            x = F.relu((layer(x)))

        x = self.output(x)
        return x

class NN_8(nn.Module):

    def __init__(self, system_size, number_of_actions, device):
        super(NN_8, self).__init__()
        self.device = device

        sizes_linear = [2*system_size**2, 2028, 1024, 12, 2024, 1056]
        self.linear = [nn.Linear(sizes_linear[i-1], sizes_linear[i]) for i in range(1, len(sizes_linear))]

        self.output = nn.Linear(sizes_linear[-1], number_of_actions)

    def forward(self, x):

        n_features = np.prod(x.size()[1:])
        x = x.view(-1, n_features)

        for layer in self.linear:
            layer.to(self.device)
            x = F.relu((layer(x)))

        x = self.output(x)
        return x


class NN_9(nn.Module):

    def __init__(self, system_size, number_of_actions, device):
        super(NN_9, self).__init__()
        self.conv1 = nn.Conv2d(2, 256, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 71, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(71, 64, kernel_size=3, stride=1)
        self.linear1 = nn.Linear(64, 129)        
        self.linear2 = nn.Linear(129, 3) # 0: x, 1: y, 3: z

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        n_features = np.prod(x.size()[1:])
        x = x.view(-1, n_features)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


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




class NN_12(nn.Module):

    def __init__(self, system_size, number_of_actions, device):
        super(NN_12, self).__init__()

        self.conv1 = nn.Conv2d(2, 200, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(200, 190, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(190, 189, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(189, 160, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(160, 150, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(150, 132, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(132, 128, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(128, 120, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(120, 111, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(111, 104, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(104, 103, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(103, 90, kernel_size=3, stride=1, padding=1)
        self.conv13 = nn.Conv2d(90, 80 , kernel_size=3, stride=1, padding=1)
        self.conv14 = nn.Conv2d(80, 73 , kernel_size=3, stride=1, padding=1)
        self.conv15 = nn.Conv2d(73, 71 , kernel_size=3, stride=1, padding=1)
        self.conv16 = nn.Conv2d(71, 64, kernel_size=3, stride=1)
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
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = F.relu(self.conv14(x))
        x = F.relu(self.conv15(x))
        x = F.relu(self.conv16(x))


        n_features = np.prod(x.size()[1:])
        x = x.view(-1, n_features)
        x = self.linear1(x)

        return x


class NN_13(nn.Module):

    def __init__(self, system_size, number_of_actions, device):
        super(NN_13, self).__init__()
        self.conv1 = nn.Conv2d(2, 300, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(300, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 189, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(189, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 100, kernel_size=3, stride=1)
        output_from_conv = conv_to_fully_connected(system_size, 3, 0, 1)
        self.linear = nn.Linear(100*int(output_from_conv)**2, 3)

    def forward(self, x):
        x = pad_circular(x, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        n_features = np.prod(x.size()[1:])
        x = x.view(-1, n_features)
        x = self.linear(x)
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
        # output_from_conv = conv_to_fully_connected(system_size, 3, 0, 1)
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
        

class NN_18(nn.Module):

    def __init__(self, system_size, number_of_actions, device):
        super(NN_18, self).__init__()
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
        self.conv20 = nn.Conv2d(204, 204 , kernel_size=3, stride=1, padding=1)
        self.conv21 = nn.Conv2d(204, 204 , kernel_size=3, stride=1, padding=1)
        self.conv22 = nn.Conv2d(204, 200 , kernel_size=3, stride=1, padding=1)
        self.conv23 = nn.Conv2d(200, 200 , kernel_size=3, stride=1, padding=1)
        self.conv24 = nn.Conv2d(200, 200 , kernel_size=3, stride=1, padding=1)
        self.conv25 = nn.Conv2d(200, 200 , kernel_size=3, stride=1)
        # output_from_conv = conv_to_fully_connected(system_size, 3, 0, 1)
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
        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = F.relu(self.conv23(x))
        x = F.relu(self.conv24(x))
        x = F.relu(self.conv25(x))

        n_features = np.prod(x.size()[1:])
        x = x.view(-1, n_features)
        x = self.linear1(x)

        return x
