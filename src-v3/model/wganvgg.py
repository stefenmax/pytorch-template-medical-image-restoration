## TMI-2018 Low Dose CT Image Denoising Using a Generative Adversarial Network with Wasserstein Distance and Perceptual Loss
## https://arxiv.org/abs/1708.00961

import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model(args, parent=False):
    return WGANVGG(args)

## korean version
#class WGANVGG(nn.Module):
#    def __init__(self, args):
#        super(WGANVGG, self).__init__()
#        layers = [nn.Conv2d(1,32,3,1,1), nn.ReLU()]
#        for i in range(2, 8):
#            layers.append(nn.Conv2d(32,32,3,1,1))
#            layers.append(nn.ReLU())
#        layers.extend([nn.Conv2d(32,1,3,1,1), nn.ReLU()])
#        self.net = nn.Sequential(*layers)
#
#    def forward(self, x):
#        out = self.net(x)
#        return out

# Convert version


class WGANVGG(nn.Module):
    def __init__(self, args, in_channels=1, out_channels=1, padding=0):
        super(WGANVGG, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=padding, bias=False)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=padding, bias=False)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=padding, bias=False)
        self.conv5 = nn.Conv2d(32, 32, 3, padding=padding, bias=False)
        self.conv6 = nn.Conv2d(32, 32, 3, padding=padding, bias=False)
        self.conv7 = nn.Conv2d(32, 32, 3, padding=padding, bias=False)
        self.conv8 = nn.Conv2d(32, out_channels, 3, padding=padding, bias=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        return x
