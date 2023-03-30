# Deep Convolutional Neural Network for Inverse Problems in Imaging
# https://ieeexplore.ieee.org/document/7949028
# original U-Net code:https://github.com/milesial/Pytorch-UNet
import torch.nn as nn
from model import common


def make_model(args, parent=False):
    return FBPCONV(args)

    
class FBPCONV(nn.Module):
    def __init__(self, args, bilinear=False):
        super(FBPCONV, self).__init__()
        self.args = args
        self.n_channels = args.n_colors
        self.n_classes = args.n_colors
        self.bilinear = bilinear
      
        self.inc = (common.DoubleConv(self.n_channels, 64))
        self.down1 = (common.Down(64, 128))
        self.down2 = (common.Down(128, 256))
        self.down3 = (common.Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (common.Down(512, 1024 // factor))
        self.up1 = (common.Up(1024, 512 // factor, bilinear))
        self.up2 = (common.Up(512, 256 // factor, bilinear))
        self.up3 = (common.Up(256, 128 // factor, bilinear))
        self.up4 = (common.Up(128, 64, bilinear))
        self.outc = (common.OutConv(64, self.n_classes))

    def forward(self, x):
        residual = x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        logits = logits + residual
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)