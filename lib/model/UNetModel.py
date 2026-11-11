import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """ (conv 3x3 -> batchNorm -> ReLU) * 2""" 
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,kernel_size=3,padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self,x):
        return self.block(x)

class Down(nn.Module):
    """Encoder step: 2x2 max pool then doubleconv"""
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch,out_ch)

    def forward(self,x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Decoder step: transposed conv, concatenate the skip connection from the encoder then doubleconv"""
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch,in_ch//2,kernel_size=2,stride=2)
        self.conv = DoubleConv(in_ch//2 + out_ch, out_ch)

    def forward(self,x,skip):
        x = self.up(x)
        x = torch.cat([skip,x],dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]   # 32, 64, 128, 256, 512
        # encoder
        self.inc   = DoubleConv(in_ch, c[0])   # 3   -> 32,  full size
        self.down1 = Down(c[0], c[1])          # 32  -> 64,  1/2 size
        self.down2 = Down(c[1], c[2])          # 64  -> 128, 1/4
        self.down3 = Down(c[2], c[3])          # 128 -> 256, 1/8
        self.down4 = Down(c[3], c[4])          # 256 -> 512, 1/16 (bottleneck)
        # decoder
        self.up1 = Up(c[4], c[3])              # 512 -> 256, 1/8
        self.up2 = Up(c[3], c[2])              # 256 -> 128, 1/4
        self.up3 = Up(c[2], c[1])              # 128 -> 64,  1/2
        self.up4 = Up(c[1], c[0])              # 64  -> 32,  full size
        self.outc = nn.Conv2d(c[0], out_ch, kernel_size=1)   # 32 -> 1 logit per pixel

    def forward(self, x):
        x1 = self.inc(x)        # keep every encoder output: these are the skips
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)     # bottleneck
        y = self.up1(x5, x4)    # each Up gets the skip from the matching level
        y = self.up2(y, x3)
        y = self.up3(y, x2)
        y = self.up4(y, x1)
        return self.outc(y)     # logits, no activation


UNet2 = UNet   # the exercise's train script imported this name too