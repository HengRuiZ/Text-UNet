import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class DoubleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv3d = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv3d(x)


class Down3D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up3D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class CrossAttention(nn.Module):
    # Cross attention module using both text and 3D image features input.
    def __init__(self, channel_I,channel_T):
        super().__init__()
        self.text_fc11=nn.Linear(channel_T,channel_I)
        self.text_fc12=nn.Linear(channel_T,channel_I)
        self.im_fc11=nn.Conv3d(channel_I,channel_I,1)
        self.im_fc12=nn.Conv3d(channel_I,channel_I,1)
        self.h2_conv=nn.Conv3d(channel_I,channel_I,1)
        self.out_conv=nn.Conv3d(channel_I,channel_I,1)
        return
    def forward(self,im,txt):
        txt11=self.text_fc11(txt)
        txt12=self.text_fc12(txt)
        im11=self.im_fc11(im)
        im12=self.im_fc12(im)
        im11=im11.view(im11.size(0), im11.size(1), -1)
        h1=torch.bmm(txt11,im11)/sqrt(txt11.shape[2])
        h1=F.softmax(h1,dim=-1)
        h1=h1.transpose(1,2)
        h2=torch.bmm(h1,txt12)/sqrt(h1.shape[2])
        h2=h2.transpose(1,2).view(im.size())
        h2=self.h2_conv(h2)
        h3=h2*im12
        h4=self.out_conv(h3)
        return h4
    

class Text-UNet-3D(nn.Module):
    def __init__(self, inChannel, outChannel, fChannel=32, bilinear=True):
        super(Text-UNet-3D, self).__init__()
        self.inChannel = inChannel
        self.outChannel = outChannel
        self.fChannel=fChannel
        self.bilinear = bilinear

        self.inc = DoubleConv3D(inChannel, fChannel)
        self.down1 = Down3D(fChannel, fChannel*2)
        self.down2 = Down3D(fChannel*2, fChannel*4)
        self.down3 = Down3D(fChannel*4, fChannel*8)
        factor = 2 if bilinear else 1
        self.down4 = Down3D(fChannel*8, fChannel*16 // factor)

        self.CA1=CrossAttention(32,768)
        self.CA2=CrossAttention(64,768)
        self.CA3=CrossAttention(128,768)
        self.CA4=CrossAttention(256,768)
        self.CA5=CrossAttention(256,768)

        self.up1 = Up3D(fChannel*16, fChannel*8 // factor, bilinear)
        self.up2 = Up3D(fChannel*8, fChannel*4 // factor, bilinear)
        self.up3 = Up3D(fChannel*4, fChannel*2 // factor, bilinear)
        self.up4 = Up3D(fChannel*2, fChannel, bilinear)
        self.outc = OutConv3D(fChannel, outChannel)

    def forward(self, x, text):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        #CrossAttention bottleneck
        x1=self.CA1(x1,text)
        x2=self.CA2(x2,text)
        x3=self.CA3(x3,text)
        x4=self.CA4(x4,text)
        x5=self.CA5(x5,text)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    

if __name__ == '__main__':
    model1=Text-UNet-3D(1,1,fChannel=32)
    input=torch.randn(size=[1,1,64,64,64])
    text=torch.randn(size=[1,512,768])
    output1=model1(input,text)
    pass
