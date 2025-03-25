from model.unet import *
from math import sqrt


class CrossAttention(nn.Module):
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
    

class UNetText3D(nn.Module):
    def __init__(self, inChannel, outChannel, fChannel=32, bilinear=True):
        super(UNetText3D, self).__init__()
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
    model1=UNetText3D(1,1,fChannel=32)
    input=torch.randn(size=[1,1,64,64,64])
    text=torch.randn(size=[1,512,768])
    output1=model1(input,text)
    pass