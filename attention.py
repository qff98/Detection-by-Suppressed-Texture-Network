import torch
from torch import nn
import math

class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan=2048*2, out_chan=2048, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU()
        )
        self.ca = senet(out_chan)
        self.init_weight()

    def forward(self, x, y):
        fuse_fea = self.convblk(torch.cat((x, y), dim=1))
        fuse_fea = fuse_fea + self.ca(fuse_fea)
        return fuse_fea

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

class senet(nn.Module):
    def __init__(self,channel,ratio=16):
        super(senet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc       = nn.Sequential(
            nn.Linear(channel,channel//ratio,False),
            nn.ReLU(),
            nn.Linear(channel//ratio,channel,False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b,c,h,w=x.size()
        avg = self.avg_pool(x).view([b,c])  #b,c,h,w->b,c,1,1->b,c
        fc=self.fc(avg).view([b,c,1,1])

        return x*fc

class AttentionMixBlock(nn.Module):  #!!

    def __init__(self, in_dim):
        super(AttentionMixBlock, self).__init__()
        self.chanel_in = 2*in_dim

        # query conv
        self.q_conv1 = nn.Conv2d(2 * in_dim, in_dim, (1,1))
        self.q_conv2 = nn.Conv2d(2 * in_dim, in_dim, (1, 1))

        self.k_conv1 = nn.Conv2d(2 * in_dim, in_dim, (1, 1))
        self.k_conv2 = nn.Conv2d(2 * in_dim, in_dim, (1, 1))

        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.x_conv = nn.Conv2d(in_dim,in_dim,(1,1))
        self.y_conv = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.x_bn = nn.BatchNorm2d(in_dim)
        self.y_bn = nn.BatchNorm2d(in_dim)

        self.softmax = nn.Softmax(dim=-1)




    def forward(self, x, y):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        B, C, H, W = x.size()

        assert W == H
        x_att = torch.cat((x, y), dim=1)
        y_att = torch.cat((y, x), dim=1)
        q_x = self.q_conv1(x_att).view(-1,H,W) # [BC,H,W]
        q_y = self.q_conv2(y_att).view(-1,H,W)#[BC,H,W]
        a_q = torch.cat((q_x,q_y),dim=2) #[BC,H,W+W]

        k_x = self.k_conv1(x_att).view(-1,W,H) #[BC,W,H]
        k_y = self.k_conv2(y_att).view(-1,W,H) #[BC,W,H]
        a_k = torch.cat((k_x,k_y),dim=1) #[BC,W+W,H]

        energy = torch.bmm(a_q,a_k) #[BC,H,H]
        attention1 = self.softmax(energy).view(B,C,H,W)
        attention2 = self.softmax(energy.permute(0,2,1)).view(B,C,W,H) #[BC,W,H]

        attentionx = self.x_bn(self.x_conv(y * attention1))
        out_x = attentionx * self.gamma1 + x

        attentiony = self.y_bn(self.y_conv(x * attention2))
        out_y = attentiony * self.gamma2 + y ##!!

        return out_x, out_y  #

class CBAM_channel(nn.Module):
    def __init__(self, channel, ratio=16):
        super(CBAM_channel, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1) #(1)为输出出去的高和宽
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel,channel//ratio,False),
            nn.ReLU(),
            nn.Linear(channel // ratio,channel, False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        b,c,h,w = x.size()
        max_out = self.max_pool(x).view([b,c])
        avg_out = self.avg_pool(x).view([b,c])

        max_fc_out = self.fc(max_out)
        avg_fc_out = self.fc(avg_out)

        out = max_fc_out+avg_fc_out

        out = self.sigmoid(out).view([b,c,1,1])

        return out*x

class CBAM_spacial(nn.Module):
    def __init__(self, kernel_size=7):
        super(CBAM_spacial, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2,1,kernel_size,padding=padding,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out,_ = torch.max(x,dim=1, keepdim=True)
        pool_out = torch.cat([avg_out,max_out],dim=1)
        out = self.conv(pool_out)
        out = self.sigmoid(out)

        # mask2 = (out != out.max(dim=2, keepdim=True)[0]).to(dtype=torch.int32)
        # out = torch.mul(mask2, out)
        # mask3 = (out != out.max(dim=3, keepdim=True)[0]).to(dtype=torch.int32)
        # out = torch.mul(mask3, out)
        # print(out.size())

        return out
class CBAM(nn.Module):
    def __init__(self, channel, ratio=16,kernel_size=7):
        super(CBAM, self).__init__()
        self.CBAM_channel = CBAM_channel(channel,ratio=ratio)
        self.CBAM_spacial = CBAM_spacial(kernel_size=kernel_size)

    def forward(self,x):
        x = self.CBAM_channel(x)
        x = self.CBAM_spacial(x)*x
        return x

class AttentionREPAMT1(nn.Module):
    def __init__(self, in_channels):
        super(AttentionREPAMT1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.pa = CBAM_spacial(7)
    def forward(self, x):
        fea = self.conv(x)
        out = self.pa(fea)

        return out


class AttentionREPAMT2(nn.Module):
    def __init__(self, in_channels):
        super(AttentionREPAMT2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.pa = CBAM_spacial(7)
    def forward(self, x):
        fea = self.conv(x)
        out = self.pa(fea)

        return out

