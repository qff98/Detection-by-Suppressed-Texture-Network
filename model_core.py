import torch
import torch.nn as nn
import torch.nn.functional as F
from xception_tiny import return_pytorch04_xception
from attention import AttentionMixBlock,AttentionREPAMT2,AttentionREPAMT1,CBAM_spacial
from AMTEN import REAMTEN




class Two_Stream_Net(nn.Module):
    def __init__(self,
                 weights: str='',
                 num_classes=1000,
                 modelchoice: str='',
                 ):
        super().__init__()
        if weights != "":
            pre=True
        else: pre=False
        if modelchoice == 'xception':
            self.xception_rgb = self._init_xcep_fad(pre,weights)
            self.xception_clue = self._init_xcep_fad(pre,weights)
            # self.fusion = FeatureFusionModule()
        self.model = modelchoice


        self.amten = REAMTEN()
        self.amtenconv = nn.Conv2d(12, 32, 3, 2, 0, bias=False)
        self.dp = nn.Dropout(p=0.5)
        self.amtenatt1 = CBAM_spacial(7)
        self.amtenatt2 = CBAM_spacial(7)
        self.feature13 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.feature11 = nn.Conv2d(32, 32, 1, 1, 0, bias=True)
        self.feature23 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.feature21 = nn.Conv2d(64, 64, 1, 1, 0, bias=True)

        self.AttentionMixBlock0 = AttentionMixBlock(in_dim=728)
        self.AttentionMixBlock1 = AttentionMixBlock(in_dim=1024)


        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(4096 if self.model == 'xception' else 512, num_classes)

    def _init_xcep_fad(self,pre,weights):
        premode =  return_pytorch04_xception(pre,weights)

        return premode


    def _norm_feature(self, x):
        x = self.relu(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        return x

    def classifier(self, features):
        x = self.dp(features)
        out = self.fc(x)



        return out


    def forward(self, x):
        x_clue = self.amten(x)

        if self.model == 'xception':
            x_clue1 = self.amtenconv(x_clue)
            x_clue3 = self.xception_clue.fea_part1_1(x_clue1)

            x = self.xception_rgb.fea_part1_0(x)
            x = self.xception_rgb.fea_part1_1(x)

            xfea13 = self.feature13(x) - x
            xfea11 = self.feature11(x)-x
            xfea1 = xfea13+xfea11
            x_clue3 = x_clue3 + xfea1
            x_clue3 = self.relu(x_clue3)

            x = self.amtenatt1(x_clue3) * x

            x_clue3 = self.xception_clue.fea_part1_2(x_clue3)

            x = self.xception_rgb.fea_part1_2(x)

            xfea23 = self.feature23(x) - x
            xfea21 = self.feature21(x) - x
            xfea2 = xfea23 + xfea21
            x_clue3 = x_clue3 + xfea2
            x_clue3 = self.relu(x_clue3)

            x = self.amtenatt2(x_clue3) * x

            clue = self.xception_clue.fea_part2_0(x_clue3)

            x = self.xception_rgb.fea_part2_0(x)


            clue = self.xception_clue.fea_part2_1(clue)

            x = self.xception_rgb.fea_part2_1(x)

            x,clue = self.AttentionMixBlock0(x,clue)

            clue = self.xception_clue.fea_part2_2(clue)

            x = self.xception_rgb.fea_part2_2(x)

            x, clue = self.AttentionMixBlock1(x, clue)

            clue = self.xception_clue.fea_part2_3(clue)

            x = self.xception_rgb.fea_part2_3(x)



        clue = self._norm_feature(clue)
        x = self._norm_feature(x)
        fea = torch.cat((x, clue), dim=1)

        out = self.classifier(fea)
        return out


    
