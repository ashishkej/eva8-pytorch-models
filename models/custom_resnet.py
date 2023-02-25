import torch
import torch.nn as nn
import torch.nn.functional as F


#ResBlock
class ResBlock(nn.Module):
  def __init__(self, in_ch, out_ch):
    super(ResBlock, self).__init__()
    self.in_chan = in_ch
    self.out_chan = out_ch

    self.resblock = nn.Sequential(
        nn.Conv2d(in_channels=self.in_chan, out_channels=self.out_chan, kernel_size=(3, 3), padding=1, bias=False),
        nn.BatchNorm2d(self.out_chan),
        nn.ReLU(),
        nn.Conv2d(in_channels=self.out_chan, out_channels=self.out_chan, kernel_size=(3, 3), padding=1, bias=False),
        nn.BatchNorm2d(self.out_chan),
        nn.ReLU(),
    )

  def forward(self, x):
    x = self.resblock(x)
    return x

class MyResnet(nn.Module):
    def __init__(self):
        super(MyResnet, self).__init__()

        self.preplayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.layer1_X = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer1_R = ResBlock(128,128)

        self.layer2_X = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer3_X = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3_R = ResBlock(512,512)

        self.maxpool_4 = nn.MaxPool2d(4) 
        self.linear = nn.Linear(512, 10)


    def forward(self, x):
        x =  self.preplayer(x)
        x1 =  self.layer1_X(x)
        R1 = self.layer1_R(x1)
        x2 = x1 + R1
        x2 = self.layer2_X(x2)
        x3 =  self.layer3_X(x2)
        R3 = self.layer3_R(x3)
        x3 = x3 + R3
        x4 = self.maxpool_4(x3)
        out = x4.view(x4.size(0), -1)
        out = self.linear(out)

        return F.log_softmax(out, dim=-1)


