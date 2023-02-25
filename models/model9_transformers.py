import torch
import torch.nn as nn
import torch.nn.functional as F


#ULTIMUS 
class ULTIMUS(nn.Module):
  def __init__(self):
    super(ULTIMUS, self).__init__()


  def forward(self, x):
    K = nn.Linear(48,8)(x)
    Q = nn.Linear(48,8)(x)
    V = nn.Linear(48,8)(x)

    scores = torch.matmul(Q.transpose(-2, -1), K) /  torch.sqrt(8)

    AM = F.softmax(scores, dim=-1)
    Z = torch.matmul(scores, V)
    out = nn.Linear(8,48)(Z)
    return out

class Vit(nn.Module):
    def __init__(self):
        super(Vit, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )
        self.gap = nn.AvgPool2d(5) 
        self.u1 = ULTIMUS()
        self.u2 = ULTIMUS()
        self.u3 = ULTIMUS()
        self.u4 = ULTIMUS()
        self.linear = nn.Linear(48, 10)



    def forward(self, x):
        x =  self.layer1(x)
        print(x.shape)
        x = self.gap(x)
        print(x.shape)
        x = x.view(x.size(0), -1)
        print(x.shape)
        x = self.u1(x)
        x = self.u2(x)
        x = self.u3(x)
        x = self.u4(x)
        out = self.linear(x)

        return F.log_softmax(out, dim=-1)


