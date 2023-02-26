import torch
import torch.nn as nn
import torch.nn.functional as F


#ULTIMUS 
class ULTIMUS(nn.Module):
  def __init__(self):
    super(ULTIMUS, self).__init__()
    self.k = nn.Linear(48,8)
    self.q = nn.Linear(48,8)
    self.v = nn.Linear(48,8)
    self.z = nn.Linear(8,48)

  def forward(self, x):
    K = torch.unsqueeze(self.k(x),-2)
    print(K.shape)
    Q = torch.unsqueeze(self.q(x),-1)
    print(Q.shape)
    V = torch.unsqueeze(self.v(x),-1)
    print(V.shape)

    scores = torch.bmm(Q, K) /  torch.sqrt(torch.tensor(8))
    print(scores.shape)

    AM = F.softmax(scores, dim=-1)
    print(AM.shape)
    Z = torch.bmm(AM, V)
    print(Z.shape)
    out = self.z(Z)
    print(out.shape)
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
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.u1(x)
        x = self.u2(x)
        x = self.u3(x)
        x = self.u4(x)
        out = self.linear(x)

        return F.log_softmax(out, dim=-1)


