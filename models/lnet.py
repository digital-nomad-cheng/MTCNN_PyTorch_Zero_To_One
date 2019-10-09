import math

import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class ConvTanhAbsPool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pooling_size, padding=0):
        super(ConvTanhAbsPool, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.tanh = nn.Tanh()
        self.pool = nn.MaxPool2d(kernel_size=pooling_size, stride=2)
    def forward(self, x):
        x = self.conv(x)
        x = self.tanh(x).abs()
        x = self.pool(x)
        return x

class ConvTanhAbs(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(ConvTanhAbs, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.tanh = nn.Tanh()
    def forward(self, x):
        x = self.conv(x)
        x = self.tanh(x).abs()
        return x

class LNet(nn.Module):
    ''' LNet '''
    def __init__(self, num_landmarks=5):
        super(LNet, self).__init__()
        self.pre_layer = nn.Sequential(
            ConvTanhAbsPool(3, 32, 5, 2, 2),
            ConvTanhAbsPool(32, 48, 5, 2, 2),
            ConvTanhAbsPool(48, 64, 3, 3, 0),
            ConvTanhAbs(64, 80, 3, 0) 
        )
        self.fc = nn.Linear(80*4*4, 512)
        self.tanh5 = nn.Tanh()  # prelu5
        self.landmarks = nn.Linear(512, 2*num_landmarks)
        # weight initiation with xavier
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        # backend
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.tanh5(x).abs()
        landmark = self.landmarks(x)
        # return det, box, landmark
        return landmark
if __name__ == "__main__":
    model = LNet(num_landmarks=98)
    # model = ConvTanhAbsPool(3, 32, 3, 2)
    summary(model.cuda(), (3, 60, 60))
