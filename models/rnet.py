import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class RNet(nn.Module):
    ''' RNet '''

    def __init__(self, num_landmarks=5):
        super(RNet, self).__init__()
    
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 28, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),  # prelu1
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
            nn.Conv2d(28, 48, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),  # prelu2
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
            nn.Conv2d(48, 64, kernel_size=2, stride=1),  # conv3
            nn.PReLU()  # prelu3
        )
        self.fc = nn.Linear(64*2*2, 128)
        self.prelu4 = nn.PReLU()  # prelu4
        # detection
        self.conv5_1 = nn.Linear(128, 1)
        # bounding box regression
        self.conv5_2 = nn.Linear(128, 4)
        # landmark localization
        self.conv5_3 = nn.Linear(128, num_landmarks*2)
        
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
        x = self.prelu4(x)
        # detection
        det = F.sigmoid(self.conv5_1(x))
        box = self.conv5_2(x)
        landmark = self.conv5_3(x)
        return det, box, landmark

if __name__ == "__main__":
    model = RNet(num_landmarks=98)
    model.load_state_dict(torch.load('./results/rnet/check_point/98_landmarks_model_050.pth', map_location='cpu'))
    torch.onnx.export(model, torch.randn(1, 3, 24, 24), 'onnx2ncnn/rnet.onnx')
    summary(model.cuda(), (3, 24, 24))
