import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class ONet(nn.Module):
    ''' ONet '''

    def __init__(self, num_landmarks=5):
        super(ONet, self).__init__()

        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),  # prelu1
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),  # prelu2
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # conv3
            nn.PReLU(), # prelu3
            nn.MaxPool2d(kernel_size=2,stride=2), # pool3
            nn.Conv2d(64, 128,kernel_size=2,stride=1), # conv4
            nn.PReLU() # prelu4
        )
        self.fc = nn.Linear(128*2*2, 256)
        self.prelu5 = nn.PReLU()  # prelu5
        # detection
        self.conv6_1 = nn.Linear(256, 1)
        # bounding box regression
        self.conv6_2 = nn.Linear(256, 4)
        # lanbmark localization, modify here to change the number of landmarks
        self.conv6_3 = nn.Linear(256, num_landmarks*2)

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
        x = self.prelu5(x)
        # detection
        det = F.sigmoid(self.conv6_1(x))
        box = self.conv6_2(x)
        landmark = self.conv6_3(x)
        return det, box, landmark

if __name__ == "__main__":
    model = ONet(num_landmarks=98)
    model.load_state_dict(torch.load('./results/onet/check_point/98_landmarks_model_050.pth', map_location='cpu'))
    torch.onnx.export(model, torch.randn(1, 3, 48, 48), 'onnx2ncnn/onet.onnx')
    summary(model.cuda(), (3, 48, 48))
