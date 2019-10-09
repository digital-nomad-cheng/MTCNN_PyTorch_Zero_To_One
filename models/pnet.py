import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class PNet(nn.Module):
    ''' PNet '''

    def __init__(self):
        super(PNet, self).__init__()

        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),  # PReLU1
            nn.MaxPool2d(kernel_size=2, stride=2),  # pool1
            nn.Conv2d(10, 16, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),  # PReLU2
            nn.Conv2d(16, 32, kernel_size=3, stride=1),  # conv3
            nn.PReLU()  # PReLU3
        )

        # detection: decides whether this proposal contains face
        self.conv4_1 = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        # bounding box regresion
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)

        # weight initiation with xavier
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        x = self.pre_layer(x)
        label = F.sigmoid(self.conv4_1(x))
        offset = self.conv4_2(x)
        return label, offset

if __name__ == "__main__":
    model = PNet()
    model.load_state_dict(torch.load('./results/pnet/check_point/98_landmarks_model_050.pth', map_location='cpu'))
    torch.onnx.export(model, torch.randn(1, 3, 12, 12), 'onnx2ncnn/pnet.onnx')
    summary(model.cuda(), (3, 12, 12))
