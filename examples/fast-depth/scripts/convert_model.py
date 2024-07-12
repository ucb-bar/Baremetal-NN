
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from barstools import TorchConverter

torch.manual_seed(0)


class MobileNetSkipAdd(nn.Module):
    def __init__(self):
        super(MobileNetSkipAdd, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(16, 56, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(56, 56, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=56, bias=False),
            nn.BatchNorm2d(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(56, 88, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(88, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(88, 88, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=88, bias=False),
            nn.BatchNorm2d(88, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(88, 120, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(120, 120, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=120, bias=False),
            nn.BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(120, 144, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False),
            nn.BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(144, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256, bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(256, 408, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(408, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(408, 408, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=408, bias=False),
            nn.BatchNorm2d(408, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(408, 376, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(376, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(376, 376, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=376, bias=False),
            nn.BatchNorm2d(376, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(376, 272, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(272, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(272, 272, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=272, bias=False),
            nn.BatchNorm2d(272, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(272, 288, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=288, bias=False),
            nn.BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(288, 296, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(296, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(296, 296, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=296, bias=False),
            nn.BatchNorm2d(296, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(296, 328, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(328, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(328, 328, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=328, bias=False),
            nn.BatchNorm2d(328, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(328, 480, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False),
            nn.BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(480, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
        )
        self.decode_conv1 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=512, bias=False),
                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(512, 200, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ),
        )
        self.decode_conv2 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(200, 200, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=200, bias=False),
                nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(200, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ),
        )
        self.decode_conv3 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=256, bias=False),
                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 120, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ),
        )
        self.decode_conv4 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(120, 120, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=120, bias=False),
                nn.BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(120, 56, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ),
        )
        self.decode_conv5 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(56, 56, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=56, bias=False),
                nn.BatchNorm2d(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(56, 16, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ),
        )
        self.decode_conv6 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # skip connections: dec4: enc1
        # dec 3: enc2 or enc3
        # dec 2: enc4 or enc5
        for i in range(14):
            layer = getattr(self, 'conv{}'.format(i))
            x = layer(x)
            
            if i==1:
                x1 = x
            elif i==3:
                x2 = x
            elif i==5:
                x3 = x
        for i in range(1,6):
            layer = getattr(self, 'decode_conv{}'.format(i))
            x = layer(x)
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            if i==4:
                x = x + x1
            elif i==3:
                x = x + x2
            elif i==2:
                x = x + x3
        x = self.decode_conv6(x)
        return x
    

# Tracing the module
m = MobileNetSkipAdd()

m.load_state_dict(torch.load("checkpoints/mobilenet_skip_add.pth", map_location=torch.device('cpu')))
m.eval()



# TorchConverter(m).print()

input_file = "data/visual_1.png"
img = cv2.imread(input_file)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# convert to tensor
img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0

test_input = torch.tensor(img).unsqueeze(0)

print(test_input)

with torch.no_grad():
    output = m.forward(test_input)
    print(output)

output = TorchConverter(m).convert(test_input)
# print(output)