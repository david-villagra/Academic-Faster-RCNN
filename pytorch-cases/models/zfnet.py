"""
ZFNET in pytorch

David Villagrá
"""

import torch.nn as nn
import torch
from conf import settings


def nrelu():
    if settings.ACT is 'relu':
        nn.ReLU(inplace=True)
    elif settings.ACT is 'lrelu':
        nn.LeakyReLU(inplace=True)


class ZFNet(nn.Module):

    def __init__(self, num_classes=20):
        super(ZFNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            #nrelu(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(96, 256, kernel_size=5, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            #nrelu(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            #nrelu(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            #nrelu(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            #nrelu(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            #nrelu(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            #nrelu(),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def zfnet():
    return ZFNet()
