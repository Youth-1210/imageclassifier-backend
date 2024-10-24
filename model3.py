# backend/model/model3.py

import torch
from torch import nn
from torchvision import models

class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self):
        super(MultiScaleFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=1, padding=3)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # 提取不同尺度的特征并进行池化
        features1 = self.relu(self.conv1(x))
        features1 = self.pool(features1)
        features1 = self.pool(features1)

        features2 = self.relu(self.conv2(x))
        features2 = self.pool(features2)
        features2 = self.pool(features2)

        features3 = self.relu(self.conv3(x))
        features3 = self.pool(features3)
        features3 = self.pool(features3)

        # 拼接不同尺度的特征
        combined_features = torch.cat((features1, features2, features3), dim=1)
        return combined_features

class DualBranchImageModel(nn.Module):
    def __init__(self, num_classes):
        super(DualBranchImageModel, self).__init__()
        # 分支1: 多尺度特征提取
        self.branch1 = MultiScaleFeatureExtractor()

        # 分支2: MobileNetV2
        self.branch2 = models.mobilenet_v2(pretrained=False)
        self.branch2.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(self.branch2.last_channel, 128),
            nn.ReLU(inplace=True)
        )

        # 融合层
        # 多尺度特征输出48个通道（16 * 3），假设图像大小为224x224，经过两次池化后为56x56
        self.fc = nn.Sequential(
            nn.Linear(16 * 56 * 56 * 3 + 128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # 图像输入到分支1（多尺度特征提取）
        branch1_output = self.branch1(x)
        branch1_flat = branch1_output.view(branch1_output.size(0), -1)

        # 图像输入到分支2（MobileNetV2）
        branch2_output = self.branch2(x)

        # 融合两个分支的特征
        combined_features = torch.cat((branch1_flat, branch2_output), dim=1)

        # 输出分类结果
        output = self.fc(combined_features)
        return output
