import torch.nn as nn
import torchvision.models as models


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.resnet = models.resnet18()

        # 1. 修改输入层以适配1通道图像
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # 2. 去掉最大池化层，避免28x28图像被下采样过多
        self.resnet.maxpool = nn.Identity()

        # 3. 修改输出层
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
