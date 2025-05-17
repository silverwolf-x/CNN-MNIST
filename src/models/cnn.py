from torch import nn


class SimpleCNN(nn.Module):
    """
    输入维度: (1, 28, 28)

    CNN经过卷积Conv2d(1, 32, 3),即1个通道，输出通道数32，卷积核大小:3x3的输出为
    (28-3)/1 + 1 = 26

    输出维度: (32, 26, 26)

    注意torch.nn.CrossEntropyLoss自带了softmax，因此最后不需要softmax避免双重softmax导致权值差异压缩，使得权值的分布产生平均化的趋势
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # https://github.com/pytorch/examples/blob/main/mnist/main.py
        self.cnn = nn.Sequential(  # (1,28,28)
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),  # (32, 26, 26 )
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),  # (64, 24, 24)
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.25),  # (64 ,12, 12)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(nn.Linear(64 * 12 * 12, 128), nn.Dropout(), nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
