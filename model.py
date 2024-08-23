import torch.nn as nn

class CNN(nn.Module):
    '''
    输入维度: (1, 28, 28)

    CNN经过卷积Conv2d(1, 32, 3),即1个通道，输出通道数32，卷积核大小:3x3的输出为
    (28-3)/1 + 1 = 26

    输出维度: (32, 26, 26)

    注意torch.nn.CrossEntropyLoss自带了softmax，因此最后不需要softmax避免双重softmax导致权值差异压缩，使得权值的分布产生平均化的趋势
    '''
    
    def __init__(self):
        super(CNN, self).__init__()
        # self.conv = nn.Sequential(  # (1,28,28)
        #     nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5)),
        #     nn.ReLU(),  # (6,24,24)
        #     nn.MaxPool2d((2, 2)),  # (6,12,12)
        #     nn.Conv2d(6, 12, 3),
        #     nn.ReLU(),  # (12,10,10)
        #     nn.MaxPool2d((2, 2))  # (12,5,5)
        # )

        # self.fc = nn.Sequential(nn.Linear(12 * 5 * 5, 128), nn.ReLU(),
        #                         nn.Linear(128, 32), nn.ReLU(),
        #                         nn.Linear(32, 10))

        # https://github.com/pytorch/examples/blob/main/mnist/main.py
        self.cnn = nn.Sequential( # (1,28,28)
            nn.Conv2d(1, 32, 3), 
            nn.ReLU(), # (32, 26, 26 )
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),# (64, 24, 24)
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.25), # (64 ,12, 12)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(64 * 12 * 12 , 128),
            nn.Dropout(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x