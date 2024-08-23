import torch.nn as nn

class CNN(nn.Module):
    '''定义模型'''

    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(  # (1,28,28)
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5)),
            nn.ReLU(),  # (6,24,24)
            nn.MaxPool2d((2, 2)),  # (6,12,12)
            nn.Conv2d(6, 12, 3),
            nn.ReLU(),  # (12,10,10)
            nn.MaxPool2d((2, 2))  # (12,5,5)
        )

        self.fc = nn.Sequential(nn.Linear(12 * 5 * 5, 128), nn.ReLU(),
                                nn.Linear(128, 32), nn.ReLU(),
                                nn.Linear(32, 10))
        self.net = nn.Sequential(  # (1,28,28)
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5)),
            nn.ReLU(),  # (6,24,24)
            nn.MaxPool2d((2, 2)),  # (6,12,12)
            nn.Conv2d(6, 12, 3),
            nn.ReLU(),  # (12,10,10)
            nn.MaxPool2d((2, 2)),  # (12,5,5)
            nn.Flatten(),
            nn.Linear(12 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 10))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x