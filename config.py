import torch.backends.cudnn

from dataclasses import dataclass
from model import CNN


@dataclass
class Config:
    device: str = "cuda"
    model: type = CNN
    seed: int = 45
    batch_size: int = 1024
    valid_ratio: int = 0.1
    folder: str = "run"
    early_stop: int = 3
    learning_rate: int = 0.0001
    n_epoches: int = 10


config = Config()

if config.device == "cuda":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
