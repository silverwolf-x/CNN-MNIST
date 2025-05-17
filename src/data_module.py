import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


class MNISTDataset(Dataset):
    def __init__(self, train=True, root="./", download=True, transform=None):
        self.dataset = datasets.MNIST(
            root=root, train=train, download=download, transform=transform or transforms.ToTensor()
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return {"input": image, "label": torch.tensor(label, dtype=torch.long)}


class DataModule(LightningDataModule):
    def __init__(self, data_dir: str = "../", batch_size: int = 64, num_workers: int = 0, valid_ratio: float = 0.1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_ratio = valid_ratio
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.setup()

    def setup(self, stage=None):
        full_train = MNISTDataset(train=True, root=self.data_dir, download=True, transform=self.transform)

        n_total = len(full_train)
        n_valid = int(n_total * self.valid_ratio)
        n_train = n_total - n_valid

        self.train_ds, self.val_ds = random_split(full_train, [n_train, n_valid])

        self.test_ds = MNISTDataset(train=False, root=self.data_dir, download=True, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )


if __name__ == "__main__":
    from hydra import compose, initialize
    from lightning.pytorch import Trainer, seed_everything

    seed_everything(42, workers=True)  # 设置随机种子

    with initialize(config_path="../configs", version_base=None):
        cfg: DictConfig = compose(
            config_name="default", overrides=["hydra.output_subdir=null"]
        )  # 不生成子目录，直接在当前目录运行

    dm = DataModule()
    dm.setup()
    # 打印dataloader的一个样例
    for batch in dm.train_dataloader():
        print(batch)
        print(batch["input"].shape, batch["label"].shape)
        break
