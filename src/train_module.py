import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, ConfusionMatrix
from torchmetrics.classification import MulticlassF1Score


class TrainModule(LightningModule):
    """ """

    def __init__(self, model: torch.nn.Module, num_classes: int, cfg: DictConfig):
        """
        Args:
            model: torch.nn.Module, 模型
            num_classes: int, 类别数
            cfg: DictConfig, 配置文件, 入参为cfg.train_module
        """
        super().__init__()
        self.model = model
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.test_cm = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self.acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_cm.reset()
        self.cfg = cfg
        self.save_hyperparameters(logger=False)  # 保存init传参到checkpoint中,不要存到logger避免启动写入额外耗时

    def forward(self, x):
        return self.model(x)

    def log(self, name, value, **kwargs):
        # 如果没有指定prog_bar，默认True
        if "prog_bar" not in kwargs:
            kwargs["prog_bar"] = True
        super().log(name, value, **kwargs)

    def training_step(self, batch):
        logits = self(batch["input"])
        loss = self.loss_fn(logits, batch["label"])
        self.log("train_loss", loss)
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", current_lr)

        return loss

    def validation_step(self, batch):
        logits = self(batch["input"])
        loss = self.loss_fn(logits, batch["label"])
        preds = torch.argmax(logits, dim=1)
        f1 = self.f1(preds, batch["label"])
        self.log("val_loss", loss)
        self.log("val_f1", f1)
        return loss

    def test_step(self, batch):
        logits = self(batch["input"])
        loss = self.loss_fn(logits, batch["label"])
        preds = torch.argmax(logits, dim=1)
        f1 = self.f1(preds, batch["label"])
        acc = self.acc(logits, batch["label"])

        self.log("test_loss", loss)
        self.log("test_f1", f1)
        self.log("test_acc", acc)
        self.test_cm.update(preds, batch["label"])

    def on_test_epoch_end(self):
        cm = self.test_cm.compute()
        cm_result = cm.cpu().numpy()
        return cm_result

    def predict_step(self, batch) -> dict[str, torch.Tensor]:
        logits = self(batch["input"])
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)
        return {"preds": preds, "probs": probs}

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.cfg.optimizer, params=self.parameters())

        if "lr_scheduler" in self.cfg:
            self.cfg.lr_scheduler.total_steps = self.trainer.estimated_stepping_batches
            scheduler = hydra.utils.instantiate(self.cfg.lr_scheduler, optimizer=optimizer)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        return optimizer


if __name__ == "__main__":
    from hydra import compose, initialize

    from models.cnn import SimpleCNN

    with initialize(config_path="../configs", version_base=None):
        cfg: DictConfig = compose(
            config_name="default", overrides=["hydra.output_subdir=null"]
        )  # 不生成子目录，直接在当前目录运行

    model = SimpleCNN(num_classes=10)
    train_module = TrainModule(model=model, num_classes=10, cfg=cfg.train_module)

    print(train_module)
