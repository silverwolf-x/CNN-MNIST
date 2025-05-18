import logging
import os
import warnings

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary, StochasticWeightAveraging
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.profilers import SimpleProfiler
from pytorch_lightning.utilities import rank_zero_info

from src.data_module import DataModule
from src.train_module import TrainModule

warnings.filterwarnings("ignore")

torch.set_float32_matmul_precision("medium")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


@hydra.main(config_path="configs", config_name="resnet", version_base=None)
def main(cfg: DictConfig):
    # 0. 设置日志
    # 自动添加到hydra的logger中,具体样式通过hydra的配置文件进行配置
    logging.getLogger("pytorch_lightning").propagate = True
    logging.getLogger("lightning_fabric").propagate = True

    # 1. 数据模块
    seed_everything(cfg.seed, workers=True)
    datamodule = DataModule(**cfg.data)

    # 2. 模型
    model = hydra.utils.instantiate(cfg.model)
    train_module = TrainModule(model=model, num_classes=10, cfg=cfg.train_module)

    # 3. Trainer（主入口）
    callbacks = [
        ModelCheckpoint(
            **cfg.callbacks.model_checkpoint,
            filename=f"{model.__class__.__name__}-epoch={{epoch:02d}}-val_f1={{val_f1:.4f}}",
            auto_insert_metric_name=False,
        ),
        EarlyStopping(**cfg.callbacks.early_stopping),
        ModelSummary(max_depth=2),
        StochasticWeightAveraging(swa_lrs=1e-2),
    ]
    trainer = Trainer(
        **cfg.trainer, callbacks=callbacks, logger=CSVLogger(cfg.trainer.default_root_dir, name=None, version="")
    )

    # 4. 开始训练
    trainer.fit(train_module, datamodule=datamodule)

    # 5. 测试集性能
    test_metric = trainer.test(train_module, datamodule.test_dataloader())
    cm = train_module.on_test_epoch_end()
    rank_zero_info(f"\nConfusion matrix: \n{cm}")
    D = {k: round(v, 4) for k, v in test_metric[0].items()}
    rank_zero_info(f"\nTest metric: {D}")

    # 6. 预测
    # predictions = trainer.predict(train_module, datamodule.test_dataloader())


if __name__ == "__main__":
    main()
