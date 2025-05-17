import argparse
import os
import pdb
import warnings

import torch
from hydra import compose, initialize
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from src.data_module import DataModule
from src.train_module import TrainModule

warnings.filterwarnings("ignore")

torch.set_float32_matmul_precision("medium")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


parser = argparse.ArgumentParser()
parser.add_argument("--log_id", help="日志目录名，例如 20250517_2350", default="20250517_2350")
args = parser.parse_args()

PATH = os.path.join("logs", args.log_id)

if __name__ == "__main__":
    # 1. 载入配置文件
    with initialize(config_path=PATH, version_base=None):
        cfg: DictConfig = compose(
            config_name="config",
            overrides=["hydra.output_subdir=null", "trainer.default_root_dir=.", "+trainer.logger=false"],
        )  # 禁用hydra和trainer的logger

    # 2. 自动查找ckpt文件
    checkpoint_path = next((os.path.join(PATH, f) for f in os.listdir(PATH) if f.endswith(".ckpt")), None)
    if not checkpoint_path:
        raise FileNotFoundError(f"未找到ckpt文件于: {PATH}")
    print(f"加载模型: {os.path.basename(checkpoint_path)}")

    train_module = TrainModule.load_from_checkpoint(checkpoint_path)

    # 3. 数据模块
    datamodule = DataModule(**cfg.data)

    # 4. Trainer
    trainer = Trainer(**cfg.trainer)

    # 5. 预测
    outputs = trainer.predict(train_module, datamodule.test_dataloader())
    all_preds = torch.cat([batch["preds"] for batch in outputs], dim=0).numpy()
    all_probs = torch.cat([batch["probs"] for batch in outputs], dim=0).numpy()

    print(all_preds, all_preds.shape)
    # predictions = torch.cat(predictions).numpy()
