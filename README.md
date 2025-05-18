


# MNIST 手写数字分类任务

本项目基于**PyTorch Lightning** ，实现了 **MNIST 手写数字分类任务**，包含以下内容：  
* 使用 **PyTorch Lightning** 模块化构建训练与推理流程，简洁清晰、易于维护；
* 借助 [**Hydra**](https://hydra.cc/) 管理配置项与训练输出，支持动态参数覆盖与多实验管理；


| 模型 | 配置文件 | 训练轮次 | 验证集 F1 | 测试集 F1 | 参数量 |
| --------------- | ------------ | ---- | ------ | ------ | ----- |
| CNN             | default.yaml | 10   | 0.9872 | 0.9891 | 1.2M  |
| ResNet18  | resnet.yaml  | 5    | 0.9938 | 0.9933 | 11.2M |


自动寻找 batch_size 和学习率的功能尚未实现。可以参考 [PyTorch Lightning 的训练技巧](https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html) 以手动调整这些参数。
(https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html)




## 仓库结构

```
.
├── configs/                     # 配置文件目录
│   └── default.yaml             
├── models/                      # 模型结构定义
│   ├── cnn.py                   
│   └── ...                      
├── src/                         
│   ├── data_module.py           # 定义 MNISTDataModule（数据加载与预处理）
│   └── train_module.py          # 定义 LightningModule（模型训练）
├── train.py                     # 训练脚本（运行入口）
├── predict.py                   # 推理脚本（模型预测）
├── requirements.txt             
└── README.md                    # 说明文档（本文件）
```
## 快速开始
### 0. 安装依赖

```bash
git clone ...
conda create -n mnist python=3.13
conda activate mnist
pip install -r requirements.txt
```

### 1. 训练模型（Train）

默认配置文件路径为 `configs/default.yaml`。

本项目使用 **Hydra** 包进行配置管理，支持模块化组织配置项，并可通过命令行以“点号路径”的形式动态覆盖默认配置值，例如：

```bash
# 使用默认配置训练模型：
python train.py

# 更换配置训练模型(hypra默认接口)：
python train.py --config-path configs --config-name default 

# 指定训练轮数等配置：
python train.py trainer.max_epochs=2
```



### 主要参数

* `model._target_`：指定所使用的模型类路径，通常对应于 `models` 模块中的某个模型结构。
* `data.data_dir`：MNIST 数据集的下载与存储路径，用于数据加载。
* `train_module.optimizer`：优化器配置，默认使用 `AdamW`。
* `train_module.lr_scheduler`：学习率调度器配置，默认使用 `OneCycleLR`。
* `trainer.max_epochs`：训练的最大轮数。
* `callbacks.early_stopping.patience`：早停策略中的容忍轮数，即在验证指标无提升的情况下允许继续训练的最大轮数。
* `data.batch_size`：每个训练批次的样本数（批次大小）。
* `hydra.run.dir`：Hydra 管理的运行目录，包含训练日志、模型权重文件以及配置（yaml）存档，默认时间戳自动生成目录`./logs/${now:%Y%m%d_%H%M}`。


### 训练输出

训练完成后，将在指定的 **`hydra.run.dir`** 目录下生成以下内容：

* **配置文件**：包括 `config.yaml`、`hparams.yaml` 和 `overrides.yaml`，用于记录当前运行的完整配置快照，由 **Hydra** 自动保存。

* **日志文件**
  * `metrics.csv`：由 **PyTorch Lightning** 记录的训练与验证指标；
  * `train.log`：训练过程中的详细日志信息，由 **Hydra** 管理。最终输出test_data的混淆矩阵和测试集指标

* **模型文件**：`*.ckpt`，表示训练过程中保存的模型权重，默认保存性能最优模型，由 **PyTorch Lightning** 管理。



### 2. 模型推理（Inference）

找到对应的运行日志 ID（如：`20250517_2350`），该 ID 对应的是保存在 `logs/` 目录下的某次训练输出。运行以下命令，即可自动加载该目录下的模型（`.ckpt`）和配置文件，执行推理：

```bash
python predict.py --log_id=20250517_2350
```

结果在终端中输出 `test_data` 数据集的预测标签。


## **实现细节**

1. **Hydra 配置层**

* **中心化管理**：在 `configs/` 目录下统一维护训练超参、模型结构与回调配置。
* **路径自动派生**：运行时根据 `log_id` 自动创建 `logs/${log_id}` 目录，用于存储所有实验产出。
* **日志整合**：通过 `rank_zero_info` 将关键信息写入 Lightning 日志，再由 Hydra 统一输出至 `train.log`。

2. **DataModule**

* 使用字典（`dict`）格式在训练、验证和测试阶段传递数据，便于后续扩展与接口统一。
* 封装 `train_dataloader()`、`val_dataloader()`、`test_dataloader()`，自动响应 `batch_size`、`shuffle` 等超参变更。

3. **Trainer 配置**

* **ModelCheckpoint**：根据验证指标自动保存最优模型权重。
* **EarlyStopping**：监控验证指标，遇到性能停滞时自动提前终止训练。
* **Stochastic Weight Averaging**：启用 [SWA](https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/)，在训练后期以较低学习率平均模型权重，提升泛化能力。
* **ModelSummary**：使用 `ModelSummary(max_depth=2)` 回调自动打印模型各子模块层次结构。
* **自动混合精度**：通过 `precision="16-mixed"` 启用 AMP，加速训练并显著节省显存。

4. **模型服务化**

* **热加载权重**：只需调用 `TrainModule.load_from_checkpoint()` 即可在推理脚本中快速恢复模型（前提是 `TrainModule` 类定义不变）。
* **一致性推理**：提供 `predict.py --log_id XXX` 接口，自动定位配置与权重，实现与训练完全一致的推理环境。


