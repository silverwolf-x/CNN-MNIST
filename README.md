

## 概览

本项目基于**PyTorch Lightning** ，实现了 **MNIST 手写数字分类任务**，包含以下内容：  
* 使用 **PyTorch Lightning** 模块化构建训练与推理流程，简洁清晰、易于维护；
* 借助 [**Hydra**](https://hydra.cc/) 管理配置项与训练输出，支持动态参数覆盖与多实验管理；
* 在默认配置下训练 CNN 模型，验证集 F1 分数：0.9874，测试集 F1 分数：0.9883




## 特性

* **统一日志输出管理**
  `train.py` 使用 **PyTorch Lightning** 与 **Hydra** 集成，将所有日志统一记录至 Hydra 默认日志文件 `train.log`，并通过 `rank_zero_info` 输出关键日志。训练完成后自动打印 **测试集指标** 和 **混淆矩阵**。

* **模块结构解耦**
  `src/train_module.py` 中定义了自定义的 `LightningModule`，实现训练、验证与测试逻辑的解耦。使用回调 `ModelSummary(max_depth=2)` 自动输出模型各子模块参数结构，方便调试与结构检查。

* **标准化数据传输**
  `src/data_module.py` 中的 `MNISTDataModule` 将数据加载流程封装统一，**使用字典**传递数据，便于扩展与集成。

* **便捷推理接口**
  通过 `predict.py` 并指定 `--log_id`，即可自动加载对应日志目录下保存的模型权重和配置文件，实现一致性推理，**无需手动指定路径或参数**。




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

## 0. 安装依赖

```bash
git clone ...
conda create -n mnist python=3.13
conda activate mnist
pip install -r requirements.txt
```

## 1. 训练模型（Train）

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

* `model._target_`：指定所使用的模型类路径，通常对应于 `models` 模块中的某个模型结构，默认使用`src.models.cnn.SimpleCNN`。
* `data.data_dir`：MNIST 数据集的下载与存储路径，用于数据加载。
* `train_module.optimizer`：优化器配置，默认使用 `AdamW`。
* `train_module.lr_scheduler`：学习率调度器配置，默认使用 `OneCycleLR`。
* `trainer.max_epochs`：训练的最大轮数，默认值为 `10`。
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



## 2. 模型推理（Inference）

找到对应的运行日志 ID（如：`20250517_2350`），该 ID 对应的是保存在 `logs/` 目录下的某次训练输出。运行以下命令，即可自动加载该目录下的模型（`.ckpt`）和配置文件，执行推理：

```bash
python predict.py --log_id=20250517_2350
```

结果在终端中输出 `test_data` 数据集的预测标签。

