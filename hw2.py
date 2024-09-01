# !/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
@ author: 何雨轩
@ title: homework 2: MINST
@ description:手写数字识别--分类问题
MNIST 数据集来自美国国家标准与技术研究所National Institute of Standards and Technology (NIST).
# 60000张训练集，10000张测试集，
# 1*28*28图片

@ note:
H_out = H_in - Kernel_size + 1
H_out = H_in \ Pooling_size
使用卷积神经网络CNN，分类交叉熵Cross-Entropy做损失函数，Adam做优化
器做图片的分类，net的output是分类的数目，先用classes储存所有类别，之后用torch.max(pred,dim=1)[1]返回分类可能性最大的那个类的index

@ v0.1: 2023-04-15
自带dataset的函数：len(s)查看数据量,s.data查看x数据,s.targets查看y
回顾TensorDataset(x,y):把tensor打包为dataset
torch.FloatTensor(numpy)，把numpy转化为小数点形式的tensor

@ v0.2
UserWarning: (Triggered internally at  ..\torch\csrc\utils\tensor_numpy.cpp:180.)
  return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s
点击跳转到该py文件，改为copy=Ture就没有这个warning了

@ v1.0 2023-04-18
修改了因为自定义split函数导致traindata拆分不均，导致准确率最高0.7的错误。现在准确率能达到0.98，回归正常水平

@ v1.1 2023-04-19
增加了绘制错误分辨图形的可视化展示
"""

import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import time
from sklearn.metrics import confusion_matrix
import logging

# 防止torch包与Anaconda环境中的同一个文件出现了冲突，画不出图
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ===========
# from model import CNN as MyModel
from utils import same_seed, cm_plot, incorrect_plot, loss_plot
from train import trainer, predict
from set_config import config, save_model, MyModel, save_file

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:[%(asctime)s][%(filename)s]: %(message)s',
    datefmt='%m%d_%H:%M',
    handlers=[
        logging.FileHandler(save_file("training.log")),
        logging.StreamHandler(),  # 既输出log,也输出终端
    ],
)

if __name__ == "__main__":
    start_time = time.time()  # 获取当前时间
    start_time = time.time()  # 获取当前时间

    same_seed(config.seed)
    logging.info(f"{torch.__version__=}\n{config.device=}")

    # ===data processing===将原数据<class 'PIL.Image.Image'>转成tensor，并作标准化处理
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(
        root="./", train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        root="./", train=False, download=True, transform=transform
    )

    n_valid = int(len(train_data) * config.valid_ratio)
    n_train = len(train_data) - n_valid
    train_dataset, valid_dataset = random_split(
        train_data, [n_train, n_valid], torch.Generator().manual_seed(config.seed)
    )

    # ======data processing end==
    train_loader, valid_loader = map(
        lambda data: DataLoader(
            data,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=0,
            drop_last=True,
        ),
        [train_dataset, valid_dataset],
    )

    # ===training===
    model = MyModel().to(config.device)
    # logging.info(model)
    train_loss, valid_loss, best_loss = trainer(train_loader, valid_loader, model)

    # ===predict===
    model = MyModel().to(config.device)
    model.load_state_dict(
        torch.load(save_model(best_loss), map_location=config.device, weights_only=True)
    )
    # 使用之前的model迁移学习
    # model.load_state_dict(torch.load(r'.\run\2023-04-18_22.38_epoch1000_score0.989000_model.ckpt',map_location=config.device),strict=False)
    preds, accuracy, incorrect_index = predict(test_data, model)
    logging.info(f"test accuracy:{accuracy:.4f}")
    os.rename(save_model(best_loss), save_model(best_loss, accuracy))

    # ===confusion_matrix===
    cm = confusion_matrix(
        test_data.targets.numpy(), preds, labels=[i for i in range(10)]
    )
    end_time = time.time()

    # ===plot loss===
    loss_plot(train_loss, valid_loss)
    cm_plot(cm, accuracy)
    logging.info(cm)

    # ===incorrect comparasion===
    incorrect_plot(test_data, preds, incorrect_index)
    logging.info("===FINISH!===")
    logging.info(f"Total time: {end_time - start_time:.2f} seconds")
