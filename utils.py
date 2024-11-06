import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.utils import make_grid
import os
import time
from config import config

time_flag = True  # 防止跨分钟训练导致重命名错误
T = None
FOLDER = config.folder


def save_file(path):
    os.makedirs(FOLDER, exist_ok=True)
    global time_flag, T
    if time_flag:
        T = time.localtime()
        time_flag = False
    time_str = time.strftime(r"%Y-%m-%d_%H.%M_", T)
    return os.path.join(FOLDER, time_str + path)


def save_model(loss, accuracy=None):
    if accuracy is None:
        path = f"loss{loss:.4f}_model.ckpt"
    else:
        path = f"accuracy{accuracy:.3f}_model.ckpt"
    return save_file(path)


def same_seed(seed):
    """固定seed保证复现"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cm_plot(cm, accuracy):
    plt.figure()
    sns.heatmap(
        cm, annot=True, fmt="d", linewidths=0.3, cmap=sns.color_palette("Blues")
    )
    plt.xlabel("predict")
    plt.ylabel("true")
    plt.title(f"accuracy{accuracy:}_model's confusion matrix")
    plt.savefig(save_file("confusion matrix.png"))
    plt.show()


def incorrect_plot(test_data, preds, incorrect_index):
    """绘制左右子图，每个图像的位置上绘制相应的标签数字"""
    num_images = len(incorrect_index)
    images = [test_data[i][0] for i in incorrect_index]

    fix_rows = 10  # 列
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(make_grid(images, nrow=fix_rows).permute(1, 2, 0))
    axs[0].set_title("True images")

    white_image = torch.ones_like(images[0], dtype=int).fill_(255)
    axs[1].imshow(make_grid([white_image] * num_images, nrow=fix_rows).permute(1, 2, 0))
    axs[1].set_title("Predicts")
    axs[1].axis("off")
    for i in range(num_images):
        # 每个框线2像素
        axs[1].text(
            i % fix_rows * 30 + 16,
            i // fix_rows * 30 + 16,
            str(preds[i]),
            color="black",
            ha="center",
            va="center",
        )

    plt.suptitle("incorrect comparison")
    plt.savefig(save_file("incorrect comparison.png"))
    plt.show()


def loss_plot(train_loss, valid_loss):
    """画损失图，训练误差和泛化误差"""
    plt.figure()
    plt.plot(train_loss, label="train_loss")
    plt.plot(valid_loss, label="valid_loss")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("training loss")
    plt.legend()
    plt.savefig(save_file("training loss.png"))
    plt.show()
