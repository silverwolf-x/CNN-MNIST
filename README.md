# 深度学习测试demo

手写数字识别--十分类问题

- MNIST 数据集来自美国国家标准与技术研究所National Institute of Standards and Technology (NIST).
- 60000张训练集，10000张测试集，
- 1\*28\*28图片
- H_out = H_in - Kernel_size + 1
- H_out = H_in \ Pooling_size

使用卷积神经网络CNN，分类交叉熵Cross-Entropy做损失函数，Adam做优化器做图片的分类

net的output是分类的数目，先用classes储存所有类别，之后用`torch.max(pred,dim=1)[1]`返回分类可能性最大的那个类的index

## v0.1: 2023-04-15
自带`dataset`的函数：`len(s)`查看数据量,`s.data`查看x数据,`s.targets`查看y

回顾`TensorDataset(x,y)`把tensor打包为dataset
`torch.FloatTensor(numpy)`，把numpy转化为小数点形式的tensor

## v0.2
```python
UserWarning: (Triggered internally at  ..\torch\csrc\utils\tensor_numpy.cpp:180.)
  return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s
```
点击跳转到该py文件，改为copy=Ture就没有这个warning了

## v1.0 2023-04-18
修改了因为自定义split函数导致traindata拆分不均，导致准确率最高0.7的错误。现在准确率能达到0.98，回归正常水平

## v1.1 2023-04-19
增加了绘制错误分辨图形的可视化展示

## v1.2 2023-11-19
上传github存档

## v2.0 alpha 2024-09-01

- 暂时取消学习率，即使研究表明衰减自适应优化器和需要不需要LR scheduler几乎是的没有关系的，他们经常需要同时（叠加）工作。https://www.zhihu.com/question/315772308/answer/1636730368
- numworker = 1 -->0 ，加快整体速度
- train_loop 新设置mininterval=1，并在train_loop.set_postfix设置refresh=False
- batch:256
TODO list:
1. 比较各optim收敛速率
2. 改为轻量代码

TODO: 解耦各板块,config使用yaml输入
util.py
train.py
model.py
其中模型增加Resnet9
[Resnet9](https://github.com/VanekPetr/ResNet-9/blob/main/model.py)

## v2.0 beta
- [hw2.py][logging.StreamHandler()]同时输出log和终端
- 解耦模块
- [set_config.py][save_file()]修复设置时间重命名时的错位
现在MNIST到达一个阶段，准确率达到98.8%左右
之后更换另外一个主题训练

## v2.1 alpha
使用AdamW+余弦lr，灵感来自[LORA训练](https://zhuanlan.zhihu.com/p/618758020)，参考[余弦退火学习率](https://zhuanlan.zhihu.com/p/261134624)，效果一般般

## v2.2

- refactor: 使用dataclass规范config
- feat: 使用resnet18架构（不用预训练权重）
- perf: resnet + CE loss + NAdam + 每个epoch更新一次余弦退火lr调整 = valid loss 3.18e-05 + test acc 0.9920 (目前最高)

> [!note]
> 灵感来源: [余弦退火设置](https://blog.csdn.net/qq_29007291/article/details/126094939)
> [骚操作](https://www.zhihu.com/question/666647497/answer/3627168088)