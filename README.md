# MNIST 手写数字识别（CNN · PyTorch）

用卷积神经网络识别手写数字 0–9，配套一个可实时手写输入的可视化界面。

## 项目结构

```
CNN_MINIST/
├── model.py            # CNN 网络结构（MNIST_CNN）
├── train.py            # 训练脚本（Warmup+余弦退火、EarlyStopping）
├── utils.py            # 数据加载、预处理、设备选择、训练单轮逻辑
├── predict.py          # 推理与测试集评估
├── app.py              # tkinter 手写面板（画板 → 28×28 → 预测）
├── best_mnist_cnn.pth  # 训练好的模型权重
└── data/               # MNIST 数据集
```

## 模型结构

两段「卷积 → BN → ReLU → 最大池化」，再经全连接分类：

```
输入 1×28×28
  Conv(1→32)  + BN + ReLU + MaxPool   28→14
  Conv(32→64) + BN + ReLU + MaxPool   14→7
  Flatten  64×7×7=3136
  Linear(3136→128) + ReLU + Dropout(0.25)
  Linear(128→10)   # 输出 logits
```

## 训练策略

- 优化器 AdamW，`lr=1e-3`，`weight_decay=1e-4`
- 学习率：1 轮 Warmup + 余弦退火（`1e-3 → 1e-5`）
- 早停：验证损失连续 3 轮不下降即停止（上限 10 轮）
- 数据：55000 训练 / 5000 验证，按验证损失最低处保存权重

## 关键点

**标准化必须与训练一致。** 训练用了
`transforms.Normalize(mean=0.1307, std=0.3081)`，因此推理前同样要做
`(img - 0.1307) / 0.3081`，否则预测会明显失准。

手写输入还需先对齐 MNIST 规范：裁剪笔画包围盒 → 最长边等比缩放到 20px → 居中放入 28×28。

## 使用

```bash
# 训练（生成 best_mnist_cnn.pth）
python train.py

# 测试集评估
python predict.py

# 手写界面：鼠标写数字，松开即自动识别
python app.py
```

## 结果

| 指标 | 数值 |
| --- | --- |
| 测试集准确率 | **99.16%**（9916 / 10000） |

## 环境

Python 3.13 · PyTorch · torchvision · numpy · Pillow · tkinter（CPU 即可运行）
