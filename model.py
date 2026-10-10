'''
定义CNN卷积神经网络主模型
'''
import torch
import torch.nn as nn
from utils import DEVICE


class MNIST_CNN(nn.Module):
    def __init__(self, dropout_p=0.25):
        super().__init__()

        #卷积部分
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), #第一层卷积核使用3*3*1尺寸，一共32个，表示提取三十二个维度的特征。padding填充1以便输出特征图尺寸不变。
            nn.BatchNorm2d(32),#BatchNorm2d标准化
            nn.ReLU(),#增加非线性
            nn.MaxPool2d(kernel_size=2),  # 池化部分，使用最大池化，维度变化28 -> 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),##第二层卷积，使用3*3*32尺寸，一共64个.
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 14 -> 7，最终输出特征图为64*7*7尺寸，一共64张特征图，每张像素7*7
        )

        #全连接部分
        self.classifier = nn.Sequential(
            nn.Flatten(),#展平最后特征图
            nn.Linear(64 * 7 * 7, 128),#全链接第一层，使用128维度
            nn.ReLU(),
            nn.Dropout(p=dropout_p),#随机dropout神经元，抑制过拟合
            nn.Linear(128, 10),#全链接输出层，128维映射为10维度分类
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):#只对卷积层和全链接线性层含有参数的部分初始化
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu") #初始化权重参数，使用kaiming初始化方法
                if module.bias is not None:
                    nn.init.zeros_(module.bias)#初始化偏置

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)  # logits变换而不是softmax




