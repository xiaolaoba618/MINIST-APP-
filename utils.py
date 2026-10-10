
import random
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# %%设置人工种子以便复现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

#判断设备类型
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("PyTorch:", torch.__version__)
print("Device:", DEVICE)

# %%读取数据板块

BATCH_SIZE = 128#设定单次参数更新选择128张图片训练
MEAN, STD = (0.1307,), (0.3081,)#手动设定灰度信息均值和标准差，便于Z-Score标准化

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(MEAN, STD),])#标准化灰度信息

full_train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)#读取图片数据
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)#读取测试图片数据

train_size = 55_000#划分训练集与测试集
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size],generator=torch.Generator().manual_seed(SEED))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)#训练时随机打乱顺序，以免学到无用的顺序信息
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)#验证集与测试集无需打乱

images, labels = next(iter(train_loader))#划分单次训练的图片数据与标签，图片数据为[128,1,28,28]张量，代表128张照片，1个通道，28*28像素。标签为一维128向量

#使用Warm-up+余弦退火学习率，增加训练精度和速率
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs,
                 max_lr, min_lr=1e-5):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.max_lr = max_lr
        self.min_lr = min_lr

    def step(self, epoch):
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            lr = self.max_lr * (epoch + 1) / self.warmup_epochs
        else:
            cosine_epochs = max(1, self.total_epochs - self.warmup_epochs)
            t = min(epoch - self.warmup_epochs, cosine_epochs) / cosine_epochs
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                1 + math.cos(math.pi * t)
            )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

# %%定义一个训练epoch
def run_epoch(model, loader, criterion, optimizer=None, device=DEVICE):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                loss.backward()
                optimizer.step()

        n = y.size(0)
        total_loss += loss.item() * n
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += n

    return total_loss / total_samples, total_correct / total_samples


