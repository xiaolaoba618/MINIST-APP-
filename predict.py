# predict.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MNIST_CNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "best_mnist_cnn.pth"

# MNIST 官方均值/方差（务必与训练时一致！）
MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)

_model = None  # 缓存，避免重复加载


def load_model(ckpt_path: str = CKPT_PATH, device=DEVICE):
    """加载训练好的模型，返回 eval 模式下的模型。"""
    global _model
    ckpt = torch.load(ckpt_path, map_location=device)

    # 兼容两种保存格式
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        dropout_p = ckpt.get("dropout_p", 0.25)
    else:
        state_dict, dropout_p = ckpt, 0.25

    model = MNIST_CNN(dropout_p=dropout_p).to(device)
    model.load_state_dict(state_dict)
    model.eval()          # 关键：关掉 Dropout / BN 更新
    _model = model
    return model


@torch.no_grad()
def predict(images, model=None, return_prob: bool = False):
    """
    参数
    ----
    images : torch.Tensor
        形状 (N,1,28,28) / (1,28,28) / (28,28)，取值 [0,1] 的 float 张量
    return_prob : bool
        是否同时返回置信度

    返回
    ----
    preds : list[int]            预测类别
    confs : list[float]（可选）   对应置信度
    """
    model = model or _model or load_model()

    if images.dim() == 2:                 # (28,28)
        images = images.unsqueeze(0).unsqueeze(0)
    elif images.dim() == 3:               # (1,28,28) 或 (N,28,28)
        images = images.unsqueeze(0) if images.shape[0] == 1 else images.unsqueeze(1)

    images = images.to(DEVICE).float()
    probs = F.softmax(model(images), dim=-1)
    confs, preds = probs.max(dim=-1)

    preds = preds.cpu().tolist()
    confs = confs.cpu().tolist()
    return (preds, confs) if return_prob else preds


def evaluate_testset(model=None, batch_size: int = 256, root: str = "./data"):
    """在 MNIST 官方测试集（10k 张）上评估准确率。"""
    model = model or _model or load_model()

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])
    test_ds = datasets.MNIST(root=root, train=False, download=True, transform=tf)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = correct / total
    print(f"Test accuracy: {acc:.4f}  ({correct}/{total})")
    return acc


evaluate_testset()


