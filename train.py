from model import MNIST_CNN
import torch.nn as nn
import torch
import copy
from utils import train_loader,val_loader,WarmupCosineScheduler,DEVICE,run_epoch
MAX_EPOCHS = 10
WARMUP_EPOCHS = 1
PATIENCE = 3
MAX_LR = 1e-3
MIN_LR = 1e-5
WEIGHT_DECAY = 1e-4

model = MNIST_CNN(dropout_p=0.25).to(DEVICE) #CNN模型
criterion = nn.CrossEntropyLoss()#定义交叉熵损失函数
optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)#选择AdamW优化器，定义权重衰减抑制过拟合
scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=WARMUP_EPOCHS,total_epochs=MAX_EPOCHS, max_lr=MAX_LR, min_lr=MIN_LR)#选择热身-余弦退火学习率

history = {
    "train_loss": [], "val_loss": [],
    "train_acc": [], "val_acc": [], "lr": []
}
best_val_loss = float("inf")#初始化最小loss
best_state = None#初始化最佳训练结果
epochs_without_improvement = 0#初始化记录训练结果

# %%开始训练
for epoch in range(MAX_EPOCHS):
    lr_now = scheduler.step(epoch)#学习率
    train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer=optimizer)#训练参数，同时记录训练时loss与准确率指标
    val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)#记录验证集指标

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["lr"].append(lr_now)

    if val_loss < best_val_loss:#设置early-stopping部分
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    print(
        f"Epoch {epoch+1:02d}/{MAX_EPOCHS} | lr={lr_now:.2e} | "
        f"train loss={train_loss:.4f}, acc={train_acc:.4f} | "
        f"val loss={val_loss:.4f}, acc={val_acc:.4f}"
    )

    if epochs_without_improvement >= PATIENCE:#三个epoch便跳出训练
        print("Early stopping: 验证损失连续多轮未改善。")
        break

if best_state is not None:
    model.load_state_dict(best_state)
    print(f"已恢复验证损失最低的模型，best val loss={best_val_loss:.4f}")

    checkpoint = {
        "model_state_dict": best_state,
        "best_val_loss": best_val_loss,
        "dropout_p": 0.25,        # 记录结构超参，加载时保持一致
        "epoch": epoch + 1,
        "max_epochs": MAX_EPOCHS,
    }
    torch.save(checkpoint, "best_mnist_cnn.pth")
    print("已保存到 best_mnist_cnn.pth")




