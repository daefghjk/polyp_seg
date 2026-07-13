import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import set_seed, split_dataset, KvasirDataset, calculate_metrics
from unet_baseline import UNetBaseline

# ====================== 超参数设置 ======================
SEED = 42
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-4
PATIENCE = 10  # 早停耐心值

# 路径配置
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CODE_DIR)
DATA_ROOT = os.path.join(PROJECT_ROOT, "dataset", "Kvasir-SEG")
WEIGHT_SAVE_DIR = os.path.join(PROJECT_ROOT, "weights")
RESULT_SAVE_DIR = os.path.join(PROJECT_ROOT, "results")

# 设备选择
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# ====================== 初始化 ======================
set_seed(SEED)
os.makedirs(WEIGHT_SAVE_DIR, exist_ok=True)
os.makedirs(RESULT_SAVE_DIR, exist_ok=True)

# 数据集划分与加载
train_list, val_list, _ = split_dataset(DATA_ROOT)
train_dataset = KvasirDataset(DATA_ROOT, train_list, IMG_SIZE)
val_dataset = KvasirDataset(DATA_ROOT, val_list, IMG_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 模型、损失、优化器
model = UNetBaseline(n_channels=3, n_classes=1).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()  # A组基线：仅BCE损失
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ====================== 训练函数 ======================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for imgs, masks in tqdm(loader, desc="训练中", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

# ====================== 验证函数 ======================
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_prec = 0.0
    total_rec = 0.0

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="验证中", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            total_loss += loss.item() * imgs.size(0)

            dice, iou, prec, rec = calculate_metrics(outputs, masks)
            total_dice += dice * imgs.size(0)
            total_iou += iou * imgs.size(0)
            total_prec += prec * imgs.size(0)
            total_rec += rec * imgs.size(0)

    n = len(loader.dataset)
    return total_loss/n, total_dice/n, total_iou/n, total_prec/n, total_rec/n

# ====================== 主训练循环 ======================
if __name__ == "__main__":
    best_dice = 0.0
    early_stop_counter = 0
    log_history = []

    print("="*50)
    print("开始A组基线实验：U-Net + BCE Loss")
    print("="*50)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_dice, val_iou, val_prec, val_rec = validate(model, val_loader, criterion, DEVICE)

        # 记录日志
        log_history.append([epoch, train_loss, val_loss, val_dice, val_iou, val_prec, val_rec])
        print(f"Epoch {epoch:03d}/{EPOCHS} | "
              f"训练Loss: {train_loss:.4f} | "
              f"验证Loss: {val_loss:.4f} | "
              f"Dice: {val_dice:.4f} | IoU: {val_iou:.4f}")

        # 保存最优模型
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(WEIGHT_SAVE_DIR, "best_unet_baseline.pth"))
            early_stop_counter = 0
            print(f"--> 最优模型已更新，最佳Dice: {best_dice:.4f}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= PATIENCE:
                print(f"早停触发，连续{PATIENCE}轮未提升，训练结束")
                break

    # ====================== 保存训练日志 ======================
    csv_path = os.path.join(RESULT_SAVE_DIR, "train_log_baseline.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "val_precision", "val_recall"])
        writer.writerows(log_history)
    print(f"\n训练日志已保存: {csv_path}")

    # ====================== 绘制训练曲线 ======================
    epochs_list = [row[0] for row in log_history]
    train_loss_list = [row[1] for row in log_history]
    val_loss_list = [row[2] for row in log_history]
    val_dice_list = [row[3] for row in log_history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(epochs_list, train_loss_list, label="Train Loss")
    axes[0].plot(epochs_list, val_loss_list, label="Val Loss")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs_list, val_dice_list, label="Val Dice", color="orange")
    axes[1].set_title("Validation Dice Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Coefficient")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    curve_path = os.path.join(RESULT_SAVE_DIR, "train_curve_baseline.png")
    plt.savefig(curve_path, dpi=200)
    print(f"训练曲线已保存: {curve_path}")
    print(f"\n基线实验完成，最佳验证Dice: {best_dice:.4f}")