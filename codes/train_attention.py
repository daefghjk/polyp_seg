import csv
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from attention_unet import AttentionUNet
from dataset import KvasirDataset, calculate_metrics, set_seed, split_dataset


SEED = 42
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-4
PATIENCE = 10

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CODE_DIR)
DATA_ROOT = os.path.join(PROJECT_ROOT, "dataset", "Kvasir-SEG")
WEIGHT_SAVE_DIR = os.path.join(PROJECT_ROOT, "weights")
RESULT_SAVE_DIR = os.path.join(PROJECT_ROOT, "results")
WEIGHT_NAME = "best_attention_unet_bce.pth"
LOG_NAME = "train_log_attention_bce.csv"
CURVE_NAME = "train_curve_attention_bce.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, masks in tqdm(loader, desc="训练中", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = total_dice = total_iou = total_precision = total_recall = 0.0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="验证中", leave=False):
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            total_loss += loss.item() * images.size(0)

            dice, iou, precision, recall = calculate_metrics(logits, masks)
            batch_size = images.size(0)
            total_dice += dice * batch_size
            total_iou += iou * batch_size
            total_precision += precision * batch_size
            total_recall += recall * batch_size

    sample_count = len(loader.dataset)
    return (
        total_loss / sample_count,
        total_dice / sample_count,
        total_iou / sample_count,
        total_precision / sample_count,
        total_recall / sample_count,
    )


if __name__ == "__main__":
    set_seed(SEED)
    os.makedirs(WEIGHT_SAVE_DIR, exist_ok=True)
    os.makedirs(RESULT_SAVE_DIR, exist_ok=True)

    train_list, validation_list, _ = split_dataset(DATA_ROOT)
    train_dataset = KvasirDataset(DATA_ROOT, train_list, IMG_SIZE)
    validation_dataset = KvasirDataset(DATA_ROOT, validation_list, IMG_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    validation_loader = DataLoader(
        validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = AttentionUNet(n_channels=3, n_classes=1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_dice = 0.0
    early_stop_counter = 0
    log_history = []

    print(f"使用设备: {DEVICE}")
    print("开始 C 组实验：Attention U-Net + BCE 损失")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        validation_metrics = validate(model, validation_loader, criterion, DEVICE)
        validation_loss, validation_dice, validation_iou, validation_precision, validation_recall = validation_metrics

        log_history.append(
            [
                epoch,
                train_loss,
                validation_loss,
                validation_dice,
                validation_iou,
                validation_precision,
                validation_recall,
            ]
        )
        print(
            f"轮次 {epoch:03d}/{EPOCHS} | 训练损失: {train_loss:.4f} | "
            f"验证损失: {validation_loss:.4f} | Dice: {validation_dice:.4f} | "
            f"IoU: {validation_iou:.4f}"
        )

        if validation_dice > best_dice:
            best_dice = validation_dice
            torch.save(model.state_dict(), os.path.join(WEIGHT_SAVE_DIR, WEIGHT_NAME))
            early_stop_counter = 0
            print(f"最优模型已更新，Dice: {best_dice:.4f}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= PATIENCE:
                print(f"早停触发，连续 {PATIENCE} 轮未提升")
                break

    with open(os.path.join(RESULT_SAVE_DIR, LOG_NAME), "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "val_precision", "val_recall"]
        )
        writer.writerows(log_history)

    epochs = [row[0] for row in log_history]
    train_losses = [row[1] for row in log_history]
    validation_losses = [row[2] for row in log_history]
    validation_dice_scores = [row[3] for row in log_history]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_losses, label="Train Loss")
    axes[0].plot(epochs, validation_losses, label="Val Loss")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, validation_dice_scores, label="Val Dice", color="orange")
    axes[1].set_title("Validation Dice Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Coefficient")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    figure.tight_layout()
    curve_path = os.path.join(RESULT_SAVE_DIR, CURVE_NAME)
    figure.savefig(curve_path, dpi=200)
    plt.close(figure)

    print(f"训练日志已保存至: {os.path.join(RESULT_SAVE_DIR, LOG_NAME)}")
    print(f"训练曲线已保存至: {curve_path}")
    print(f"最佳验证集 Dice: {best_dice:.4f}")
