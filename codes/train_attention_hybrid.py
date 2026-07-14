import argparse
import csv
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from attention_unet import AttentionUNet
from dataset import KvasirDataset, calculate_metrics, set_seed, split_dataset
from experiment_utils import (
    experiment_name,
    format_learning_rate,
    loss_weight_ratio,
    positive_integer,
    positive_learning_rate,
)
from losses import BCEDiceLoss


SEED = 42
IMG_SIZE = 256
EPOCHS = 100
PATIENCE = 10

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CODE_DIR)
DATA_ROOT = os.path.join(PROJECT_ROOT, "dataset", "Kvasir-SEG")
EXPERIMENT_ROOT = os.path.join(PROJECT_ROOT, "experiments")
MODEL_NAME = "attention_unet_hybrid"


def parse_arguments():
    parser = argparse.ArgumentParser(description="训练 Attention U-Net + BCE + Dice 混合损失模型（D 组）。")
    parser.add_argument("--batch_size", required=True, type=positive_integer, help="训练批大小。")
    parser.add_argument("--lr", required=True, type=positive_learning_rate, help="Adam 学习率。")
    parser.add_argument(
        "--loss_weights",
        default="1:1",
        type=loss_weight_ratio,
        help="BCE 与 Dice 损失权重比，例如 1:1、1:2、2:1。",
    )
    return parser.parse_args()


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


def save_training_curve(log_history, curve_path, loss_label):
    epochs = [row[0] for row in log_history]
    train_losses = [row[1] for row in log_history]
    validation_losses = [row[2] for row in log_history]
    validation_dice_scores = [row[3] for row in log_history]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_losses, label="Train Loss")
    axes[0].plot(epochs, validation_losses, label="Val Loss")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel(loss_label)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, validation_dice_scores, label="Val Dice", color="orange")
    axes[1].set_title("Validation Dice Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Coefficient")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(curve_path, dpi=200)
    plt.close(figure)


def main():
    arguments = parse_arguments()
    bce_weight, dice_weight = arguments.loss_weights
    tag = experiment_name(MODEL_NAME, arguments.batch_size, arguments.lr, bce_weight, dice_weight)
    experiment_dir = os.path.join(EXPERIMENT_ROOT, tag)
    os.makedirs(experiment_dir, exist_ok=True)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_list, validation_list, _ = split_dataset(
        DATA_ROOT, save_path=os.path.join(experiment_dir, "data_split.txt")
    )
    train_dataset = KvasirDataset(DATA_ROOT, train_list, IMG_SIZE)
    validation_dataset = KvasirDataset(DATA_ROOT, validation_list, IMG_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=arguments.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(
        validation_dataset, batch_size=arguments.batch_size, shuffle=False, num_workers=0
    )

    model = AttentionUNet(n_channels=3, n_classes=1).to(device)
    criterion = BCEDiceLoss(bce_weight=bce_weight, dice_weight=dice_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(arguments.lr))
    weight_path = os.path.join(experiment_dir, f"best_{tag}.pth")
    log_path = os.path.join(experiment_dir, "train_log_attention_hybrid.csv")
    curve_path = os.path.join(experiment_dir, "train_curve_attention_hybrid.png")
    config_path = os.path.join(experiment_dir, "experiment_config.txt")

    best_dice = 0.0
    early_stop_counter = 0
    log_history = []

    print(f"使用设备: {device}")
    print(
        f"开始 D 组实验：Attention U-Net + BCE + Dice | batch_size={arguments.batch_size}, "
        f"lr={format_learning_rate(arguments.lr)}, {criterion.loss_label()}"
    )
    print(f"实验目录: {experiment_dir}")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        validation_metrics = validate(model, validation_loader, criterion, device)
        validation_loss, validation_dice, validation_iou, validation_precision, validation_recall = validation_metrics
        log_history.append(
            [epoch, train_loss, validation_loss, validation_dice, validation_iou, validation_precision, validation_recall]
        )
        print(
            f"轮次 {epoch:03d}/{EPOCHS} | 训练损失: {train_loss:.4f} | "
            f"验证损失: {validation_loss:.4f} | Dice: {validation_dice:.4f} | IoU: {validation_iou:.4f}"
        )

        if validation_dice > best_dice:
            best_dice = validation_dice
            torch.save(model.state_dict(), weight_path)
            early_stop_counter = 0
            print(f"最优模型已更新，Dice: {best_dice:.4f}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= PATIENCE:
                print(f"早停触发，连续 {PATIENCE} 轮未提升")
                break

    with open(log_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "val_precision", "val_recall"])
        writer.writerows(log_history)
    save_training_curve(log_history, curve_path, criterion.loss_label())

    with open(config_path, "w", encoding="utf-8") as file:
        file.write(f"experiment_group: D\n")
        file.write(f"model: Attention U-Net + BCE + Dice\n")
        file.write(f"batch_size: {arguments.batch_size}\n")
        file.write(f"learning_rate: {format_learning_rate(arguments.lr)}\n")
        file.write(f"loss_weights: {criterion.loss_label()}\n")
        file.write(f"best_val_dice: {best_dice:.6f}\n")

    print(f"训练日志已保存至: {log_path}")
    print(f"训练曲线已保存至: {curve_path}")
    print(f"最佳验证集 Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
