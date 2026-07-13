import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from attention_unet import AttentionUNet
from dataset import KvasirDataset, calculate_metrics, set_seed, split_dataset


SEED = 42
IMG_SIZE = 256
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CODE_DIR)
DATA_ROOT = os.path.join(PROJECT_ROOT, "dataset", "Kvasir-SEG")
WEIGHT_PATH = os.path.join(PROJECT_ROOT, "weights", "best_attention_unet_bce.pth")
SAVE_DIR = os.path.join(PROJECT_ROOT, "results", "prediction_samples_attention_bce")
METRICS_SAVE_PATH = os.path.join(PROJECT_ROOT, "results", "test_metrics_attention_bce.txt")
SAMPLE_SAVE_PATH = os.path.join(SAVE_DIR, "attention_prediction_samples_bce.png")


def evaluate(model, loader, device):
    total_dice = total_iou = total_precision = total_recall = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            dice, iou, precision, recall = calculate_metrics(logits, masks)
            batch_size = images.size(0)
            total_dice += dice * batch_size
            total_iou += iou * batch_size
            total_precision += precision * batch_size
            total_recall += recall * batch_size

    sample_count = len(loader.dataset)
    return (
        total_dice / sample_count,
        total_iou / sample_count,
        total_precision / sample_count,
        total_recall / sample_count,
    )


def denormalize(image_tensor):
    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    standard_deviation = np.array([0.229, 0.224, 0.225])
    return np.clip(image * standard_deviation + mean, 0, 1)


if __name__ == "__main__":
    set_seed(SEED)
    _, _, test_list = split_dataset(DATA_ROOT)
    test_dataset = KvasirDataset(DATA_ROOT, test_list, IMG_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = AttentionUNet(n_channels=3, n_classes=1).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    dice, iou, precision, recall = evaluate(model, test_loader, DEVICE)
    result_text = (
        "========== Attention U-Net + BCE 测试集指标 ==========\n"
        f"测试样本数: {len(test_dataset)}\n"
        f"Dice:      {dice:.4f}\n"
        f"IoU:       {iou:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall:    {recall:.4f}\n"
        "========================================================="
    )
    print(result_text)

    with open(METRICS_SAVE_PATH, "w", encoding="utf-8") as file:
        file.write(result_text)

    os.makedirs(SAVE_DIR, exist_ok=True)
    sample_count = min(8, len(test_dataset))
    sample_indices = np.random.choice(len(test_dataset), size=sample_count, replace=False)
    figure, axes = plt.subplots(sample_count, 3, figsize=(9, 2.5 * sample_count), squeeze=False)

    for row, index in enumerate(sample_indices):
        image_tensor, mask_tensor = test_dataset[index]
        with torch.no_grad():
            logits = model(image_tensor.unsqueeze(0).to(DEVICE))
            prediction = (torch.sigmoid(logits) > 0.5).float().squeeze().cpu().numpy()

        axes[row, 0].imshow(denormalize(image_tensor))
        axes[row, 0].set_title("Original Image", fontsize=10)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(mask_tensor.squeeze().cpu().numpy(), cmap="gray")
        axes[row, 1].set_title("Ground Truth", fontsize=10)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(prediction, cmap="gray")
        axes[row, 2].set_title("Prediction", fontsize=10)
        axes[row, 2].axis("off")

    figure.tight_layout()
    figure.savefig(SAMPLE_SAVE_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print(f"测试指标已保存至: {METRICS_SAVE_PATH}")
    print(f"预测示例图已保存至: {SAMPLE_SAVE_PATH}")
