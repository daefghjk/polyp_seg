import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import KvasirDataset, calculate_metrics, set_seed, split_dataset
from unet_baseline import UNetBaseline


SEED = 42
IMG_SIZE = 256
BATCH_SIZE = 8
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CODE_DIR)
DATA_ROOT = os.path.join(PROJECT_ROOT, "dataset", "Kvasir-SEG")


def parse_arguments():
    parser = argparse.ArgumentParser(description="使用 U-Net + BCE + Dice 模型进行预测与测试。")
    parser.add_argument("--weight_path", required=True, help="要加载的混合损失模型权重路径。")
    parser.add_argument("--output_dir", required=True, help="测试指标和预测图的输出目录。")
    return parser.parse_args()


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


def save_prediction_samples(model, dataset, device, save_path):
    sample_count = min(8, len(dataset))
    sample_indices = np.random.choice(len(dataset), size=sample_count, replace=False)
    figure, axes = plt.subplots(sample_count, 3, figsize=(9, 2.5 * sample_count), squeeze=False)

    for row, index in enumerate(sample_indices):
        image_tensor, mask_tensor = dataset[index]
        with torch.no_grad():
            logits = model(image_tensor.unsqueeze(0).to(device))
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
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main():
    arguments = parse_arguments()
    weight_path = os.path.abspath(arguments.weight_path)
    output_dir = os.path.abspath(arguments.output_dir)
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"找不到模型权重: {weight_path}")

    os.makedirs(output_dir, exist_ok=True)
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_list = split_dataset(DATA_ROOT, save_path=os.path.join(output_dir, "data_split.txt"))
    test_dataset = KvasirDataset(DATA_ROOT, test_list, IMG_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = UNetBaseline(n_channels=3, n_classes=1).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval()

    dice, iou, precision, recall = evaluate(model, test_loader, device)
    result_text = (
        "========== B 组模型（U-Net + BCE + Dice）测试集指标 ==========\n"
        f"测试样本总数: {len(test_dataset)}\n"
        f"Dice 系数:    {dice:.4f}\n"
        f"IoU:          {iou:.4f}\n"
        f"Precision:    {precision:.4f}\n"
        f"Recall:       {recall:.4f}\n"
        "================================================================"
    )
    print(result_text)

    metrics_path = os.path.join(output_dir, "test_metrics_hybrid.txt")
    with open(metrics_path, "w", encoding="utf-8") as file:
        file.write(result_text)

    prediction_dir = os.path.join(output_dir, "prediction_samples_hybrid")
    os.makedirs(prediction_dir, exist_ok=True)
    sample_path = os.path.join(prediction_dir, "hybrid_prediction_samples.png")
    save_prediction_samples(model, test_dataset, device, sample_path)

    print(f"测试指标已保存至: {metrics_path}")
    print(f"预测示例图已保存至: {sample_path}")


if __name__ == "__main__":
    main()
