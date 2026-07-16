"""Generate a single-image segmentation comparison figure.

Example:
    python test/demo_single_image.py ^
        --image_path dataset/kvasir-seg/images/example.jpg ^
        --mask_path dataset/kvasir-seg/masks/example.jpg ^
        --model_path experiments/unet_baseline_bs8_lr0.001/best_unet_baseline_bs8_lr0.001.pth
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "codes"
sys.path.insert(0, str(CODE_DIR))

from attention_unet import AttentionUNet
from unet_baseline import UNetBaseline


IMAGE_SIZE = 256
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="对单张图像进行息肉分割，并保存原图、真值和预测结果的对比图。"
    )
    parser.add_argument("--image_path", required=True, help="输入图像路径。")
    parser.add_argument("--mask_path", required=True, help="对应真值 mask 路径。")
    parser.add_argument("--model_path", required=True, help="模型权重 .pth 文件路径。")
    parser.add_argument(
        "--output_path",
        default=None,
        help="对比图输出路径，默认为 test/demo_output/<图片名>_comparison.png。",
    )
    parser.add_argument(
        "--model_type",
        choices=["auto", "baseline", "attention"],
        default="auto",
        help="模型类型；默认依据权重内容自动识别。",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="二值化阈值，默认 0.5。"
    )
    return parser.parse_args()


def validate_path(path, name):
    if not path.is_file():
        raise FileNotFoundError(f"找不到{name}: {path}")


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("模型权重必须是 state_dict 或包含 state_dict 的字典。")

    for key in ("state_dict", "model_state_dict"):
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint


def build_model(state_dict, model_type):
    has_attention_gate = any(".attention." in key for key in state_dict)
    if model_type == "auto":
        model_type = "attention" if has_attention_gate else "baseline"
    elif model_type == "attention" and not has_attention_gate:
        raise ValueError("指定了 attention 模型，但权重中未发现注意力门参数。")
    elif model_type == "baseline" and has_attention_gate:
        raise ValueError("指定了 baseline 模型，但权重包含注意力门参数。")

    if model_type == "attention":
        return AttentionUNet(n_channels=3, n_classes=1), "Attention U-Net"
    return UNetBaseline(n_channels=3, n_classes=1), "U-Net"


def calculate_dice(prediction, target):
    intersection = np.logical_and(prediction, target).sum(dtype=np.float64)
    total = prediction.sum(dtype=np.float64) + target.sum(dtype=np.float64)
    return (2.0 * intersection + 1e-6) / (total + 1e-6)


def load_inputs(image_path, mask_path):
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    original_image = np.asarray(image)

    image_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    mask_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=Image.Resampling.NEAREST),
            transforms.ToTensor(),
        ]
    )
    image_tensor = image_transform(image).unsqueeze(0)
    mask_array = mask_transform(mask).squeeze(0).numpy() > 0.5
    return original_image, image_tensor, mask_array


def save_comparison(original_image, target, prediction, dice, model_name, output_path):
    figure, axes = plt.subplots(1, 3, figsize=(14, 5))
    panels = [
        (original_image, "Input"),
        (target, "Ground Truth Mask"),
        (prediction, "Prediction"),
    ]

    for axis, (content, title) in zip(axes, panels):
        if content.ndim == 2:
            axis.imshow(content, cmap="gray", vmin=0, vmax=1)
        else:
            axis.imshow(content)
        axis.set_title(title)
        axis.axis("off")

    figure.suptitle(f"{model_name} | Dice: {dice:.4f}", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main():
    arguments = parse_arguments()
    if not 0.0 <= arguments.threshold <= 1.0:
        raise ValueError("--threshold 必须在 0 到 1 之间。")

    image_path = Path(arguments.image_path).resolve()
    mask_path = Path(arguments.mask_path).resolve()
    model_path = Path(arguments.model_path).resolve()
    validate_path(image_path, "输入图像")
    validate_path(mask_path, "mask")
    validate_path(model_path, "模型权重")

    output_path = (
        Path(arguments.output_path).resolve()
        if arguments.output_path
        else PROJECT_ROOT / "test" / "demo_output" / f"{image_path.stem}_comparison.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = extract_state_dict(checkpoint)
    model, model_name = build_model(state_dict, arguments.model_type)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    original_image, image_tensor, target = load_inputs(image_path, mask_path)
    with torch.no_grad():
        logits = model(image_tensor.to(device))
        prediction = (torch.sigmoid(logits) >= arguments.threshold).squeeze().cpu().numpy()

    dice = calculate_dice(prediction, target)
    save_comparison(original_image, target, prediction, dice, model_name, output_path)
    print(f"模型类型: {model_name}")
    print(f"Dice: {dice:.4f}")
    print(f"对比图已保存至: {output_path}")


if __name__ == "__main__":
    main()
