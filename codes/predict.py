import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import set_seed, split_dataset, KvasirDataset, calculate_metrics
from unet_baseline import UNetBaseline

# ====================== 自动路径定位（和其他文件保持一致）======================
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CODE_DIR)

# 配置参数
SEED = 42
IMG_SIZE = 256
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 路径配置
DATA_ROOT = os.path.join(PROJECT_ROOT, "dataset", "Kvasir-SEG")
WEIGHT_PATH = os.path.join(PROJECT_ROOT, "weights", "best_unet_baseline.pth")
SAVE_DIR = os.path.join(PROJECT_ROOT, "results", "prediction_samples")
METRICS_SAVE_PATH = os.path.join(PROJECT_ROOT, "results", "test_metrics_baseline.txt")

# ====================== 加载数据与模型 ======================
set_seed(SEED)
# 获取测试集划分
_, _, test_list = split_dataset()
# 构建测试集加载器
test_dataset = KvasirDataset(data_root=DATA_ROOT, name_list=test_list, img_size=IMG_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 加载基线模型与最优权重
model = UNetBaseline(n_channels=3, n_classes=1).to(DEVICE)
model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE, weights_only=True))
model.eval()
print(f"✅ 模型加载成功，权重路径: {WEIGHT_PATH}")
print(f"📌 使用设备: {DEVICE}")

# ====================== 主程序：测试集指标计算 + 可视化 ======================
if __name__ == "__main__":
    # 创建保存目录
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ---------- 1. 计算测试集整体指标 ----------
    total_dice = 0.0
    total_iou = 0.0
    total_prec = 0.0
    total_rec = 0.0
    total_samples = len(test_dataset)

    print("\n🔍 正在计算测试集指标...")
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            outputs = model(imgs)
            dice, iou, prec, rec = calculate_metrics(outputs, masks)
            # 按batch大小累加
            batch_size = imgs.size(0)
            total_dice += dice * batch_size
            total_iou += iou * batch_size
            total_prec += prec * batch_size
            total_rec += rec * batch_size

    # 计算平均值
    avg_dice = total_dice / total_samples
    avg_iou = total_iou / total_samples
    avg_prec = total_prec / total_samples
    avg_rec = total_rec / total_samples

    # 格式化输出并保存指标
    result_text = (
        "========== 基线模型（U-Net + BCE）测试集指标 ==========\n"
        f"测试样本总数: {total_samples}\n"
        f"Dice 系数:    {avg_dice:.4f}\n"
        f"IoU:          {avg_iou:.4f}\n"
        f"Precision:    {avg_prec:.4f}\n"
        f"Recall:       {avg_rec:.4f}\n"
        "======================================================"
    )
    print("\n" + result_text)

    with open(METRICS_SAVE_PATH, "w", encoding="utf-8") as f:
        f.write(result_text)
    print(f"\n📄 指标结果已保存: {METRICS_SAVE_PATH}")

    # ---------- 2. 生成可视化对比图（原图 + 真值 + 预测） ----------
    print("\n🎨 正在生成预测可视化对比图...")
    # 随机抽取8张测试样本
    sample_indices = np.random.choice(total_samples, size=8, replace=False)

    fig, axes = plt.subplots(8, 3, figsize=(9, 20))
    for i, idx in enumerate(sample_indices):
        img_tensor, mask_tensor = test_dataset[idx]
        img_input = img_tensor.unsqueeze(0).to(DEVICE)

        # 模型预测
        with torch.no_grad():
            pred_logits = model(img_input)
            pred_mask = (torch.sigmoid(pred_logits) > 0.5).float().squeeze().cpu().numpy()

        # 反归一化，还原原图用于显示
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)

        mask_np = mask_tensor.squeeze().cpu().numpy()

        # 第一列：原图
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title("Original Image", fontsize=10)
        axes[i, 0].axis("off")

        # 第二列：真值掩码
        axes[i, 1].imshow(mask_np, cmap="gray")
        axes[i, 1].set_title("Ground Truth", fontsize=10)
        axes[i, 1].axis("off")

        # 第三列：预测结果
        axes[i, 2].imshow(pred_mask, cmap="gray")
        axes[i, 2].set_title("Prediction", fontsize=10)
        axes[i, 2].axis("off")

    plt.tight_layout()
    sample_save_path = os.path.join(SAVE_DIR, "baseline_prediction_samples.png")
    plt.savefig(sample_save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"🖼️  预测对比图已保存: {sample_save_path}")
    print("\n🎉 预测任务全部完成！")