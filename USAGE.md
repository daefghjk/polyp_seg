# 工程结构与使用说明

## 1. 项目用途

本项目用于肠镜息肉图像二分类分割，包含两组模型：

- U-Net 基线模型：`U-Net + BCE Loss`。
- Attention U-Net：在四条跳跃连接中加入 Attention Gate，训练损失仍为 `BCEWithLogitsLoss`。

训练和预测均使用 Kvasir-SEG 数据集，输入图像尺寸固定为 `256 x 256`。数据集通过随机种子 `42` 按 `8:1:1` 划分训练集、验证集和测试集。

## 2. 目录结构

```text
polyp_seg/
├── codes/
│   ├── unet_baseline.py        # U-Net 基线模型定义
│   ├── attention_unet.py       # Attention U-Net 与 Attention Gate 定义
│   ├── dataset.py              # 数据集、固定划分与评价指标
│   ├── experiment_utils.py     # 参数校验与实验目录命名
│   ├── train.py                # U-Net 基线训练入口
│   ├── train_attention.py      # Attention U-Net 训练入口
│   ├── predict.py              # U-Net 基线预测与测试入口
│   ├── predict_attention.py    # Attention U-Net 预测与测试入口
│   └── plot_training_log.py    # 由训练日志 CSV 绘制曲线
├── dataset/
│   └── Kvasir-SEG/
│       ├── images/             # 原始图像
│       └── masks/              # 二值掩码
├── experiments/                # 每个模型和超参数组合的完整实验产物
├── README.md                   # 项目讨论提纲
└── USAGE.md                    # 本使用说明
```

## 3. 环境准备

项目依赖 PyTorch、TorchVision、NumPy、Pillow、Matplotlib 和 tqdm。

数据集目录必须为：

```text
dataset/Kvasir-SEG/images
dataset/Kvasir-SEG/masks
```

## 4. 训练

两个训练脚本都必须提供 `--batch_size` 和 `--lr`。缺少任一参数会立即报错。

```powershell
# A 组：U-Net + BCE
python codes\train.py --batch_size 8 --lr 1e-4

# C 组：Attention U-Net + BCE
python codes\train_attention.py --batch_size 8 --lr 1e-4
```

`batch_size` 必须是正整数，`lr` 必须是有限正数。学习率会被规范为小数形式，因此 `1e-4` 和 `0.0001` 指向同一组实验标识。

训练会创建以下格式的目录：

```text
experiments/<模型名>_bs<batch_size>_lr<learning_rate>/
```

例如：

```text
experiments/unet_baseline_bs8_lr0.0001/
experiments/attention_unet_bce_bs8_lr0.0001/
```

每个实验目录均包含：

- 带模型名和超参数的最优权重文件，例如 `best_unet_baseline_bs8_lr0.0001.pth`。
- `data_split.txt`：本实验使用的数据划分。
- `train_log_*.csv`：每轮训练和验证指标。
- `train_curve_*.png`：训练损失和验证 Dice 曲线。

重新训练相同模型和相同超参数时，会覆盖该实验目录中同名的权重、日志和曲线。

## 5. 预测与测试

预测脚本必须传入要加载的权重路径和结果输出目录。建议将输出目录设置为权重所在实验目录，使模型和对应结果保持在同一位置。

```powershell
# 基线模型预测
python codes\predict.py `
  --weight_path experiments\unet_baseline_bs8_lr0.0001\best_unet_baseline_bs8_lr0.0001.pth `
  --output_dir experiments\unet_baseline_bs8_lr0.0001

# Attention U-Net 预测
python codes\predict_attention.py `
  --weight_path experiments\attention_unet_bce_bs8_lr0.0001\best_attention_unet_bce_bs8_lr0.0001.pth `
  --output_dir experiments\attention_unet_bce_bs8_lr0.0001
```

预测会在输出目录写入测试集 Dice、IoU、Precision、Recall，以及原图、真实掩码和预测掩码的对比图。预测时也会写入 `data_split.txt`，以确保测试集与训练使用同一固定划分。

## 6. 由 CSV 重绘训练曲线

`plot_training_log.py` 读取任意符合当前训练日志字段的 CSV，并保存曲线图片。两个参数均为必填：

```powershell
python codes\plot_training_log.py `
  --csv_path experiments\attention_unet_bce_bs8_lr0.0001\train_log_attention_bce.csv `
  --output_path experiments\attention_unet_bce_bs8_lr0.0001\train_curve_from_csv.png
```

CSV 必须包含以下字段：`epoch`、`train_loss`、`val_loss`、`val_dice`。

## 7. 当前已训练实验

当前已有以下实验结果：

- `experiments/unet_baseline_bs8_lr0.0001/`
- `experiments/attention_unet_bce_bs8_lr0.0001/`

两个目录均已包含对应模型权重、训练日志、训练曲线、测试指标和预测示例图。
