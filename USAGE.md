# 工程使用说明

本项目使用 Kvasir-SEG 数据集完成肠镜息肉二分类分割。所有脚本从项目根目录运行，输入尺寸固定为 `256 x 256`，随机种子为 `42`，数据按 `8:1:1` 划分训练、验证和测试集。每次测试均为 100 张测试图像。

## 1. 工程结构

```text
polyp_seg/
├── codes/                         # 模型、训练、测试和结果汇总脚本
│   ├── unet_baseline.py            # U-Net
│   ├── attention_unet.py           # Attention U-Net 和 Attention Gate
│   ├── losses.py                   # BCE + Dice 混合损失
│   ├── dataset.py                  # 数据集、固定划分和评价指标
│   ├── experiment_utils.py         # 参数校验和实验目录命名
│   ├── train.py                    # A 组训练：U-Net + BCE
│   ├── train_hybrid.py             # B 组训练：U-Net + BCE + Dice
│   ├── train_attention.py          # C 组训练：Attention U-Net + BCE
│   ├── train_attention_hybrid.py   # D 组训练：Attention U-Net + BCE + Dice
│   ├── predict*.py                 # 四组模型的测试与预测可视化
│   ├── run_param_experiments.py    # B 组调参和 D 组训练的批量入口
│   ├── build_tuning_tables.py      # 自动生成调参表格和曲线
│   ├── plot_training_log.py        # 单个训练日志曲线
│   └── plot_experiment_comparison.py # 多个实验训练曲线对比
├── dataset/Kvasir-SEG/
│   ├── images/
│   └── masks/
├── experiments/                    # 各实验的权重、日志、指标和预测图
├── README.md                       # 项目讨论提纲
├── USAGE.md                        # 本使用说明
└── result.md                       # 已完成实验的测试集结果汇总
```

## 2. 环境与数据

推荐使用项目虚拟环境：

```powershell
.\.venv\Scripts\python.exe --version
```

工程未提供锁定版本的 `requirements.txt`。若需新建环境，安装运行所需依赖：

```powershell
python -m pip install torch torchvision numpy pillow matplotlib tqdm
```

数据集目录必须严格为：

```text
dataset/Kvasir-SEG/images/
dataset/Kvasir-SEG/masks/
```

图像与掩码文件名应一一对应。下文以 `.\.venv\Scripts\python.exe` 代替 Python 解释器；若未使用虚拟环境，请替换为实际的 `python` 命令。

## 3. 实验分组

| 组别 | 模型 | 损失函数 | 训练脚本 | 测试脚本 |
|---|---|---|---|---|
| A | U-Net | BCE | `train.py` | `predict.py` |
| B | U-Net | BCE + Dice | `train_hybrid.py` | `predict_hybrid.py` |
| C | Attention U-Net | BCE | `train_attention.py` | `predict_attention.py` |
| D | Attention U-Net | BCE + Dice | `train_attention_hybrid.py` | `predict_attention_hybrid.py` |

所有训练入口都要求 `--batch_size` 和 `--lr`。两者必须为正数；`1e-4` 与 `0.0001` 代表同一学习率。混合损失脚本额外接受 `--loss_weights`，格式为 `BCE:Dice`，默认 `1:1`。

## 4. 单次训练

```powershell
# A: U-Net + BCE
.\.venv\Scripts\python.exe codes\train.py --batch_size 8 --lr 1e-4

# B: U-Net + BCE + Dice（默认 BCE:Dice=1:1）
.\.venv\Scripts\python.exe codes\train_hybrid.py --batch_size 8 --lr 1e-4

# B: 调整混合损失权重
.\.venv\Scripts\python.exe codes\train_hybrid.py --batch_size 8 --lr 1e-4 --loss_weights 2:1

# C: Attention U-Net + BCE
.\.venv\Scripts\python.exe codes\train_attention.py --batch_size 8 --lr 1e-4

# D: Attention U-Net + BCE + Dice
.\.venv\Scripts\python.exe codes\train_attention_hybrid.py --batch_size 8 --lr 1e-4 --loss_weights 1:1
```

`train_hybrid.py` 还支持 `--epochs`（默认 100）、`--patience`（默认 10）和 `--quick`。`--quick` 等价于 `--epochs 25 --patience 5`，用于快速排查流程，不能与完整训练结果直接作公平比较。

训练产物写入以下目录，其中学习率和损失权重会写入目录名：

```text
experiments/<模型名>_bs<批大小>_lr<学习率>[_bce<BCE权重>_dice<Dice权重>]/
```

例如：

```text
experiments/unet_baseline_bs8_lr0.0001/
experiments/unet_hybrid_bs8_lr0.0001_bce2_dice1/
experiments/attention_unet_hybrid_bs8_lr0.0001_bce1_dice1/
```

目录中包含最优权重 `best_*.pth`、训练日志 `train_log_*.csv`、训练曲线 `train_curve_*.png`、数据划分 `data_split.txt`；混合损失实验另有 `experiment_config.txt`。以相同配置再次训练会覆盖同名产物。

## 5. 测试与预测图

测试命令须提供权重路径与输出目录。建议将输出目录设为权重所在实验目录，使权重、指标和预测图保持对应。

```powershell
# A 组
.\.venv\Scripts\python.exe codes\predict.py `
  --weight_path experiments\unet_baseline_bs8_lr0.0001\best_unet_baseline_bs8_lr0.0001.pth `
  --output_dir experiments\unet_baseline_bs8_lr0.0001

# B 组
.\.venv\Scripts\python.exe codes\predict_hybrid.py `
  --weight_path experiments\unet_hybrid_bs8_lr0.0001_bce1_dice1\best_unet_hybrid_bs8_lr0.0001_bce1_dice1.pth `
  --output_dir experiments\unet_hybrid_bs8_lr0.0001_bce1_dice1

# C 组
.\.venv\Scripts\python.exe codes\predict_attention.py `
  --weight_path experiments\attention_unet_bce_bs8_lr0.0001\best_attention_unet_bce_bs8_lr0.0001.pth `
  --output_dir experiments\attention_unet_bce_bs8_lr0.0001

# D 组
.\.venv\Scripts\python.exe codes\predict_attention_hybrid.py `
  --weight_path experiments\attention_unet_hybrid_bs8_lr0.0001_bce1_dice1\best_attention_unet_hybrid_bs8_lr0.0001_bce1_dice1.pth `
  --output_dir experiments\attention_unet_hybrid_bs8_lr0.0001_bce1_dice1
```

对应输出包括 `test_metrics_*.txt`、`data_split.txt` 和预测示例图目录。测试指标为 Dice、IoU、Precision 与 Recall；完整已测结果见 [result.md](result.md)。

## 6. 批量参数实验与汇总

下列批量脚本依次运行 B 组学习率扫描（`1e-3`、`1e-4`、`3e-4`）、B 组损失权重扫描（`1:1`、`1:2`、`2:1`），并默认训练和测试 D 组的 `lr=1e-4, BCE:Dice=1:1` 方案。

```powershell
# 使用当前虚拟环境；默认 batch_size=8
.\.venv\Scripts\python.exe codes\run_param_experiments.py

# 指定批大小和解释器
.\.venv\Scripts\python.exe codes\run_param_experiments.py `
  --batch_size 8 `
  --python .\.venv\Scripts\python.exe

# 只运行 B 组两类扫描，不训练 D 组
.\.venv\Scripts\python.exe codes\run_param_experiments.py --batch_size 8 --skip_final

# 从 experiments/ 重新汇总表格和曲线
.\.venv\Scripts\python.exe codes\build_tuning_tables.py

# 指定汇总输出目录
.\.venv\Scripts\python.exe codes\build_tuning_tables.py --output_dir results\custom_summary
```

默认汇总目录为 `results/division4/`，包含学习率、权重和 A/B/C/D 消融的 Markdown/CSV 表格，以及对应曲线图。批量脚本会重新训练同名实验，因此运行前应确认是否需要保留已有权重和结果。

## 7. 训练曲线绘制

单个日志重绘：

```powershell
.\.venv\Scripts\python.exe codes\plot_training_log.py `
  --csv_path experiments\unet_hybrid_bs8_lr0.0001_bce1_dice1\train_log_hybrid.csv `
  --output_path experiments\unet_hybrid_bs8_lr0.0001_bce1_dice1\train_curve_from_csv.png
```

日志 CSV 必须包含 `epoch`、`train_loss`、`val_loss` 和 `val_dice` 四列。多个实验的训练/验证损失和验证 Dice 可用下列命令对比：

```powershell
.\.venv\Scripts\python.exe codes\plot_experiment_comparison.py `
  --csv_paths `
    experiments\unet_baseline_bs8_lr0.0001\train_log_baseline.csv `
    experiments\unet_hybrid_bs8_lr0.0001_bce1_dice1\train_log_hybrid.csv `
  --labels "A: U-Net + BCE" "B: U-Net + BCE + Dice" `
  --output_path results\curve_ablation_ab.png `
  --loss_ylabel "Loss"
```

`--csv_paths` 与 `--labels` 的数量必须一致。

## 8. 结果文件定位

每个实验目录的 `test_metrics_*.txt` 是该模型在测试集上的原始结果。根目录 [result.md](result.md) 汇总了当前 `experiments/` 下全部 15 份测试指标，并提供消融、学习率、损失权重与批大小对比和初步结论。
