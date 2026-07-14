# 分工四：混合损失与参数实验

本目录对应五人分工中的**第 4 项**，交付物包括：

1. 混合损失函数实现（BCE + Dice）
2. 参数调优实验脚本（学习率、损失权重）
3. 调参结果表格（Markdown / CSV）
4. 训练曲线与对比图

## 1. 混合损失函数

实现文件：`codes/losses.py`

总损失定义为：

\[
\mathcal{L} = \lambda_{\text{BCE}} \cdot \mathcal{L}_{\text{BCE}} + \lambda_{\text{Dice}} \cdot \mathcal{L}_{\text{Dice}}
\]

其中：

- \(\mathcal{L}_{\text{BCE}}\)：像素级二值交叉熵（`BCEWithLogitsLoss`），保证分类稳定；
- \(\mathcal{L}_{\text{Dice}} = 1 - \text{Dice}\)：Soft Dice Loss，缓解息肉区域较小导致的前景/背景不平衡。

默认权重为 `BCE:Dice = 1:1`，可通过命令行参数 `--loss_weights` 调整为 `1:2`、`2:1` 等。

## 2. 新增脚本一览

| 脚本 | 对应组别 | 功能 |
|---|---|---|
| `train_hybrid.py` | B | U-Net + BCE + Dice 训练 |
| `train_attention_hybrid.py` | D | Attention U-Net + BCE + Dice 训练 |
| `predict_hybrid.py` | B | 混合损失 U-Net 测试与可视化 |
| `predict_attention_hybrid.py` | D | 混合损失 Attention U-Net 测试与可视化 |
| `run_param_experiments.py` | - | 批量运行参数实验 |
| `build_tuning_tables.py` | - | 汇总调参表格与对比曲线 |
| `plot_experiment_comparison.py` | - | 自定义多实验训练曲线对比 |

**对已有代码的改动**：仅扩展 `experiment_utils.py`，增加损失权重解析与实验目录命名，不修改原有训练/预测逻辑。

## 3. 单次训练示例

```powershell
# B 组：U-Net + BCE + Dice（默认 1:1，完整训练）
python codes/train_hybrid.py --batch_size 8 --lr 1e-4

# B 组快速模式（CPU 推荐，约 2-4 小时）
python codes/train_hybrid.py --batch_size 16 --lr 1e-4 --quick

# 调整混合损失权重为 1:2
python codes/train_hybrid.py --batch_size 8 --lr 1e-4 --loss_weights 1:2

# D 组：Attention U-Net + BCE + Dice
python codes/train_attention_hybrid.py --batch_size 8 --lr 1e-4 --loss_weights 1:1
```

实验目录命名示例：

```text
experiments/unet_hybrid_bs8_lr0.0001_bce1_dice1/
experiments/unet_hybrid_bs8_lr0.0001_bce1_dice2/
experiments/attention_unet_hybrid_bs8_lr0.0001_bce1_dice1/
```

每个实验目录自动保存：

- `train_log_hybrid.csv` / `train_log_attention_hybrid.csv`
- `train_curve_hybrid.png` / `train_curve_attention_hybrid.png`
- `experiment_config.txt`（记录超参数与最佳验证 Dice）
- `best_*.pth`（最优权重）

## 4. 参数实验方案

按 README 要求，选取有代表性的设置（非全排列穷举）：

### 4.1 学习率扫描（固定 BCE:Dice = 1:1）

| 学习率 | 目的 |
|---|---|
| `1e-3` | 较大步长，观察是否震荡 |
| `1e-4` | 与 A/C 组对齐的基准 |
| `3e-4` | 中等步长折中 |

### 4.2 混合损失权重扫描（固定 lr = 1e-4）

| BCE:Dice | 目的 |
|---|---|
| `1:1` | 均衡组合 |
| `1:2` | 强化区域重叠优化 |
| `2:1` | 强化像素级分类稳定 |

### 4.3 最终方案

在 `lr=1e-4, BCE:Dice=1:1` 下训练 D 组（Attention U-Net + BCE + Dice）。

## 5. 一键批量运行

```powershell
# 安装依赖
pip install -r requirements.txt

# 依次完成：学习率扫描 → 权重扫描 → D 组训练（含自动预测）
python codes/run_param_experiments.py --batch_size 8

# 仅跑调参、不训练 D 组
python codes/run_param_experiments.py --batch_size 8 --skip_final

# 生成调参表格与对比曲线
python codes/build_tuning_tables.py
```

输出目录：`results/division4/`

| 文件 | 内容 |
|---|---|
| `tuning_lr_sweep.md` / `.csv` | 学习率调参表 |
| `tuning_weight_sweep.md` / `.csv` | 损失权重调参表 |
| `ablation_summary.md` / `.csv` | A/B/C/D 四组消融汇总 |
| `curve_lr_sweep.png` | 不同学习率验证 Dice 曲线 |
| `curve_weight_sweep.png` | 不同损失权重验证 Dice 曲线 |

## 6. 自定义训练曲线对比

```powershell
python codes/plot_experiment_comparison.py `
  --csv_paths `
    experiments/unet_baseline_bs8_lr0.0001/train_log_baseline.csv `
    experiments/unet_hybrid_bs8_lr0.0001_bce1_dice1/train_log_hybrid.csv `
  --labels "A: U-Net+BCE" "B: U-Net+Hybrid" `
  --output_path results/division4/curve_ablation_ab.png `
  --loss_ylabel "Loss"
```

## 7. 测试集评估

```powershell
python codes/predict_hybrid.py `
  --weight_path experiments/unet_hybrid_bs8_lr0.0001_bce1_dice1/best_unet_hybrid_bs8_lr0.0001_bce1_dice1.pth `
  --output_dir experiments/unet_hybrid_bs8_lr0.0001_bce1_dice1

python codes/predict_attention_hybrid.py `
  --weight_path experiments/attention_unet_hybrid_bs8_lr0.0001_bce1_dice1/best_attention_unet_hybrid_bs8_lr0.0001_bce1_dice1.pth `
  --output_dir experiments/attention_unet_hybrid_bs8_lr0.0001_bce1_dice1
```

## 8. 报告撰写建议

在实验章节可直接引用 `results/division4/` 下的表格与曲线，重点讨论：

1. 混合损失相对纯 BCE（A 组）是否提升 Dice / Recall；
2. 不同 `BCE:Dice` 权重对小息肉、边界模糊样本的影响；
3. 学习率过大（`1e-3`）是否导致验证集震荡；
4. D 组（注意力 + 混合损失）是否为最优组合。

## 9. 与全组实验的对应关系

| 组别 | 模型 | 损失 | 训练脚本 |
|---|---|---|---|
| A | U-Net | BCE | `train.py`（已有） |
| B | U-Net | BCE + Dice | `train_hybrid.py`（新增） |
| C | Attention U-Net | BCE | `train_attention.py`（已有） |
| D | Attention U-Net | BCE + Dice | `train_attention_hybrid.py`（新增） |
