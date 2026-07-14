# 分工四实验结果目录

本目录存放混合损失与参数实验的汇总产物，由 `codes/build_tuning_tables.py` 自动生成。

## 文件说明

| 文件 | 说明 |
|---|---|
| `tuning_lr_sweep.md/.csv` | 学习率扫描（B 组，BCE:Dice=1:1） |
| `tuning_weight_sweep.md/.csv` | 损失权重扫描（B 组，lr=1e-4） |
| `ablation_summary.md/.csv` | A/B/C/D 四组消融汇总 |
| `curve_lr_sweep.png` | 学习率对比训练曲线 |
| `curve_weight_sweep.png` | 损失权重对比训练曲线 |
| `curve_ablation_ac.png` | 四组验证 Dice/损失对比曲线 |

## 如何更新

完成训练后重新生成：

```powershell
python codes/build_tuning_tables.py
```

批量运行全部参数实验：

```powershell
python codes/run_param_experiments.py --batch_size 8
```
