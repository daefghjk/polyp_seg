# 四组消融实验汇总

| 组别 | 模型 | 学习率 | 损失权重 | 最佳验证 Dice | 测试 Dice | 实验目录 |
| --- | --- | --- | --- | --- | --- | --- |
| A | U-Net + BCE | 1e-4 | - | 0.7899 | 0.7887 | unet_baseline_bs8_lr0.0001 |
| C | Attention U-Net + BCE | 1e-4 | - | 0.8010 | 0.8085 | attention_unet_bce_bs8_lr0.0001 |
| B | U-Net + BCE + Dice | 1e-4 | 1:1 | 0.7666 | 0.7851 | unet_hybrid_bs16_lr0.0001_bce1_dice1 |
| B | U-Net + BCE + Dice | 1e-4 | 1:1 | - | - | unet_hybrid_bs8_lr0.0001_bce1_dice1 |
