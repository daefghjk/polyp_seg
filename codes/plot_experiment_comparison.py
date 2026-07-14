import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REQUIRED_COLUMNS = ("epoch", "train_loss", "val_loss", "val_dice")


def parse_arguments():
    parser = argparse.ArgumentParser(description="对比多个实验的训练曲线。")
    parser.add_argument(
        "--csv_paths",
        nargs="+",
        required=True,
        help="一个或多个训练日志 CSV 路径。",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="与 CSV 一一对应的图例标签。",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="输出对比图路径。",
    )
    parser.add_argument(
        "--loss_ylabel",
        default="Hybrid Loss",
        help="损失曲线纵轴标签。",
    )
    return parser.parse_args()


def load_training_log(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"找不到训练日志 CSV 文件: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV 文件缺少表头: {csv_path}")

        missing_columns = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing_columns:
            raise ValueError(f"CSV 文件缺少必要字段: {', '.join(missing_columns)}")

        history = {column: [] for column in REQUIRED_COLUMNS}
        for row in reader:
            for column in REQUIRED_COLUMNS:
                history[column].append(float(row[column]))
    return history


def main():
    arguments = parse_arguments()
    if len(arguments.csv_paths) != len(arguments.labels):
        raise ValueError("csv_paths 与 labels 数量必须一致。")

    output_path = Path(arguments.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for csv_path, label in zip(arguments.csv_paths, arguments.labels):
        history = load_training_log(csv_path)
        axes[0].plot(history["epoch"], history["train_loss"], label=f"{label} (Train)")
        axes[0].plot(history["epoch"], history["val_loss"], linestyle="--", label=f"{label} (Val)")
        axes[1].plot(history["epoch"], history["val_dice"], label=label)

    axes[0].set_title("Loss Curve Comparison")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel(arguments.loss_ylabel)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Validation Dice Comparison")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Coefficient")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    print(f"对比训练曲线已保存至: {output_path}")


if __name__ == "__main__":
    main()
