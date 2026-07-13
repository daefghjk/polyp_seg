import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REQUIRED_COLUMNS = ("epoch", "train_loss", "val_loss", "val_dice")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="读取训练日志 CSV 并绘制损失与验证集 Dice 曲线。"
    )
    parser.add_argument(
        "--csv_path",
        required=True,
        help="训练日志 CSV 文件路径。",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="输出 PNG 图像路径。",
    )
    return parser.parse_args()


def load_training_log(csv_path):
    if not csv_path.is_file():
        raise FileNotFoundError(f"找不到训练日志 CSV 文件: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("CSV 文件缺少表头。")

        missing_columns = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing_columns:
            raise ValueError(f"CSV 文件缺少必要字段: {', '.join(missing_columns)}")

        history = {column: [] for column in REQUIRED_COLUMNS}
        for row_number, row in enumerate(reader, start=2):
            try:
                for column in REQUIRED_COLUMNS:
                    history[column].append(float(row[column]))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"CSV 第 {row_number} 行的字段 {column} 不是有效数值。"
                ) from error

    if not history["epoch"]:
        raise ValueError("CSV 文件不包含训练记录。")
    return history


def plot_history(history, output_path):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["epoch"], history["train_loss"], label="Train Loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["epoch"], history["val_dice"], label="Val Dice", color="orange")
    axes[1].set_title("Validation Dice Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Coefficient")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main():
    arguments = parse_arguments()
    csv_path = Path(arguments.csv_path)
    output_path = Path(arguments.output_path)

    history = load_training_log(csv_path)
    plot_history(history, output_path)
    print(f"训练曲线已保存至: {output_path}")


if __name__ == "__main__":
    main()
