import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments"
RESULTS_DIR = PROJECT_ROOT / "results" / "division4"


def parse_arguments():
    parser = argparse.ArgumentParser(description="汇总混合损失参数实验并生成调参表格。")
    parser.add_argument(
        "--output_dir",
        default=str(RESULTS_DIR),
        help="调参表格与汇总图输出目录，默认 results/division4。",
    )
    return parser.parse_args()


def read_best_val_dice(experiment_dir):
    config_path = experiment_dir / "experiment_config.txt"
    if config_path.is_file():
        for line in config_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("best_val_dice:"):
                return float(line.split(":", 1)[1].strip())

    for log_name in (
        "train_log_hybrid.csv",
        "train_log_attention_hybrid.csv",
        "train_log_baseline.csv",
        "train_log_attention_bce.csv",
    ):
        log_path = experiment_dir / log_name
        if not log_path.is_file():
            continue
        with log_path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            return max(float(row["val_dice"]) for row in reader)
    return None


def read_test_metrics(experiment_dir):
    metrics = {"dice": None, "iou": None, "precision": None, "recall": None}
    for metrics_name in (
        "test_metrics_hybrid.txt",
        "test_metrics_attention_hybrid.txt",
        "test_metrics_baseline.txt",
        "test_metrics_attention_bce.txt",
    ):
        metrics_path = experiment_dir / metrics_name
        if not metrics_path.is_file():
            continue
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if "Dice" in line and ":" in line:
                metrics["dice"] = float(line.split(":")[-1].strip())
            elif "IoU" in line and ":" in line:
                metrics["iou"] = float(line.split(":")[-1].strip())
            elif "Precision" in line and ":" in line:
                metrics["precision"] = float(line.split(":")[-1].strip())
            elif "Recall" in line and ":" in line:
                metrics["recall"] = float(line.split(":")[-1].strip())
        if metrics["dice"] is not None:
            return metrics
    return metrics


def parse_experiment_name(name):
    parts = name.split("_")
    info = {
        "experiment": name,
        "model": None,
        "batch_size": None,
        "learning_rate": None,
        "bce_weight": None,
        "dice_weight": None,
    }

    if name.startswith("unet_hybrid"):
        info["model"] = "U-Net + BCE + Dice"
        info["group"] = "B"
    elif name.startswith("attention_unet_hybrid"):
        info["model"] = "Attention U-Net + BCE + Dice"
        info["group"] = "D"
    else:
        return None

    for index, part in enumerate(parts):
        if part.startswith("bs") and part[2:].isdigit():
            info["batch_size"] = int(part[2:])
        elif part.startswith("lr"):
            info["learning_rate"] = part[2:].replace("p", ".")
        elif part.startswith("bce"):
            info["bce_weight"] = part[3:].replace("p", ".")
        elif part.startswith("dice"):
            info["dice_weight"] = part[4:].replace("p", ".")

    return info


def collect_hybrid_experiments():
    records = []
    if not EXPERIMENT_ROOT.is_dir():
        return records

    for experiment_dir in sorted(EXPERIMENT_ROOT.iterdir()):
        if not experiment_dir.is_dir():
            continue
        info = parse_experiment_name(experiment_dir.name)
        if info is None:
            continue

        info["best_val_dice"] = read_best_val_dice(experiment_dir)
        info.update(read_test_metrics(experiment_dir))
        records.append(info)
    return records


def format_float(value):
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def write_markdown_table(path, title, headers, rows):
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv_table(path, headers, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)


def build_lr_sweep_table(records):
    rows = []
    for record in records:
        if record.get("group") != "B":
            continue
        if record.get("bce_weight") != "1" or record.get("dice_weight") != "1":
            continue
        rows.append(
            [
                record["learning_rate"],
                format_float(record["best_val_dice"]),
                format_float(record["dice"]),
                format_float(record["iou"]),
                format_float(record["precision"]),
                format_float(record["recall"]),
                record["experiment"],
            ]
        )
    rows.sort(key=lambda row: row[0])
    return rows


def build_weight_sweep_table(records):
    rows = []
    for record in records:
        if record.get("group") != "B":
            continue
        if record.get("learning_rate") != "0.0001":
            continue
        rows.append(
            [
                f"{record['bce_weight']}:{record['dice_weight']}",
                format_float(record["best_val_dice"]),
                format_float(record["dice"]),
                format_float(record["iou"]),
                format_float(record["precision"]),
                format_float(record["recall"]),
                record["experiment"],
            ]
        )
    rows.sort(key=lambda row: row[0])
    return rows


def build_ablation_table(records):
    baseline_dir = EXPERIMENT_ROOT / "unet_baseline_bs8_lr0.0001"
    attention_dir = EXPERIMENT_ROOT / "attention_unet_bce_bs8_lr0.0001"

    rows = []
    if baseline_dir.is_dir():
        metrics = read_test_metrics(baseline_dir)
        rows.append(
            [
                "A",
                "U-Net + BCE",
                "1e-4",
                "-",
                format_float(read_best_val_dice(baseline_dir)),
                format_float(metrics["dice"]),
                "unet_baseline_bs8_lr0.0001",
            ]
        )
    if attention_dir.is_dir():
        metrics = read_test_metrics(attention_dir)
        rows.append(
            [
                "C",
                "Attention U-Net + BCE",
                "1e-4",
                "-",
                format_float(read_best_val_dice(attention_dir)),
                format_float(metrics["dice"]),
                "attention_unet_bce_bs8_lr0.0001",
            ]
        )

    for record in records:
        if record.get("group") == "B" and record.get("learning_rate") == "0.0001" and record.get("bce_weight") == "1" and record.get("dice_weight") == "1":
            rows.append(
                [
                    "B",
                    record["model"],
                    "1e-4",
                    "1:1",
                    format_float(record["best_val_dice"]),
                    format_float(record["dice"]),
                    record["experiment"],
                ]
            )
        if record.get("group") == "D" and record.get("learning_rate") == "0.0001" and record.get("bce_weight") == "1" and record.get("dice_weight") == "1":
            rows.append(
                [
                    "D",
                    record["model"],
                    "1e-4",
                    "1:1",
                    format_float(record["best_val_dice"]),
                    format_float(record["dice"]),
                    record["experiment"],
                ]
            )
    return rows


def plot_lr_comparison(records, output_path):
    figure, axis = plt.subplots(figsize=(8, 5))
    plotted = False
    for learning_rate in ["0.001", "0.0001", "0.0003"]:
        for record in records:
            if record.get("group") != "B":
                continue
            if record.get("learning_rate") != learning_rate:
                continue
            if record.get("bce_weight") != "1" or record.get("dice_weight") != "1":
                continue
            log_path = EXPERIMENT_ROOT / record["experiment"] / "train_log_hybrid.csv"
            if not log_path.is_file():
                continue
            with log_path.open("r", newline="", encoding="utf-8-sig") as file:
                reader = csv.DictReader(file)
                epochs = [int(row["epoch"]) for row in reader]
                file.seek(0)
                reader = csv.DictReader(file)
                val_dice = [float(row["val_dice"]) for row in reader]
            axis.plot(epochs, val_dice, label=f"lr={learning_rate}")
            plotted = True
            break

    if plotted:
        axis.set_title("Learning Rate Sweep (U-Net + BCE + Dice, BCE:Dice=1:1)")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Validation Dice")
        axis.grid(alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_weight_comparison(records, output_path):
    figure, axis = plt.subplots(figsize=(8, 5))
    plotted = False
    for weight_tag in ["1:1", "1:2", "2:1"]:
        bce_weight, dice_weight = weight_tag.split(":")
        for record in records:
            if record.get("group") != "B":
                continue
            if record.get("learning_rate") != "0.0001":
                continue
            if record.get("bce_weight") != bce_weight or record.get("dice_weight") != dice_weight:
                continue
            log_path = EXPERIMENT_ROOT / record["experiment"] / "train_log_hybrid.csv"
            if not log_path.is_file():
                continue
            with log_path.open("r", newline="", encoding="utf-8-sig") as file:
                reader = csv.DictReader(file)
                epochs = [int(row["epoch"]) for row in reader]
                file.seek(0)
                reader = csv.DictReader(file)
                val_dice = [float(row["val_dice"]) for row in reader]
            axis.plot(epochs, val_dice, label=f"BCE:Dice={weight_tag}")
            plotted = True
            break

    if plotted:
        axis.set_title("Loss Weight Sweep (U-Net + BCE + Dice, lr=1e-4)")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Validation Dice")
        axis.grid(alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_ablation_comparison(output_path):
    """绘制已有 A/C 组（及可选 B/D 组）验证 Dice 对比曲线。"""
    candidates = [
        ("unet_baseline_bs8_lr0.0001", "train_log_baseline.csv", "A: U-Net+BCE"),
        ("unet_hybrid_bs8_lr0.0001_bce1_dice1", "train_log_hybrid.csv", "B: U-Net+Hybrid"),
        ("attention_unet_bce_bs8_lr0.0001", "train_log_attention_bce.csv", "C: Attn+BCE"),
        (
            "attention_unet_hybrid_bs8_lr0.0001_bce1_dice1",
            "train_log_attention_hybrid.csv",
            "D: Attn+Hybrid",
        ),
    ]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    plotted = False
    for experiment_name, log_name, label in candidates:
        log_path = EXPERIMENT_ROOT / experiment_name / log_name
        if not log_path.is_file():
            continue
        with log_path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            rows = list(reader)
        epochs = [int(row["epoch"]) for row in rows]
        val_loss = [float(row["val_loss"]) for row in rows]
        val_dice = [float(row["val_dice"]) for row in rows]
        axes[0].plot(epochs, val_loss, label=f"{label} (Val)")
        axes[1].plot(epochs, val_dice, label=label)
        plotted = True

    if plotted:
        axes[0].set_title("Validation Loss Comparison (A/B/C/D)")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)

        axes[1].set_title("Validation Dice Comparison (A/B/C/D)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Dice Coefficient")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)

        figure.tight_layout()
        figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main():
    arguments = parse_arguments()
    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = collect_hybrid_experiments()
    lr_headers = ["学习率", "最佳验证 Dice", "测试 Dice", "测试 IoU", "测试 Precision", "测试 Recall", "实验目录"]
    weight_headers = ["BCE:Dice", "最佳验证 Dice", "测试 Dice", "测试 IoU", "测试 Precision", "测试 Recall", "实验目录"]
    ablation_headers = ["组别", "模型", "学习率", "损失权重", "最佳验证 Dice", "测试 Dice", "实验目录"]

    lr_rows = build_lr_sweep_table(records)
    weight_rows = build_weight_sweep_table(records)
    ablation_rows = build_ablation_table(records)

    write_markdown_table(output_dir / "tuning_lr_sweep.md", "学习率调参结果（B 组，BCE:Dice=1:1）", lr_headers, lr_rows)
    write_markdown_table(output_dir / "tuning_weight_sweep.md", "混合损失权重调参结果（B 组，lr=1e-4）", weight_headers, weight_rows)
    write_markdown_table(output_dir / "ablation_summary.md", "四组消融实验汇总", ablation_headers, ablation_rows)

    write_csv_table(output_dir / "tuning_lr_sweep.csv", lr_headers, lr_rows)
    write_csv_table(output_dir / "tuning_weight_sweep.csv", weight_headers, weight_rows)
    write_csv_table(output_dir / "ablation_summary.csv", ablation_headers, ablation_rows)

    plot_lr_comparison(records, output_dir / "curve_lr_sweep.png")
    plot_weight_comparison(records, output_dir / "curve_weight_sweep.png")
    plot_ablation_comparison(output_dir / "curve_ablation_ac.png")

    print(f"调参表格已保存至: {output_dir}")
    print("生成文件:")
    for file_name in (
        "tuning_lr_sweep.md",
        "tuning_weight_sweep.md",
        "ablation_summary.md",
        "tuning_lr_sweep.csv",
        "tuning_weight_sweep.csv",
        "ablation_summary.csv",
        "curve_lr_sweep.png",
        "curve_weight_sweep.png",
        "curve_ablation_ac.png",
    ):
        print(f"  - {output_dir / file_name}")


if __name__ == "__main__":
    main()
