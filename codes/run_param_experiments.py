import argparse
import subprocess
import sys
from pathlib import Path

from experiment_utils import experiment_name, positive_learning_rate


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments"
RESULTS_DIR = PROJECT_ROOT / "results" / "division4"

DEFAULT_BATCH_SIZE = 8
DEFAULT_LOSS_WEIGHTS = "1:1"
DEFAULT_LEARNING_RATE = "1e-4"

LR_SWEEP = ["1e-3", "1e-4", "3e-4"]
WEIGHT_SWEEP = ["1:1", "1:2", "2:1"]


def parse_arguments():
    parser = argparse.ArgumentParser(description="批量运行混合损失参数实验。")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="统一批大小，默认 8。")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="用于启动训练脚本的 Python 解释器路径。",
    )
    parser.add_argument(
        "--skip_final",
        action="store_true",
        help="仅运行学习率与损失权重扫描，不训练 D 组最终方案。",
    )
    return parser.parse_args()


def run_training(script_name, batch_size, learning_rate, loss_weights, python_executable):
    command = [
        python_executable,
        str(CODE_DIR / script_name),
        "--batch_size",
        str(batch_size),
        "--lr",
        learning_rate,
        "--loss_weights",
        loss_weights,
    ]
    print("\n" + "=" * 72)
    print("执行命令:", " ".join(command))
    print("=" * 72)
    subprocess.run(command, check=True)


def run_predict(script_name, experiment_dir, python_executable):
    weight_candidates = list(experiment_dir.glob("best_*.pth"))
    if not weight_candidates:
        print(f"跳过预测：未找到权重文件 {experiment_dir}")
        return
    weight_path = weight_candidates[0]
    command = [
        python_executable,
        str(CODE_DIR / script_name),
        "--weight_path",
        str(weight_path),
        "--output_dir",
        str(experiment_dir),
    ]
    print("执行预测:", " ".join(command))
    subprocess.run(command, check=True)


def main():
    arguments = parse_arguments()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("阶段 1/3：学习率扫描（U-Net + BCE + Dice，固定 BCE:Dice = 1:1）")
    for learning_rate in LR_SWEEP:
        run_training(
            "train_hybrid.py",
            arguments.batch_size,
            learning_rate,
            DEFAULT_LOSS_WEIGHTS,
            arguments.python,
        )
        tag = experiment_name(
            "unet_hybrid",
            arguments.batch_size,
            positive_learning_rate(learning_rate),
            1,
            1,
        )
        run_predict("predict_hybrid.py", EXPERIMENT_ROOT / tag, arguments.python)

    print("\n阶段 2/3：混合损失权重扫描（固定 lr = 1e-4）")
    for loss_weights in WEIGHT_SWEEP:
        bce_weight, dice_weight = [float(part) for part in loss_weights.split(":")]
        run_training(
            "train_hybrid.py",
            arguments.batch_size,
            DEFAULT_LEARNING_RATE,
            loss_weights,
            arguments.python,
        )
        tag = experiment_name(
            "unet_hybrid",
            arguments.batch_size,
            positive_learning_rate(DEFAULT_LEARNING_RATE),
            bce_weight,
            dice_weight,
        )
        run_predict("predict_hybrid.py", EXPERIMENT_ROOT / tag, arguments.python)

    if not arguments.skip_final:
        print("\n阶段 3/3：D 组最终方案（Attention U-Net + BCE + Dice，lr=1e-4, BCE:Dice=1:1）")
        run_training(
            "train_attention_hybrid.py",
            arguments.batch_size,
            DEFAULT_LEARNING_RATE,
            DEFAULT_LOSS_WEIGHTS,
            arguments.python,
        )
        tag = experiment_name(
            "attention_unet_hybrid",
            arguments.batch_size,
            positive_learning_rate(DEFAULT_LEARNING_RATE),
            1,
            1,
        )
        run_predict("predict_attention_hybrid.py", EXPERIMENT_ROOT / tag, arguments.python)

    print("\n全部参数实验已完成。请运行 build_tuning_tables.py 生成调参表格。")


if __name__ == "__main__":
    main()
