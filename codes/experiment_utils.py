import argparse
from decimal import Decimal, InvalidOperation


def positive_integer(value):
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("batch_size 必须是正整数。") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("batch_size 必须大于 0。")
    return number


def positive_learning_rate(value):
    try:
        number = Decimal(value)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError("lr 必须是正数。") from error
    if not number.is_finite() or number <= 0:
        raise argparse.ArgumentTypeError("lr 必须是有限的正数。")
    return number


def format_learning_rate(learning_rate):
    text = format(learning_rate.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def loss_weight_ratio(value):
    text = str(value).strip()
    if ":" not in text:
        raise argparse.ArgumentTypeError("损失权重必须使用 BCE:Dice 格式，例如 1:1、1:2、2:1。")

    parts = text.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("损失权重必须使用 BCE:Dice 格式，例如 1:1、1:2、2:1。")

    try:
        bce_weight = float(parts[0])
        dice_weight = float(parts[1])
    except ValueError as error:
        raise argparse.ArgumentTypeError("损失权重必须是正数。") from error

    if bce_weight <= 0 or dice_weight <= 0:
        raise argparse.ArgumentTypeError("损失权重必须是正数。")
    return bce_weight, dice_weight


def format_loss_weight_tag(bce_weight, dice_weight):
    def format_weight(weight):
        if float(weight).is_integer():
            return str(int(weight))
        return str(weight).replace(".", "p")

    return f"bce{format_weight(bce_weight)}_dice{format_weight(dice_weight)}"


def experiment_name(model_name, batch_size, learning_rate, bce_weight=None, dice_weight=None):
    tag = f"{model_name}_bs{batch_size}_lr{format_learning_rate(learning_rate)}"
    if bce_weight is not None and dice_weight is not None:
        tag = f"{tag}_{format_loss_weight_tag(bce_weight, dice_weight)}"
    return tag
