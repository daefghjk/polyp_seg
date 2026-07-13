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


def experiment_name(model_name, batch_size, learning_rate):
    return f"{model_name}_bs{batch_size}_lr{format_learning_rate(learning_rate)}"
