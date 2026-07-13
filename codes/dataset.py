import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# ====================== 自动定位路径（核心修复）======================
# 获取当前脚本所在的 codes 目录绝对路径
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根目录 = codes 的上一级
PROJECT_ROOT = os.path.dirname(CODE_DIR)
# 数据集固定路径
DATA_ROOT = os.path.join(PROJECT_ROOT, "dataset", "Kvasir-SEG")

# ====================== 全局固定随机种子（全组统一42）======================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ====================== 8:1:1 数据集划分 ======================
def split_dataset(data_root=DATA_ROOT, save_path=None):
    # 默认保存路径
    if save_path is None:
        save_path = os.path.join(PROJECT_ROOT, "results", "data_split.txt")
    
    img_dir = os.path.join(data_root, "images")
    # 先检查路径是否存在，不存在直接报错提示
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"找不到图像目录，请检查数据集路径: {img_dir}")
    
    all_names = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    total = len(all_names)
    
    train_num = int(total * 0.8)
    val_num = int(total * 0.1)
    test_num = total - train_num - val_num

    random.shuffle(all_names)
    train_list = all_names[:train_num]
    val_list = all_names[train_num : train_num + val_num]
    test_list = all_names[train_num + val_num :]

    # 保存划分记录，交付全组使用
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"固定随机种子: 42\n")
        f.write(f"总样本数: {total}\n")
        f.write(f"训练集: {train_num} 张\n")
        f.write(f"验证集: {val_num} 张\n")
        f.write(f"测试集: {test_num} 张\n\n")
        f.write("=== 训练集列表 ===\n" + "\n".join(train_list) + "\n\n")
        f.write("=== 验证集列表 ===\n" + "\n".join(val_list) + "\n\n")
        f.write("=== 测试集列表 ===\n" + "\n".join(test_list) + "\n")
    
    print(f"数据集划分完成：训练{train_num}张，验证{val_num}张，测试{test_num}张")
    print(f"划分记录已保存至: {save_path}")
    return train_list, val_list, test_list

# ====================== 数据集加载类 ======================
class KvasirDataset(Dataset):
    def __init__(self, data_root=DATA_ROOT, name_list=None, img_size=256):
        self.img_dir = os.path.join(data_root, "images")
        self.mask_dir = os.path.join(data_root, "masks")
        self.names = name_list
        self.img_size = img_size

        # 原图预处理：resize + 转张量 + 归一化
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 掩码预处理：resize + 转张量（自动从0-255归一化到0-1）
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, name)).convert("L")  # 转灰度图

        img_tensor = self.img_transform(img)
        mask_tensor = self.mask_transform(mask)
        # 强制二值化，消除掩码灰度误差
        mask_tensor = (mask_tensor > 0.5).float()
        return img_tensor, mask_tensor

# ====================== 分割指标计算（全组统一标准）======================
def calculate_metrics(pred_logits, mask):
    """
    输入：模型输出logits、真值掩码（0-1张量）
    输出：Dice, IoU, Precision, Recall
    """
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    smooth = 1e-6

    intersection = (pred * mask).sum()
    union = pred.sum() + mask.sum()
    
    dice = (2 * intersection + smooth) / (union + smooth)
    iou = (intersection + smooth) / (union - intersection + smooth)
    
    tp = intersection
    fp = (pred - mask).clip(min=0).sum()
    fn = (mask - pred).clip(min=0).sum()
    
    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)

    return dice.item(), iou.item(), precision.item(), recall.item()

# ====================== 单独运行：划分数据集并验证加载 ======================
if __name__ == "__main__":
    set_seed(42)
    print(f"当前数据集路径: {DATA_ROOT}")
    train_list, val_list, test_list = split_dataset()
    
    # 验证数据集加载（关键字参数传参，避免顺序错误）
    train_dataset = KvasirDataset(name_list=train_list, img_size=256)
    img, mask = train_dataset[0]
    print(f"\n单样本验证：")
    print(f"图像张量形状: {img.shape}")
    print(f"掩码张量形状: {mask.shape}")
    print(f"掩码取值范围: [{mask.min().item()}, {mask.max().item()}]")
    print("✅ 数据集加载验证通过！")