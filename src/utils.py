import logging
import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


sys.path.append(os.getcwd())
from src.net.thinking import Thinking
from src.net.jepa_thinking import JepaThinking


def logging_args(args, name=""):
    # 格式化打印参数，提高可读性
    logging.info("=" * 50)
    logging.info(f"{name} Parameters:")
    logging.info("=" * 50)
    for key, value in vars(args).items():
        logging.info(f"  {key:20}: {value}")
    logging.info("=" * 50)


def setup_logging(out_dir):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(process)d %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(out_dir, "run.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )


def pre_processing(image, width, height):
    image = cv2.cvtColor(cv2.resize(image, (width, height)), cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    return image[None, :, :].astype(np.float32)


def get_device():
    """自动检测并返回最佳设备"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def save_np_as_image(array, save_path="outputs/test_images"):
    """
    将numpy数组保存为图片文件，使用随机时间戳作为文件名

    Args:
        np_array: numpy数组，表示图像
        save_path: 保存路径（目录）

    Returns:
        str: 保存的完整文件路径
    """
    # 确保保存目录存在
    os.makedirs(save_path, exist_ok=True)

    if isinstance(array, torch.Tensor):
        # 如果是torch张量，转换为numpy数组
        np_array = array.cpu().numpy()
    elif isinstance(array, np.ndarray):
        np_array = array

    np_array = np_array.squeeze()
    # 确保数据类型正确
    if np_array.dtype != np.uint8:
        # 如果是浮点数，假设范围在[0,1]或[0,255]
        if np_array.max() <= 1.0:
            np_array = (np_array * 255).astype(np.uint8)
        else:
            np_array = np_array.astype(np.uint8)

    # 使用PIL保存图片
    pil_image = Image.fromarray(np_array)

    # 生成基于时间戳格式化的文件名

    filename = f"image_{time.strftime('%Y%m%d_%H%M%S')}_{pil_image.size}.png"
    full_path = os.path.join(save_path, filename)
    pil_image.save(full_path)

    return full_path


def save_model(model: nn.Module, save_path, config_dict, attr_dict):
    """
    保存模型和配置
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config_dict,
            "attrs": attr_dict,
        },
        save_path,
    )


def load_model(class_name, model_path):
    # check model_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    cp = torch.load(model_path, weights_only=False)

    sd = cp["model_state_dict"]
    attr_dict = cp["attrs"]
    config_dict = cp["config"]
    print("*" * 60)
    print(f"Loaded model from: {model_path}")
    print(f"config_dict: {config_dict}")
    print(f"attr_dict: {attr_dict}")
    print("*" * 60)

    model = class_name(config_dict)
    model.load_state_dict(sd)
    return model


if __name__ == "__main__":
    load_model(
        JepaThinking, "outputs/compare/train_2025_0815_165647/final_model_4000.pth"
    )
