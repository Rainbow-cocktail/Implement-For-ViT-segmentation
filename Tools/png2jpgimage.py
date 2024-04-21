# --------------------------------------------------------#
#   将所有图片转化为jpg格式的图片
#   转换完成后，可以将新文件夹里面图片替换掉JPEGImages文件夹下面图片
#   后续考虑可以内置这项工具
# --------------------------------------------------------#
import os

import numpy as np
from PIL import Image
from tqdm import tqdm


def PNG2JPG(root, target_path=None):
    """新产生的图片将另存为'Converted_JPG_Images'文件夹"""
    if not os.path.exists(root):
        raise FileNotFoundError(" '{}' does not exist".format(root))
    if target_path is None:
        target_path = os.path.join(os.path.split(root)[0], 'Converted_JPG_Images')  # target_path 是与图片文件夹同级的文件夹
        if not os.path.exists(target_path):
            os.makedirs(target_path, exist_ok=True)

    for image_name in tqdm(os.listdir(root)):
        image_path = os.path.join(root, image_name)
        image = Image.open(image_path)
        image = image.convert('RGB')
        image.save(os.path.join(target_path, os.path.splitext(image_name)[0] + '.jpg'))


if __name__ == '__main__':
    # 在这里输入图片目录路径
    root = r"../../data/WindowsInHailDisaster_photoes/JPEGImages"
    PNG2JPG(root)
    print("Done!")
