"""
将labelme得到的png彩色标签图转换为单通道的可用于训练的标签图
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt


def Get_colormap():
    """不同数据集需要修改"""
    cmap = np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0]])
    classes = ['background', 'cut-thought hole', 'semi-broken hole', 'window']
    return cmap, classes


def draw_pie_from_dict(dictionary, **kwargs):
    keys = list(dictionary.keys())
    values = list(dictionary.values())
    plt.pie(values, labels=keys, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')

    if 'title' in kwargs:
        plt.title(kwargs['title'])

    plt.show()


def Convert_Mask2SegClass(mask_path, label_path, plot=True):
    """
    将掩码图转换为灰度图
    :param mask_path: 掩码图文件夹路径
    :param label_path: label文件夹路径
    :param plot: 是否画图 Default: True
    :return: None
    """
    if not os.path.exists(label_path):
        os.makedirs(label_path)

    # 获取colormap
    cmap, classes = Get_colormap()

    # 获取统计像素类型信息字典
    pixel_statistics = {name: 0 for name in classes}

    for mask in tqdm(os.listdir(mask_path)):
        img = Image.open(os.path.join(mask_path, mask))
        img = np.array(img)
        H, W = img.shape[0:2]
        label = np.zeros(shape=(H, W), dtype=int)
        for i in range(len(cmap)):
            # 判断与colormap中匹配的像素
            match_pixel = img == cmap[i]

            if len(img.shape) > 2:
                match_pixel = match_pixel.all(axis=-1)

            label[match_pixel] = i

            # 统计当前颜色像素信息
            pixel_statistics[classes[i]] += np.sum(match_pixel)

        # 保存当前灰度图
        label = Image.fromarray(label)
        label.save(os.path.join(label_path, mask))

    # 输出统计结果
    print('-' * 55)
    print("| %15s | %15s | %15s |" % ("class", "class_num", "pixels count"))
    print('-' * 55)
    for i, (class_name, num) in enumerate(pixel_statistics.items()):
        print("| %15s | %15s | %15s |" % (class_name, i, str(num)))
        print('-' * 55)

    # 画图
    draw_pie_from_dict(dictionary=pixel_statistics, title='Pixel Statistics For Each Segmentation Class')


if __name__ == '__main__':
    mask_path = r"D:\python_project\data\WindowsInHailDisaster_photoes\Mask"
    label_path = r"D:\python_project\data\WindowsInHailDisaster_photoes\SegmentationClass"
    Convert_Mask2SegClass(mask_path, label_path)
