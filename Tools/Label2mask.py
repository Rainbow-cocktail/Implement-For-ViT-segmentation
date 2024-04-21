"""
1. 对灰度图片像素种类进行计数
2. 对单张灰度图片进行计数并且无需cmap便可还原出伪RGB图像
note:
只需修改 "image_path"
image_path可以接受单张图片路径(推荐 ) 也可以接受图片文件夹路径
在接受图片文件夹路径时，需要输入文件夹底下图片通道数,灰度图像为1,彩色图像为3
建议只对单通道图片进行处理，对普通RGB处理很可能会炸掉
"""

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm


def Grey2RGB(img):
    RGB_img = plt.cm.tab20(img)
    return RGB_img


def count_unique_colors(image_path, c=None):
    if not os.path.exists(image_path):
        return print("targeted images not exist")

    if os.path.isfile(image_path):
        """对单张图片操作"""
        img = np.array(Image.open(image_path))  # (H,W,C)

        # 生成伪RGB图
        RGB_img = Grey2RGB(img)

        plt.imshow(RGB_img)
        plt.show()

        if c == 1:
            h, w = img.shape[0], img.shape[1]
            img = img.reshape(h, w, 1)

        # 合并高和宽 每一行表示一个像素的 RGB 值 (HxW,C)
        unique_colors = np.unique(img.reshape(-1, img.shape[2]), axis=0)
        return unique_colors

    elif os.path.isdir(image_path):
        """对一个文件夹下面图片进行操作"""
        img_list = os.listdir(image_path)
        c = 1
        combined = np.zeros(shape=(1, c))
        for each in tqdm(img_list):
            path = os.path.join(image_path, each)

            # 进行迭代
            single = count_unique_colors(path, c)

            # 将以往的和现在获取到的颜色表concat，再根据行去重
            combined = np.unique(np.concatenate((combined, single)), axis=0)

        return combined

    else:
        print('something went wrong')


if __name__ == "__main__":
    # 修改这个路径即可
    img_path = r'D:\python_project\data\WindowsInHailDisaster_photos\SegmentationClass\0.png'
    unique_colors = count_unique_colors(img_path, c=1)

    # 打印相关信息
    print(f"Total unique colors: {len(unique_colors)}")
    if len(unique_colors[0]) == 3:
        """对mask-RGB图像进行颜色种类查看"""
        for color in unique_colors:
            if np.sum(color != 0):
                print(f'RGB:{color}')
            else:
                print(f"RGB: {color} black background")
    if len(unique_colors[0]) == 1:
        """对标签进行查看"""
        print("index : class")
        for i in unique_colors:
            if i == 0:
                print(f'{i} : backgound')
            else:
                print(f'{i} : class {i}')
