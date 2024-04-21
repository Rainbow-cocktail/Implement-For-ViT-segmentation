"""
划分训练验证和测试集
划分好的集合会生成txt文件，保存在下 \ImageSets\Segmentation
"""

import os
import random

from tqdm import tqdm


def train_val_split(root, val_split=0.05, test_split=0.05, shuffle=False):
    """Split training , validation and test dataset, and save as 'txt' file
    :arg
        root: 根目录, 目录下面就是图片
        val_split: 验证集比例
        test_split: 训练集比例
        shuffle: 是否重新划分训练集和验证集
    """
    print("Generate txt in ImageSets.")
    img_path = os.path.join(root, "SegmentationClass")
    txt_path = os.path.join(root, r"ImageSets\Segmentation")

    img_seg = []
    for i in os.listdir(img_path):
        if i.endswith('.png'):
            img_seg.append(i)
        elif i.endswith('.jpg'):
            raise ValueError(f"{i}Not a .png file, Please find mask or label file .png format")

    print("一共找到{}张符合要求的图片".format(len(img_seg)))

    num_imgs_val = int(len(img_seg) * val_split)
    num_imgs_test = int(len(img_seg) * test_split)

    random.shuffle(img_seg)

    # 去除后缀
    img_seg = [picture.split('.')[0] for picture in img_seg]

    # 训练、验证、测试集图像名字
    img2val = img_seg[:num_imgs_val]
    img2test = img_seg[num_imgs_val:(num_imgs_val + num_imgs_test)]
    img2train = img_seg[(num_imgs_val + num_imgs_test):]
    img2train_val = img2train + img2val
    print("training set: {}".format(len(img2train)))
    print("val set: {}".format(len(img2val)))
    print("train&val set: {}".format(len(img2train_val)))
    print("test set: {}".format(len(img2test)))

    if (os.path.exists(os.path.join(txt_path, "train.txt")) and os.path.exists(os.path.join(txt_path, "val.txt"))
            and os.path.exists(os.path.join(txt_path, "test.txt")) and not shuffle):
        print("使用文件夹中已经分好的训练集和验证集")
        print("若想要重新划分，请删掉{}文件夹中文件,或者修改 shuffle 参数".format(txt_path))
    else:
        # 写入txt文件
        print("将新分的训练集和验证集写入“{}”文件夹中".format(txt_path))
        with open(os.path.join(txt_path, "train.txt"), 'w') as f:
            for name in tqdm(img2train):
                f.write(name+'\n')

        with open(os.path.join(txt_path, "test.txt"), 'w') as f:
            for name in tqdm(img2test):
                f.write(name+'\n')

        with open(os.path.join(txt_path, "val.txt"), 'w') as f:
            for name in tqdm(img2val):
                f.write(name+'\n')

        with open(os.path.join(txt_path, "trainval.txt"), 'w') as f:
            for name in tqdm(img2train_val):
                f.write(name+'\n')

    return img2train, img2val, img2test, img2train_val

if __name__ == "__main__":
    root = r"../../data/VOCtrainval2007"

    a,b,c,d= train_val_split(root,val_split=0.0,test_split=0.1,shuffle=True)
    print("划分完成！")
