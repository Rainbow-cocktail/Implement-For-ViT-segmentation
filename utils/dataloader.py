"""
Dataset
本身自带了transform,如果有自己的想法可以自己自定义transform并且传入datsaet中，下面已经给出了格式
标签中，0代表背景，1~n代表类，255代表padding或者白边（voc数据集图像有分割的物体有边界线）

FIXME crop_size 等等 提到config中，不然每次要到源码里面改很麻烦
FIXME 去白边
修改下面 以适应不同的图像 可选： 224 448
self.crop_size
self.resize
"""
import os
import random
import warnings

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as F


# 可自定义transform
class MyTransform(object):
    """自定义transform"""

    def __init__(self):
        pass

    def __call__(self, img, label):
        return img, label


class SegmentationDataset(Dataset):
    def __init__(self, data_dir: str, num_classes: int, train: bool, img_size=224, transform=None, train_val=True):
        """
        语言分割数据集，其中自带了图片的transform处理
        :param data_dir: 数据集根目录
        :param num_classes: 分割类别数(包括背景)
        :param train: 是否是训练集
        :param transform: 使用自定义的transform
        :param train_val: 是否用train和val两者的集合用做训练集
        """
        super(SegmentationDataset, self).__init__()

        self.data_dir = data_dir
        self.num_classes = num_classes
        self.train = train
        self.img_size = img_size

        # 图像变化与增强
        self.transform = transform  # 自定义transform

        # 读取训练集和验证集
        if self.train:
            train_txt_path = os.path.join(self.data_dir,
                                          'ImageSets/Segmentation/trainval.txt') if train_val else os.path.join(
                self.data_dir, 'ImageSets/Segmentation/train.txt')
            with open(train_txt_path, 'r') as f:
                self.txt_content = f.readlines()
        else:
            val_txt_path = os.path.join(self.data_dir, 'ImageSets/Segmentation/val.txt')
            with open(val_txt_path, 'r') as f:
                self.txt_content = f.readlines()
        self.txt_content = [line.strip() for line in self.txt_content]  # 去掉换行符

    def __len__(self):
        return len(self.txt_content)

    def __getitem__(self, idx):
        """与模型尺寸相适配"""
        prefix_name = self.txt_content[idx]
        img_jpg = Image.open(os.path.join(self.data_dir, 'JPEGImages', '{}.jpg'.format(prefix_name))).convert('RGB')  # 图像-RGB图
        label_png = Image.open(
            os.path.join(self.data_dir, 'SegmentationClass', '{}.png'.format(prefix_name)))  # 标签-mode=I、L、P图
        if self.transform is not None:
            img_jpg, label_png = self.transform(img_jpg, label_png)
        else:
            # FIXME
            # train
            self.base_size = 512  # 放大缩小基准尺寸
            self.crop_size = self.img_size  # 随机裁剪尺寸,最后送入模型的尺寸
            # val
            self.resize = self.img_size  # 最后送入模型的尺寸
            if self.train:
                img_jpg, label_png = self.train_transforms(img_jpg, label_png, self.base_size, self.crop_size)
            else:
                img_jpg, label_png = self.val_transforms(img_jpg, label_png, self.resize)

        return img_jpg, label_png

    @staticmethod
    def train_transforms(img, label, base_size=224, crop_size=224):
        """
        img和label需要做相同的变化,最好不要直接采用transform,而是自定义transform函数操作
        流程：
        random resize , default 512x0.8~512x1.2
        random horizontal flip
        random vertical flip
        random crop, default 400x400
        ToTensor to tensor [0,1]
        normalize to tensor [-1,1]
        由于random crop保证了图片大小的一致性，因此无需resize
        """
        hflip_pro = 0.5
        vflip_pro = 0.5
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_list = [RandomResize(base_size)]
        if hflip_pro > 0:
            transform_list.append(RandomHorizontalFlip(hflip_pro))
        if vflip_pro > 0:
            transform_list.append(RandomVerticalFlip(vflip_pro))
        transform_list.append(RandomCrop(size=crop_size))
        transform_list.append(ToTensor())
        transform_list.append(Normalize(mean=mean, std=std))
        train_transform = Compose(transform_list)

        return train_transform(img, label)

    @staticmethod
    def val_transforms(img, label, resize=224):
        """
        validation数据的transform, label无需normalization
        Note:
            val_transform需要与 train 大小一致
        """
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        val_transform = Compose([
            Resize(resize),
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])
        return val_transform(img, label)


class Compose(object):
    """自定义transform.compose,需要同时传入两个参数 """

    def __init__(self, transforms_list):
        self.transforms = transforms_list

    def __call__(self, img, label):
        for transform in self.transforms:
            img, label = transform(img, label)
        return img, label


class RandomResize(object):
    """
    随机放大或者缩小图片, 标签采用近邻插值进行放大或者缩小
    """

    def __init__(self, base_size, max_size=None, min_size=None):
        self.max_size = max_size

        if max_size is None:
            self.max_size = int(1.2 * base_size)
        else:
            assert max_size >= 1.0, f"{self.__class__.__name__}:max_size must be larger than 1.0"
            self.max_size = max_size

        if min_size is None:
            self.min_size = int(0.9 * base_size)
        else:
            assert min_size <= 1.0, f"{self.__class__.__name__}:min_size must be smaller than 1.0"
            self.min_size = min_size

    def __call__(self, img, label):
        size = random.randint(self.min_size, self.max_size)
        img = F.resize(img, size)  # 最短边resize成"size",另一边等比放大缩小
        label = F.resize(label, size, interpolation=transforms.InterpolationMode.NEAREST)  # 不能是线性插值,必须是近邻插值
        return img, label


class RandomCrop(object):
    """
    随机裁剪正方形图片
    如果图片或者标签尺寸小于size,则会自动padding
    Default : crop_size =  512 x 512
    """

    def __init__(self, size=512):
        self.size = size

    def __call__(self, img, label):
        # For PIL image, img size is (W,H,C) not (C,H,W)
        w, h = img.size
        if h < self.size or w < self.size:
            warnings.warn(
                "Required crop size {} is larger than Random_Resized image size {}. Image and label will be padded".format(
                    (self.size, self.size), (w, h)))
            img = padding_if_smaller(img, self.size)  # 0填充img
            label = padding_if_smaller(label, self.size, fill=255)  # 255填充label,计算loss时会mask掉
        rect = transforms.RandomCrop.get_params(img, (self.size, self.size))
        img = F.crop(img, *rect)
        label = F.crop(label, *rect)
        return img, label


def padding_if_smaller(img, size, fill=0):
    # For PIL image, img size is (W,H,C) not (C,H,W)
    w, h = img.size
    padding_left = (size - w) // 2 if w < size else 0
    padding_right = size - w - padding_left if w < size else 0
    padding_top = (size - h) // 2 if h < size else 0
    padding_bottom = size - h - padding_top if h < size else 0
    PaddingImage = F.pad(img, [padding_left, padding_top, padding_right, padding_bottom], fill=fill)
    return PaddingImage


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, img, label):
        # 生成随机0-1浮点数
        if random.random() < self.prob:
            img = F.hflip(img)
            label = F.hflip(label)
        return img, label


class RandomVerticalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, img, label):
        if random.random() < self.prob:
            img = F.vflip(img)
            label = F.vflip(label)
        return img, label


class ToTensor(object):
    """
    将图片和标签转为tensor,同时对图片做归一化
    note:
        transforms.ToTensor会自动做归一化
    """

    def __call__(self, img, label):
        img = F.to_tensor(img)
        label = torch.tensor(np.array(label), dtype=torch.int64)
        return img, label


class Normalize(object):
    """只对图片做normalize"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, label):
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img, label


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, label):
        img = F.center_crop(img, self.size)
        label = F.center_crop(label, self.size)
        return img, label


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, label):
        image = F.resize(img=image, size=(self.size, self.size))
        label = F.resize(img=label, size=(self.size, self.size), interpolation=transforms.InterpolationMode.NEAREST)
        return image, label


if __name__ == '__main__':
    """
    下面是两段测试代码，分别是测试是否正确transform和dataloader能正常取出
    """
    test_mode = 0
    if test_mode == 1:
        # check if transform correctly
        # 对于VOC的标签图像而言，[0,0,0]代表背景，其他1~20为20个类别。最后一个元素为目标边缘的白色线条
        # cmap对于模型是没有任何影响的，只是影响最后的可视化
        cmap = np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                         [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                         [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                         [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                         [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                         [0, 64, 128]])  # 1 + 20
        classes = ["_background_", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
                   "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
        cmap = np.append(cmap,[[255,255,255]], axis=0) # 21 + 1 最后一个元素为目标边缘的白色线条

        def gray_to_rgb(gray_label, cmap):
            """将灰度图转换为伪 RGB 图像"""
            rgb_label = np.zeros((gray_label.shape[0], gray_label.shape[1], 3), dtype=np.uint8)
            for i in range(len(cmap)):
                """
                这样写是为了把目标边缘的白色线条也显示出来,如果不写if语句，将下面的所有语句替换为
                “rgb_label[gray_label == i] = cmap[i]”
                则会自动把白色边缘默认为背景，
                因为我们初始化rgb图像的时候就是默认全黑初始的，
                然后再往全黑背景里面填彩色
                """
                if i == len(cmap) - 1:
                    rgb_label[gray_label == 255] = cmap[i]
                else:
                    rgb_label[gray_label == i] = cmap[i]
            return rgb_label


        n = 4
        fig, axs = plt.subplots(2, 4, figsize=(10, 5))
        for i in range(n):
            each_img = os.listdir(r"D:\python_project\data\VOCtrainval2007\SegmentationClass")[i]
            img = Image.open(
                os.path.join(r"D:\python_project\data\VOCtrainval2007\JPEGImages", each_img.split(".")[0] + '.jpg'))
            label = Image.open(
                os.path.join(r"D:\python_project\data\VOCtrainval2007\SegmentationClass",
                             each_img.split(".")[0] + ".png"))
            transform = Compose([
                RandomResize(300),  # 修改这个来让padding, 参考100 200 300
                RandomHorizontalFlip(0.2),  # 参考 0.5
                RandomVerticalFlip(0.2),  # 参考 0.5
                RandomCrop(size=224)
            ])
            out_img, out_label = transform(img, label)
            tensor_img = F.to_tensor(out_img)
            tensor_label = torch.tensor(np.array(out_label),dtype=torch.int64)
            numpy_label = tensor_label.numpy()
            axs[0, i].imshow(out_img)
            axs[0, i].set_title(f"RGB {i}")
            axs[0, i].axis('off')

            out_label_np = np.array(out_label)
            out_label_np = gray_to_rgb(out_label_np, cmap)

            # matplotlib中可以直接画np图像
            axs[1, i].imshow(out_label_np)
            axs[1, i].set_title(f"pseudo Label {i}")
            axs[1, i].axis('off')

        plt.tight_layout()
        plt.show()
    else:
        # check dataloader work correctly
        dataset = SegmentationDataset(data_dir=r"D:\python_project\data\VOCtrainval2007", num_classes=21, train=True)
        for idx in range(4):
            img, label = dataset[idx]
            img = img.numpy()
            label = label.numpy()
            print(img.shape)
            print(label.shape)
