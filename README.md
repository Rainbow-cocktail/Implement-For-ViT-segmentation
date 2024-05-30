# Swin-Unet for segmentation

特别感谢 ：

[bubbliiiing/unet-pytorch at bilibili (github.com)](https://github.com/bubbliiiing/unet-pytorch/tree/bilibili)

[WZMIAOMIAO/deep-learning-for-image-processing: deep learning for image processing including classification and object-detection etc. (github.com)](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing)

[HuCaoFighting/Swin-Unet: [ECCVW 2022\] The codes for the work "Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation" (github.com)](https://github.com/HuCaoFighting/Swin-Unet)

[microsoft/Swin-Transformer: This is an official implementation for "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows". (github.com)](https://github.com/microsoft/Swin-Transformer)

此仓库的代码是根据以上大佬的代码而来

## 数据准备

```python
$ tree data
Dataset
├── ImageSets
│   └── Segmentation ： 存放训练和验证的图片名称数据
│      		├── test.txt
│      		├── train.txt
│      		└── val.txt
├── JPEGImages ： 存放原始的RGB图片
│   ├── img1.jpg
│   ├── img2.jpg
│   ├── img3.jpg
│   └── ...
├── Mask ： 存放对应图片的RGB掩码图
│   ├── img1.png
│   ├── img2.png
│   ├── img3.png
│   └── ...
└── SegmentationClass ： 存放对应图片的8位灰度图 mode= P or I or L
    ├── img1.png
    ├── img2.png
    └── ...
```

本仓库的数据格式为 VOC数据格式，请用`Tools` 文件夹下面的工具处理与VOC数据格式不相符的数据

voc数据集简介，voc的标签文件读进来为`mode=P`，
直接为单通道的图片，白色的边框对应255，0代表背景，
但它显示为彩色是因为，运用了调色板，调色板相当于内置的RGB，即
比如数字0，反应到调色板上为[0,0,0]黑色,数字1为，[128,0,0]红色，
所以产生了是RGB图像的效果，局限性为只能用它内置的调色板里面的颜色，不能自己创造新颜色。
正是因为单通道产生了三通道的效果，所以更省内存，运行代码 

```python
img = img.convert('RGB')  # 将MODE=P转化为RGB的图
```



## Tools

存放处理原始数据的工具

## logs

```python
$ tree data
logs 
├── exp0
│   └── events.out
├── exp1
│ 	└── events.out
├── exp2 
│ 	└── events.out
└── ...

```

存放每次训练的日志文件，可用tensorboard打开

## results 

存放每次训练的结果，为`.txt`文件，命名同日志文件

## weights

存放每次训练的模型文件，为`.pth` , 命名同日志文件

## pretrained

存放预训练模型，分别来自 imageNet-1k 和 imageNet-22k 

## 主体代码文件

- utils
- nets ： 模型代码
- train.py :  训练代码

## 使用教程

1. 确保 图片为 `.jpg`  掩码图或者标签图为 `.png` , 如果不满足要求则使用 `Tools`下面的 `png2jpgimage.py`将图片的后缀全部转化为 `.jpg`。 图片放在`JPEGImages` 文件夹下

2. 使用 `Convert_Mask2SegClass.py`   将掩码图转化为真正的用于训练的单通道灰度图，并将这些图片放置在 `SegmentationClass` 文件夹下

3. 运行`train_val_split.py`划分训练集和验证集，划分好的为 `.txt`文件，它们会被保存在 `ImageSets-Segmentation`文件夹下，你可以进入文本文件并手动修改他们，如果需要的话。

4. 模型训练，打开`train.py`,可查看修改的超参数

   ```cmd
   python train.py --data Dataset_path 
   ```

   必须设置的参数

   - num_classes: 一定要设置，为分割的种类（不包括背景）
   - data : 数据集路径 
   - epochs：训练轮次

5. (可选) 打开tensorboard查看本次运行的结果
   ```cmd
   tensorboard --logdir path
   ```

6.  推理，打开 `predict.py`， 直接在程序修改参数一览替换为自己的参数，运行该程序即可



## 待优化
- 目前该仓库仅自用！！！！！因为结果现在是一坨，现在正在优化中
- 注重需要修改 num class 和 crop size  还有模型的img size部分，这部分的配置现在内嵌于里代码里面，还没有提出来，因此需要config文件统一写
- 修改lr rate等策略到配置文件
- yaml配置文件，避免 random crop  colormap等需要进入具体文件一个一个调整
- create model 部分需要进行优化，模型的img_size 是根据crop_size决定的
- 损失函数进行调整 查看是否两个损失接近
- 模型精度较差
