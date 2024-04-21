import os
import argparse

import torch
from matplotlib import pyplot as plt
from torchvision import transforms
import numpy as np
from PIL import Image
from nets import My_model as Swin_unet


def main(save=False):
    """对单张图片进行推理, save表示是否保存推理的图片"""
    # =============================参数设置=============================================
    num_classes = 20  # 不包括背景
    weight_path = r'D:\python_project\My_ViT_SegNet\weights\exp6\best_model.pth'  # 权重文件
    img_path = r'D:\python_project\data\VOCtrainval2007\JPEGImages\000032.jpg'  # 推理的图片
    cmap = np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                     [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                     [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                     [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                     [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                     [0, 64, 128]])  # 1 + 20 包括背景的cmap
    classes = ["_background_", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
               "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    # ===================================================================================
    assert os.path.exists(weight_path), f"weights {weight_path} not found."
    assert os.path.exists(img_path), f"images {img_path} not found."
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Swin_unet(num_classes=num_classes + 1, img_size=224)
    model.load_state_dict(torch.load(weight_path, map_location='cpu')['model'])
    model.to(device)

    # load img
    img = Image.open(img_path).convert('RGB')

    # transform
    data_transforms = transforms.Compose([transforms.ToTensor(),
                                          transforms.Resize(size=(224, 224)),
                                          transforms.Normalize(mean, std)])
    img_transform = data_transforms(img).unsqueeze(0).to(device)

    # inference
    with torch.no_grad():
        # init model
        img_height, img_width = img_transform.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        output = model(img_transform)
        prediction = output.argmax(dim=1).squeeze(0)  # 224,224
        prediction = prediction.to('cpu').numpy().astype(np.uint8)
        prediction = gray_to_rgb(prediction, cmap)

        # Plot
        n = 2
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(img)
        axs[0].set_title('original image')
        axs[0].axis('off')

        axs[1].imshow(prediction)
        axs[1].set_title('predict image')
        axs[1].axis('off')
        plt.tight_layout()
        plt.show()

        if save:
            prediction = Image.fromarray(prediction)
            prediction.save('test_result.png')


def gray_to_rgb(gray_label, cmap):
    """将灰度图转换为伪 RGB 图像"""
    rgb_label = np.zeros((gray_label.shape[0], gray_label.shape[1], 3), dtype=np.uint8)
    for i in range(len(cmap)):
        rgb_label[gray_label == i] = cmap[i]
    return rgb_label


if __name__ == '__main__':
    main()
