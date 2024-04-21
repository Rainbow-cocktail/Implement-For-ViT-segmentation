import argparse
import copy

import torch.nn as nn
import torch
import os
from nets.swin_transformer_unet import SwinTransformer


class My_model(nn.Module):
    def __init__(self, num_classes=1000, img_size=224):
        super(My_model, self).__init__()
        self.model = SwinTransformer(img_size=img_size, num_classes=num_classes)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.model(x)
        return logits

    def load_from(self, args):
        """
        如果设置了resume,则当前训练不会加载预训练权重, pretrained参数失效, 模型从 checkpoint 开始训练
        如果设置了pretrained文件路径, 则会加载pretrain模型,否则模型从零开始训练
        """
        if args.resume:
            return print("Resuming from checkpoint, setting 'args.pretrained' will be invalidated".format(args))

        if args.pretrained is not None:
            if not os.path.isfile(args.pretrained):
                raise FileNotFoundError("pretrained file {} not found.".format(args.pretrained))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(args.pretrained, map_location=device)

            # FIXME:如果 "model"键没有在 .pth 文件中,则需要进行特殊处理,但一般用不到
            if "model" not in pretrained_dict:
                print("---start load pretrained model by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key {}".format(k))
                        del pretrained_dict[k]
                msg = self.model.load_state_dict(pretrained_dict, strict=False)
                return

            print("---start load pretrained model of swin encoder---")
            pretrained_dict = pretrained_dict["model"]  # 从.pth文件中读取的模型的状态字典 (一般只含有encoder)
            model_dict = self.model.state_dict()  # 待载入模型的状态字典
            full_dict = copy.deepcopy(pretrained_dict)  # 准备储存encoder和decoder所有参数

            # 将encoder各层参数对称到decoder中
            for k, v in pretrained_dict.items():
                if "layers" in k:
                    current_layer_num = 3 - int(k[7:8])  # layers.0/1/2/3/.blocks......
                    current_k = "layers_up." + str(current_layer_num) + k[8:]  # layers_up.3/2/1/0.blocks......
                    full_dict.update({current_k: v})

            for k in list(full_dict.keys()):
                # "model."是self.model状态字典的前缀名称
                k_new = 'model.' + k
                if k_new in model_dict:
                    if full_dict[k].shape != model_dict[k_new].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, full_dict[k].shape,
                                                                                  model_dict[k_new].shape))
                        del full_dict[k]
                    else:
                        model_dict[k_new] = full_dict[k][:]
            msg = self.model.load_state_dict(model_dict, strict=False)
            print("Pretrained weights has successfully loaded from {}".format(args.pretrained))
        else:
            print("MODEL will training from scratch")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.pretrained = "D:\python_project\My_ViT_SegNet\pretrained\swin_tiny_patch4_window7_224.pth"
    args.resume = False
    model = My_model()
    model.load_from(args)
    print(model.state_dict().keys())

