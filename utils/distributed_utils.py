import torch
import torch.distributed as dist
from utils.loss_fn import DiceLoss
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, label, prediction):
        """
        根据label和prediction 在原有的矩阵上，更新混淆矩阵，即将新统计的像素加入进来

        :param label: Ground True  (B,C,H)
        :param prediction: prediction label (B,C,H)
        """
        n = self.num_classes
        if self.mat is None:
            # 创建混淆矩阵
            self.mat = torch.zeros(size=(n, n), dtype=torch.int64, device=label.device)
        with torch.no_grad():
            # 指示哪些像素属于有效的类别
            k = (label >= 0) & (label < n)
            # 统计像素真实类别a[k]被预测成类别b[k]的个数(这里的做法很巧妙,跟swin-mask一样，要动手实操才明白)
            inds = n * label[k].to(torch.int64) + prediction[k]  # inds为一维向量
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)  # 之前统计的 + 当前统计的混淆矩阵数量

    def reset(self):
        if self.mat is None:
            self.mat.zero_()

    def compute(self):
        """计算混淆矩阵中全局准确率、类别准确率、IOU"""
        h = self.mat.float()
        # 计算全局预测准确率(混淆矩阵的对角线为预测正确的个数)
        acc_global = torch.diag(h).sum() / h.sum()
        # 计算每个类别的准确率, 1e-6 是为了避免分母除0
        acc = torch.diag(h) / (torch.sum(h, dim=1) + 1e-6)
        # 计算每个类别预测与真实目标的iou, iou = (A ∩ B) / (A U B)
        iou = torch.diag(h) / ((h.sum(1) + h.sum(0) - torch.diag(h)) + 1e-6)
        return acc_global, acc, iou

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iou = self.compute()
        return (
            '\nglobal correct: {:.1f} %\n'
            'average row correct: {} %\n'
            'IoU: {} %\n'
            'mean IoU: {:.1f} %').format(
            acc_global.item() * 100,
            ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
            ['{:.1f}'.format(i) for i in (iou * 100).tolist()],
            iou.mean().item() * 100)


class DiceCoefficient(object):
    def __init__(self, num_classes: int = 2, ignore_index: int = -100):
        self.cumulative_dice = None
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.count = None

    def update(self, y_true, y_pred):
        """
        更新 dice 值 (累加)
        这里的dice值指每一类的dice求和得到的总dice
        :param y_true: B,H,W
        :param y_pred: B,CLASS,H,W
        """

        if self.cumulative_dice is None:
            self.cumulative_dice = torch.zeros(1, dtype=y_pred.dtype, device=y_pred.device)
        if self.count is None:
            self.count = torch.zeros(1, dtype=y_pred.dtype, device=y_pred.device)
        dice_ = DiceLoss(num_classes=self.num_classes, ignore_index=self.ignore_index)
        _, class_wise_dice = dice_(y_pred, y_true)
        self.cumulative_dice += torch.sum(torch.Tensor(class_wise_dice))
        self.count += 1

    @property
    def value(self):
        """:return: dice score for per Batch size"""
        if self.count == 0:
            return 0
        else:
            return (self.cumulative_dice / self.count).item()

    def reset(self):
        if self.cumulative_dice is not None:
            self.cumulative_dice.zero_()

        if self.count is not None:
            self.count.zeros_()

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.cumulative_dice)
        torch.distributed.all_reduce(self.count)


def plot_confusion_matrix(conf_matrix, classes_dict, save_path=None, epoch=None):
    """
    画混淆矩阵图保存到tensorboard中
    conf_matrix (numpy.ndarray): Confusion matrix to be visualized.
    classes_dict (dict or int) : Dictionary mapping class indices to class labels,
                                    {"class_0" : "background", "class_1" : ......}
                                    or an integer indicating the number of classes.
                                    If dict, class labels will be used from the dictionary.
                                    If int, default class labels will be used (e.g., "Class 0", "Class 1", ...).
    save_path (str, optional): Path to save the plot. If None, the plot is not saved.
    epoch (int, optional): Epoch number
    """
    fig = plt.figure(figsize=(10, 10))
    sns.set(font_scale=1.2)
    if isinstance(classes_dict, dict):
        x_tick_labels = [i for i in classes_dict.values()]
        y_tick_labels = [i for i in classes_dict.values()]
    elif isinstance(classes_dict, int):
        x_tick_labels = [f"Class {i}" for i in range(classes_dict)]
        y_tick_labels = [f"Class {i}" for i in range(classes_dict)]
    else:
        raise ValueError("Invalid classes_dict type. Must be dict or int.")

    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=x_tick_labels,
                yticklabels=y_tick_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix {epoch+1}' if epoch else "Confusion Matrix")
    if save_path:
        plt.savefig(save_path,bbox_inches='tight')

    return fig


if __name__ == '__main__':
    dice = DiceCoefficient(num_classes=4)
    dice.update(torch.rand(size=(1, 4, 4)), torch.rand(size=(1, 4, 4, 4)))
    print(dice.cumulative_dice.requires_grad)
