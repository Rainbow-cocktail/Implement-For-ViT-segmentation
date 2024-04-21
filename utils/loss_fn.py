import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    计算dice-loss
    """

    def __init__(self, num_classes=1000, ignore_index=255, softmax=True):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index  # 需要忽略的标签索引，默认为 255, 因为图片padding操作和可能的白边为255
        self.softmax = softmax

    def _one_hot_encoder(self, input_tensor):
        """transfer the target tensor (B,H,W) into one-hot label"""
        tensor_list = []
        for i in range(self.num_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)  # (B,num_class,H,W)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        """
            计算 Dice Loss. Dice loss = 1 - dice
            输入为 (B,num_class,H,W) 中 num_class维度的切片 (B,H,W)

            Args:
                score (torch.Tensor): 模型的预测结果，形状为 (B, H, W)
                target (torch.Tensor): 真实的分割标签，形状也为 (B, H, W)。

            Returns:
                torch.Tensor: 计算得到的Dice Loss值。
        """
        target = target.float()
        smooth = 1e-5
        intersection = torch.sum(score * target)  # (B,H,W) * (B,H,W)
        X_sum = torch.sum(score * score)
        Y_sum = torch.sum(target * target)
        dice = (2. * intersection + smooth) / (X_sum + Y_sum + smooth)
        return 1 - dice

    def forward(self, inputs, targets, weight=None):
        """
        计算模型预测结果和真实标签之间的 Dice Loss。

        Args:
        inputs (torch.Tensor): 模型的预测结果，形状为 (batch_size, num_classes, height, width)。
        targets (torch.Tensor): 真实的分割标签，形状为 (batch_size, height, width)。
        weight (list, optional): 每个类别的损失权重，默认为每个类别权重都为1的列表。如果未提供，则默认权重相等。

        Returns:
        tuple: 包含两个元素的元组，第一个元素为计算得到的平均 Dice Loss 值，第二个元素为每个类别的 Dice 系数列表。
            - torch.Tensor: 计算得到的平均 Dice Loss 值。
            - list: 每个类别的 Dice 系数列表，长度为 num_classes。
        """
        if self.softmax:
            inputs = torch.softmax(inputs, dim=1)  # in channel dim
        assert isinstance(targets, torch.Tensor), "targets should be a torch.Tensor"
        assert len(targets.size()) == 3, "Target must be a 3D tensor (batch_size, height, width)"

        # 如果目标值中存在需要忽略的像素，则将其掩盖掉
        mask = targets != self.ignore_index
        targets = targets * mask

        targets = self._one_hot_encoder(targets)
        if weight is None:
            # 设置默认权重为每个类别的损失贡献都相同，权重列表初始化为全1数组。
            weight = [1] * self.num_classes
        assert inputs.size() == targets.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                   targets.size())
        class_wise_dice = []  # 用于记录dice做评判，无实际作用
        loss = 0.0
        for i in range(self.num_classes):
            dice = self._dice_loss(inputs[:, i, :, :], targets[:, i, :, :])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.num_classes, class_wise_dice


class Criterion(nn.Module):
    """综合计算交叉熵损失和dice loss的损失"""

    def __init__(self, cross_entropy_weight=0.5, dice_loss_weight=0.5, num_classes=1000, ignore_index=255,
                 softmax=True):
        """
        初始化 Criterion 类。

        Args:
            cross_entropy_weight (float, optional): 交叉熵损失的权重，默认为 0.5。
            dice_loss_weight (float, optional): Dice Loss 的权重，默认为 0.5。
            num_classes (int, optional): 类别数，默认为 1000。
            ignore_index (int, optional): 需要忽略的标签索引，默认为 255。
        Note:
            忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        """
        super(Criterion, self).__init__()
        super(Criterion, self).__init__()
        self.cross_entropy_weight = cross_entropy_weight
        self.dice_loss_weight = dice_loss_weight
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.softmax = softmax

        # 创建交叉熵损失函数
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

        # 创建 Dice Loss 对象
        self.dice_loss = DiceLoss(num_classes=num_classes, ignore_index=ignore_index, softmax=self.softmax)

    def forward(self, outputs, labels):
        """
        计算综合的损失函数，结合了交叉熵损失和 Dice Loss。

        Args:
            outputs (torch.Tensor): 模型的输出，形状为 (batch_size, num_classes, height, width)。
            labels (torch.Tensor): 真实的分割标签，形状为 (batch_size, height, width)。

        Returns:
            torch.Tensor: 计算得到的综合损失。
        """
        # 计算交叉熵损失
        loss_ce = self.ce_loss(outputs, labels[:].long())

        # 计算 Dice Loss
        loss_dice, _ = self.dice_loss(outputs, labels)

        # 综合两种损失
        loss = self.cross_entropy_weight * loss_ce + self.dice_loss_weight * loss_dice

        return loss


if __name__ == '__main__':
    test = 0
    if test:
        LOSS = DiceLoss(num_classes=3)
        x = torch.rand(1, 3, 224, 224)  # num_classes = channel dim = 3
        target = torch.randint(low=0, high=3, size=(1, 224, 224), dtype=torch.int)  # 0 1 2
        loss_, dice_list = LOSS(x, target)
        print(loss_, dice_list)
    else:
        LOSS = Criterion(num_classes=3)
        x = torch.rand(1, 3, 224, 224)  # num_classes = channel dim = 3
        target = torch.randint(low=0, high=3, size=(1, 224, 224), dtype=torch.int)  # 0 1 2
        print(LOSS(x, target))
