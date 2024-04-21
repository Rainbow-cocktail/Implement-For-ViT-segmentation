import torch
from torch import nn
from tensorboardX import SummaryWriter
from utils.distributed_utils import ConfusionMatrix, DiceCoefficient


def train_one_epoch(train_loader, model, criterion, optimizer, device, lr_scheduler, epoch, iteration_num, writer):
    model.train()
    epoch_loss = 0.0
    num_batches = len(train_loader)

    for batch_idx, (data, target) in enumerate(train_loader):
        images, labels = data.to(device), target.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        iteration_num += 1

        # 可视化信息
        if iteration_num % 20 == 0:
            img = images[0, 0:1, :, :]
            img = (img - img.min()) / (img.max() - img.min())
            writer.add_image('train/Image', img, iteration_num)
            output = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
            writer.add_image('train/Prediction', output[0, ...] * 50, iteration_num)
            labs = labels[0, ...].unsqueeze(0) * 50
            writer.add_image('train/GroundTruth', labs, iteration_num)

    if lr_scheduler is not None:
        lr_scheduler.step()

    lr = optimizer.param_groups[0]["lr"]

    return epoch_loss / num_batches, iteration_num, lr


def evaluate(val_loader, model, device, num_classes):
    """返回混淆矩阵和dice分数"""
    model.eval()
    confusion_matrix = ConfusionMatrix(num_classes)
    dice = DiceCoefficient(num_classes=num_classes, ignore_index=255)
    with torch.no_grad():
        for idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            output = torch.argmax(torch.softmax(outputs, dim=1), dim=1)  # B,CLASS,H,W --> B,C,H

            confusion_matrix.update(target.flatten(), output.flatten())
            dice.update(y_pred=outputs, y_true=target)

        confusion_matrix.reduce_from_all_processes()
        dice.reduce_from_all_processes()

    return confusion_matrix, dice.value


if __name__ == '__main__':
    dice = DiceCoefficient(num_classes=4)
    dice = DiceCoefficient(num_classes=4)
    dice.update(torch.rand(size=(1, 4, 4)), torch.rand(size=(1, 4, 4, 4)))
    print(dice.cumulative_dice.requires_grad)
