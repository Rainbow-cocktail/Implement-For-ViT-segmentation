import argparse
import os

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from utils import SegmentationDataset, train_one_epoch, evaluate
from utils.loss_fn import DiceLoss, Criterion
from utils.distributed_utils import plot_confusion_matrix
from nets import My_model as Swin_unet

import time
from tensorboardX import SummaryWriter
from tqdm import tqdm
import datetime


def Get_training_GPU():
    """目前只支持单卡"""
    if torch.cuda.is_available():
        print("Training will be on ", torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print("Training will be on cpu")
        device = torch.device("cpu")
    print('--' * 10)
    return device


def main(args):
    device = Get_training_GPU()

    # 分类数+背景 = 总类
    num_classes = args.num_classes + 1

    # Dataset
    train_dataset = SegmentationDataset(data_dir=args.data, num_classes=num_classes, img_size=args.img_size, train=True,
                                        transform=None)
    val_dataset = SegmentationDataset(data_dir=args.data, num_classes=num_classes, img_size=args.img_size, train=False,
                                      transform=None)

    # DataLoader
    num_workers = min(os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=num_workers)

    data_iter = iter(train_loader)
    img, label = next(data_iter)

    print("img shape: ", img.shape)
    print("label shape: ", label.shape)
    print(type(label))
    print(len(train_loader))

    model = Swin_unet(num_classes=num_classes, img_size=args.img_size)
    model.to(device)
    model.load_from(args)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(params_to_optimize, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

    if args.loss_type == 'ce':
        criterion = nn.CrossEntropyLoss(ignore_index=255)
    elif args.loss_type == 'dice':
        criterion = DiceLoss(num_classes=num_classes, softmax=True)
    elif args.loss_type == 'mix':
        criterion = Criterion(num_classes=num_classes, softmax=True)
    else:
        raise ValueError("Invalid loss_type. Choose from 'ce', 'dice', 'mix'.")

    # 生成日志文件夹
    if args.exp_name == 'exp':
        existing_exp_names = [name for name in os.listdir("./logs") if name.startswith('exp')
                              if not ("checkpoint_epoch" in name)]
        if existing_exp_names:
            existing_exp_names = [int(name[3:]) for name in existing_exp_names]
            existing_exp_names.sort()
            for i in range(len(existing_exp_names)):
                if i != existing_exp_names[i]:
                    idx = i
                    break
                idx = i + 1
            exp_name = f'exp{str(idx)}'
        else:
            exp_name = 'exp0'
    else:
        exp_name = args.exp_name

    # resume training
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    start_epoch = 0
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError("Resume file not found. Please provide a valid file path.")
        checkpoint = torch.load(args.resume, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        exp_name = checkpoint['exp_name'] + f'_checkpoint_epoch_{checkpoint["epoch"]}'
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    # exp_name 为本次试验日志文件保存名称
    logs_dir = os.path.join(r'./logs', exp_name)
    writer = SummaryWriter(log_dir=logs_dir)

    iterator = tqdm(range(start_epoch, args.epochs))

    # Training
    start_time = time.time()
    best_dice = 0.0
    iteration_num = 0
    results_path = os.path.join('results', exp_name+'.txt')  # save .txt for training and val info
    save_weights_path = os.path.join("./weights", exp_name)  # save .pth for training weights
    os.makedirs(save_weights_path, exist_ok=True)
    for epoch in iterator:
        epoch_loss, iteration_num, learning_rate = train_one_epoch(train_loader=train_loader,
                                                                   model=model,
                                                                   criterion=criterion,
                                                                   optimizer=optimizer,
                                                                   device=device,
                                                                   lr_scheduler=lr_scheduler,
                                                                   epoch=epoch,
                                                                   iteration_num=iteration_num,
                                                                   writer=writer)
        confusion_matrix, dice_score = evaluate(val_loader=val_loader,
                                                model=model,
                                                device=device,
                                                num_classes=num_classes)

        writer.add_scalar('info/total_loss', epoch_loss, epoch)
        writer.add_scalar('info/dice_score', dice_score, epoch)
        writer.add_scalar('info/learning_rate', learning_rate, epoch)
        iterator.set_postfix(epoch=epoch, loss=epoch_loss, val_dice_score=dice_score)
        print(confusion_matrix)

        # confusion_matrix visualization
        fig = plot_confusion_matrix(conf_matrix=confusion_matrix.mat.cpu(), classes_dict=num_classes, save_path=None,
                                    epoch=epoch)
        writer.add_figure(f"Confusion Matrix For Epoch ", fig, epoch)

        # write into txt
        with open(results_path, "a") as f:
            training_info = f"==Epoch: {epoch}== \ntrain_loss: {epoch_loss:.4f} \nlr: {learning_rate:.6f}\n"
            val_info = f"val_dice: {dice_score:.3f} \nconfusion_matrix: {str(confusion_matrix)}\n\n"
            f.write(training_info)
            f.write(val_info)

        # save model
        save_file = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
            "exp_name": exp_name
        }
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if args.save_best:
            if best_dice < dice_score:
                best_dice = dice_score
                torch.save(save_file, os.path.join(save_weights_path, "best_model.pth"))
        else:
            torch.save(save_file, os.path.join(save_weights_path, "model_{}.path".format(epoch)))

    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print("Finished Training !!\ntraining time:  {} \n"
          "Your logs for tensorboard have save to {}\n"
          "Your training and validation results have save to {}\n".format(total_time_str, logs_dir, results_path))

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training for ViT-SegNet")

    parser.add_argument("--data", default="../data/VOCtrainval2007", help="Path to the Target-dataset")
    parser.add_argument("--img_size", default=224, type=int, help="Input image size for training (Not original img)")
    parser.add_argument("--num_classes", default=20, type=int, help="Num of Segmentation class Excluding background")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs to train")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch Size")
    parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help='momentum')
    parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float, help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--loss_type', type=str, default='mix', choices=['ce', 'dice', 'mix'],
                        help='Type of loss function to use (CrossEntropy/dice/mix)')
    parser.add_argument('--exp_name', type=str, default='exp', help='Experiment name for logs directory')
    parser.add_argument('--save_best', action='store_true', default=True, help='Save best model')
    parser.add_argument('--resume', type=str, default='', help='Path to Your checkpoint .pth file')

    parser.add_argument('--pretrained', type=str, default='pretrained/swin_tiny_patch4_window7_224.pth',
                        help="Path to the pretrained model. If you don't provide anything, it will train from scratch.")

    # FIXME:Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    print("Your arguments are listed below, Please check carefully before training")
    for key, value in args.__dict__.items():
        print("--" * 10)
        print("{} : {}".format(key, value))
    print("--" * 10)

    main(args)
