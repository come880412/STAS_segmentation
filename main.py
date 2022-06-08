from networks.swin_unet import SwinUnet
from networks.deeplab.deeplab import DeepLab
from dataset import cancer_seg_data, cancer_seg_semi_data
from loss import DiceFocalLoss, DiceCeLoss, FocalLoss
from networks.config import get_config

from utils import *

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
import segmentation_models_pytorch as smp

from tensorboardX import SummaryWriter
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import cv2

import argparse
import os
import tqdm
from sklearn.model_selection import train_test_split

from monai.inferers import sliding_window_inference

def fill_mask(y_pred):
    imageHeight, imageWidth = y_pred.shape
    _,binary = cv2.threshold(y_pred, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    pred_mask = np.zeros((imageHeight, imageWidth))
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50:
            continue
        pred_mask = cv2.drawContours(pred_mask, contour[np.newaxis,:,:], -1, 255, cv2.FILLED).astype('uint8')
    return pred_mask

def main(args, model, loss_func, optimizer, train_loader, valid_loader, scheduler):
    writer = SummaryWriter(log_dir='runs/%s' % args.model)
    model = model.cuda()
    loss_func = loss_func.cuda()

    max_dice = 0
    training_step = 0
    max_iterations = len(train_loader) * args.epochs
    for epoch in range(0, args.epochs):
        pbar = tqdm.tqdm(total=len(train_loader), ncols=0, desc="train[%d/%d]" % (epoch, args.epochs), unit=" step")

        model.train()

        total_loss = 0.
        TP_total, P_total, PP_total = 0, 0, 0
        total, correct = 0, 0
        for image, mask in train_loader:         
            lr = optimizer.param_groups[0]['lr']

            image, mask = image.cuda(), mask.cuda()
            mask = mask[:,0,:,:].long()

            pred = model(image)

            optimizer.zero_grad()
            loss = loss_func(pred, mask)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            mask = mask.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            pred_label = np.argmax(pred, axis=1)

            TP, P, PP = dice(pred_label, mask)
            TP_total += TP
            P_total += P
            PP_total += PP

            total, correct = get_positive_acc(pred_label, mask, total, correct)
            
            pbar.update()
            pbar.set_postfix(
                loss=f"{total_loss:.4f}",
                lr=f"{lr:.5f}",
                acc=f"{(correct / total):.4f}"
            )
            writer.add_scalar('Training_loss', loss.item(), training_step)
            training_step += 1

            if args.model == 'swinUnet':
                lr_ = args.lr * (1.0 - training_step / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_


        Recall = TP_total / P_total
        Precision = TP_total / PP_total
        dice_score = (2 * Precision * Recall) / (Precision + Recall)

        writer.add_scalar('Training_dice_score', dice_score, epoch)

        pbar.set_postfix(
            loss=f"{total_loss:.4f}",
            dice_score=f"{dice_score:.4f}",
            lr=f"{lr:.5f}",
            acc=f"{(correct / total):.4f}"
        )
        pbar.close()
        if (epoch + 1) % args.val_freq == 0 or (epoch +1) == args.epochs:
            val_dice, val_loss = valid(args, model, loss_func, valid_loader, epoch, writer)
            if max_dice < val_dice:
                max_dice = val_dice
                torch.save(model.state_dict(), os.path.join(args.save_model, 'model_best.pth'))
                print("Save best model!!!")
            torch.save(model.state_dict(), os.path.join(args.save_model, 'model_last.pth'))
        if args.model != 'swinUnet':
            scheduler.step()

def valid(args, model, loss_func, valid_loader, epoch, writer):
    model.eval()

    with torch.no_grad():
        pbar = tqdm.tqdm(total=len(valid_loader), ncols=0, desc="val[%d/%d]" % (epoch, args.epochs), unit=" step")
        mean = torch.as_tensor([0.827, 0.621, 0.769])
        std = torch.as_tensor([0.168, 0.302, 0.190])
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)

        count = 0
        total_loss = 0.
        TP_total, P_total, PP_total = 0, 0, 0
        total, correct = 0, 0

        for image, mask in valid_loader:
            image, mask = image.cuda(), mask.cuda()
            mask = mask[:,0,:,:].long()

            if args.model == 'deeplab':
                pred = model(image)
            else:
                pred = sliding_window_inference(image, (args.img_size[0], args.img_size[0]), 4, model, overlap=0.8)

            loss = loss_func(pred, mask)
            total_loss += loss.item()

            mask = mask.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            pred_label = np.argmax(pred, axis=1)

            TP, P, PP = dice(pred_label, mask)
            TP_total += TP
            P_total += P
            PP_total += PP

            total, correct = get_positive_acc(pred_label, mask, total, correct)
            
            pbar.update()
            pbar.set_postfix(
                loss=f"{total_loss:.4f}",
                acc=f"{(correct / total):.4f}"
            )

            image = image.cpu().detach()
            image = ((image * std) + mean) * 255
            image = image.numpy()
            image = image.astype(np.uint8)
            for batch_size in range(len(pred_label)):
                plt.figure(figsize=(18,12))
                plt.subplot(1,3,1)
                plt.title("image")
                plt.imshow(image[batch_size].transpose((1, 2, 0)))
                plt.subplot(1,3,2)
                plt.title("label")
                plt.imshow(mask[batch_size])
                plt.subplot(1,3,3)
                plt.title("predcited")
                plt.imshow(pred_label[batch_size])
                plt.savefig(os.path.join(args.save_fig, '%d.png' % count))
                plt.clf()
                plt.close()
                count += 1

        Recall = TP_total / P_total
        Precision = TP_total / PP_total
        dice_score = (2 * Precision * Recall) / (Precision + Recall)

        writer.add_scalar('validation loss', total_loss, epoch)
        writer.add_scalar('validation_dice_score', dice_score, epoch)

        pbar.set_postfix(
            loss=f"{total_loss:.4f}",
            dice_score=f"{dice_score:.4f}",
            acc=f"{(correct / total):.4f}"
        )
        pbar.close()
    return dice_score, total_loss

def get_args():
    parser = argparse.ArgumentParser(
        description='Train the UNet on images and target masks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root', type=str, default='../../dataset', help='path to dataset')
    parser.add_argument('--split_root', type=str, default='lung_cancer/images', help='path to data')
    parser.add_argument('--semi', type=int, default=0, help='whether to de semi_supervised')

    parser.add_argument('--num_classes', type=int,default=2, help='Number of classes')
    parser.add_argument('--cfg', type=str, default='./configs/swin_tiny_patch4_window7_224_lite.yaml', metavar="FILE", help='path to config file', )
    
    parser.add_argument('--epochs', type=int,default=100, help='Number of epochs')
    parser.add_argument('--warmup_epochs', default=10, type=int, help='number of warmup epochs')
    parser.add_argument('--img_size', type=int, default=[1280, 900], help='size of image', nargs='+') # 896 for swinUnet
    parser.add_argument('--model', type=str, default='deeplab', help='swinUnet/deeplab/unet++')
    parser.add_argument('--dice_weight', type=float, default=0.6, help='dice loss weight')
    parser.add_argument('--load', type=str, default='', help='Load model from a .pth file')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--workers', type=int, default=8, help='number of workers')
    parser.add_argument('--scheduler', type=str, default='linearwarmup', help='cosine/linearwarmup')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    parser.add_argument('--adam', action='store_true', help='Whether to use adam optimizer(default=SGD)')
    parser.add_argument('--loss', default='Focal', help='CE/Focal')
    parser.add_argument('--val_freq', type=int, default=2, help='validation freq')
    parser.add_argument('--save_model', default='./checkpoint', help='path to save model')
    parser.add_argument('--save_fig', default='./out_mask', help='path to save figures')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    args.save_model = os.path.join(args.save_model, args.model)
    args.save_fig = args.save_fig + '_' + args.model
    os.makedirs(args.save_model, exist_ok=True)
    os.makedirs(args.save_fig, exist_ok= True)

    if args.semi:
        train_data_list, valid_data_list = os.listdir(os.path.join(args.root, args.split_root, 'train')), os.listdir(os.path.join(args.root, args.split_root, 'valid'))
        train_data = cancer_seg_semi_data(args, train_data_list, isTrain=True)
        valid_data = cancer_seg_semi_data(args, valid_data_list, isTrain=False)
    else:
        train_data_list, valid_data_list = os.listdir(os.path.join(args.root, args.split_root, 'train')), os.listdir(os.path.join(args.root, args.split_root, 'valid'))
        train_data = cancer_seg_data(args, train_data_list, isTrain=True)
        valid_data = cancer_seg_data(args, valid_data_list, isTrain=False)

    print('Number of training data: ', len(train_data))
    print('Number of validation data: ', len(valid_data))

    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=1, num_workers=args.workers, shuffle=False, pin_memory=True)

    if args.model == 'swinUnet':
        """Swin_Unet"""
        config = get_config(args)
        model = SwinUnet(config, img_size=args.img_size, num_classes=args.num_classes)
        model.load_from(config)
    elif args.model == 'deeplab':
        # model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True, progress=False)
        # model.classifier[4] = nn.Conv2d(256, args.num_classes, kernel_size=(1, 1), stride=(1, 1))
        model = DeepLab(backbone='resnet')
        model.load_state_dict(torch.load('./pretrained/deeplab-resnet.pth.tar')['state_dict'])
        model.decoder.last_conv[8] = nn.Conv2d(256, args.num_classes, kernel_size=(1,1), stride=(1,1))

    if args.load:
        print('Load pretrained model!!')
        model.load_state_dict(torch.load(args.load))

    if args.adam:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001, nesterov=True)

    if args.loss == 'Focal':
        loss_func = DiceFocalLoss(args)
    elif args.loss == 'CE':
        loss_func = DiceCeLoss(args)

    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    elif args.scheduler == 'linearwarmup':
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.epochs)


    main(args, model, loss_func, optimizer, train_loader, valid_loader, scheduler)


