from networks.swin_unet import SwinUnet
from networks.deeplab.deeplab import DeepLab
from dataset import cancer_seg_data, cancer_public_seg_data
from loss import DiceLoss
from networks.config import get_config

import ttach as tta

from utils import *

from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib
matplotlib.use('agg')

import argparse
import os
import tqdm

from monai.inferers import sliding_window_inference
import warnings
warnings.filterwarnings("ignore")

def fill_mask(y_pred):
    imageHeight, imageWidth = y_pred.shape
    _, binary = cv2.threshold(y_pred, 127, 255, cv2.THRESH_BINARY)
    binary, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    pred_mask = np.zeros((imageHeight, imageWidth))
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50:
            continue
        pred_mask = cv2.drawContours(pred_mask, contour[np.newaxis,:,:], -1, 255, cv2.FILLED).astype('uint8')
    return pred_mask

def valid(args, model):
    train_data_list, valid_data_list = os.listdir(os.path.join(args.root, args.split_root)), os.listdir(os.path.join(args.root, args.split_root))
    valid_data = cancer_seg_data(args, valid_data_list, isTrain=False)

    print('Number of validation data: ', len(valid_data))
    valid_loader = DataLoader(valid_data, batch_size=1, num_workers=args.workers, shuffle=False, pin_memory=True)

    dice_loss = DiceLoss(n_classes=args.num_classes).cuda()
    ce_loss = CrossEntropyLoss(label_smoothing=0.1).cuda(6)
    model.eval()

    with torch.no_grad():
        pbar = tqdm.tqdm(total=len(valid_loader), ncols=0, desc="val", unit=" step")
        mean = torch.as_tensor([0.827, 0.621, 0.769])
        std = torch.as_tensor([0.168, 0.302, 0.190])
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)

        count = 0

        total_loss = 0.
        total, correct = 0, 0
        TP_total, P_total, PP_total = 0, 0, 0
        for image, mask in valid_loader:
            count += image.shape[0]
            image, mask = image.cuda(), mask.cuda()
            mask = mask[:, 0, :, :]
            if args.model == 'deeplab':
                pred = model(image)
            else:
                pred = sliding_window_inference(image, (args.img_size, args.img_size), 2, model, overlap=0.8)
            
            loss_dice = dice_loss(pred, mask, softmax = True)
            loss_ce = ce_loss(pred, mask.long())
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            total_loss += loss.item()

            pred = pred.cpu().detach().numpy()[0]
            mask = mask.cpu().detach().numpy()[0]

            pred_label = np.argmax(pred, axis=0) * 255
            pred_label = fill_mask(pred_label.astype(np.uint8)) / 255

            # kernel = np.ones((3, 3), np.uint8)
            # pred_label = cv2.erode(pred_label, kernel, iterations = 1)

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
            mask = mask.astype(np.uint8)
            pred_label = pred_label.astype(np.uint8)

            # plt.imshow(mask[:,:,np.newaxis] * image[0].transpose((1, 2, 0)))
            # plt.savefig(os.path.join(args.save_fig, '%d.png' % count))
            # plt.imshow(pred_label[:,:,np.newaxis] * image[0].transpose((1, 2, 0)))
            # plt.savefig(os.path.join(args.save_fig, '%d.jpg' % count))
            # plt.clf()
            # plt.close()

            plt.figure(figsize=(18,12))
            plt.subplot(1,3,1)
            plt.title("image")
            plt.imshow(image[0].transpose((1, 2, 0)))
            plt.subplot(1,3,2)
            plt.title("label")
            plt.imshow(mask[:,:,np.newaxis])
            plt.subplot(1,3,3)
            plt.title("predcited")
            plt.imshow(pred_label[:,:,np.newaxis])
            plt.savefig(os.path.join(args.save_fig, '%d.png' % count))
            plt.clf()
            plt.close()

        Recall = TP_total / P_total
        Precision = TP_total / PP_total
        dice_score = (2 * Precision * Recall) / (Precision + Recall)

        pbar.set_postfix(
            loss=f"{total_loss:.4f}",
            dice_score=f"{dice_score:.4f}",
            acc=f"{(correct / total):.4f}"
        )

        pbar.close()

def test(args, model):
    public_data = cancer_public_seg_data(root=args.root)

    print('Number of public data: ', len(public_data))
    public_loader = DataLoader(public_data, batch_size=1, num_workers=args.workers, shuffle=False, pin_memory=True)

    softmax = nn.Softmax2d()
    with torch.no_grad():
        pbar = tqdm.tqdm(total=len(public_loader), ncols=0, desc="public", unit=" step")
        mean = torch.as_tensor([0.827, 0.621, 0.769])
        std = torch.as_tensor([0.168, 0.302, 0.190])
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)

        count = 0

        for image, image_name in public_loader:
            count += image.shape[0]
            image = image.cuda()
            
            if args.model == 'ensemble':
                for i, submodel in enumerate(model):
                    if i == 0:
                        pred = softmax(sliding_window_inference(image, (args.img_size, args.img_size), 2, submodel, overlap=0.8))
                        # pred = softmax(submodel(image))
                        pred = pred.cpu().detach()
                        sum_pred = pred
                        # pred_label = torch.where(pred[:, 1, :, :] >= args.threshold, 1, 0)
                    elif i > 1:
                        pred = softmax(sliding_window_inference(image, (args.img_size, args.img_size), 2, submodel, overlap=0.8))
                        pred = pred.cpu().detach()
                        sum_pred += pred
                        # pred = torch.where(pred[:, 1, :, :] >= args.threshold, 1, 0)
                        # pred_label = pred_label.logical_or(pred)
                    else:
                        pred = softmax(submodel(image))
                        pred = pred.cpu().detach()
                        sum_pred += pred
                        # pred = torch.where(pred[:, 1, :, :] >= args.threshold, 1, 0)
                        # pred_label = pred_label.logical_or(pred)
                sum_pred /= len(model)
                pred_label = torch.where(sum_pred[:, 1, :, :] >= args.threshold, 1, 0)
                # pred_label = pred_label.logical_or(sum_pred)
                pred_label = pred_label.numpy()[0] * 255
            else:
                if args.model == 'deeplab':
                    pred = model(image)
                else:
                    pred = sliding_window_inference(image, (args.img_size, args.img_size), 2, model, overlap=0.5)

                pred = softmax(pred)
                pred = pred.cpu().detach().numpy()
                pred_label = np.where(pred[:, 1, :, :] >= args.threshold, 1, 0)[0] * 255

            pred_label = fill_mask(pred_label.astype(np.uint8))
            pbar.update()

            image = image.cpu().detach()
            image = ((image * std) + mean) * 255
            image = image.numpy()
            image = image.astype(np.uint8)
            pred_label = (pred_label).astype(np.uint8)
            
            plt.figure(figsize=(18,12))
            plt.subplot(1,2,1)
            plt.title("image")
            plt.imshow(image[0].transpose((1, 2, 0)))
            plt.subplot(1,2,2)
            plt.title("predcited")
            plt.imshow(pred_label)
            plt.savefig(os.path.join(args.public_fig + "_tmp", image_name[0][:-4] + '.png'))
            plt.clf()
            plt.close()
            cv2.imwrite(os.path.join(args.public_fig, image_name[0][:-4] + '.png'), pred_label)

        pbar.close()

def get_args():
    parser = argparse.ArgumentParser(
        description='Train the UNet on images and target masks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root', type=str, default='../../dataset', help='Number of epochs')
    parser.add_argument('--split_root', type=str, default='lung_cancer/images/valid', help='Number of epochs')
    parser.add_argument('--num_classes', type=int,default=2, help='Number of classes')
    parser.add_argument('--img_size', type=int,default=896, help='size of image')
    parser.add_argument('--cfg', type=str, default='./configs/swin_tiny_patch4_window7_224_lite.yaml', metavar="FILE", help='path to config file', )
    parser.add_argument('--workers', type=int, default=0, help='number of workers')
    parser.add_argument('--model', type=str, default='deeplab', help='swinUnet/deeplab/ensemble')
    parser.add_argument('--load', type=str, default='./checkpoint/deeplab_1280_900/model.pth', help='Load model from a .pth file')
    parser.add_argument('--scale', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--threshold', type=float, default=0.35, help='threshold for positive labels')

    parser.add_argument('--save_fig', default='./out_mask', help='path to save figures')
    parser.add_argument('--public_fig', default='./publicFig_deeplab', help='path to save figures')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    args.save_fig = args.save_fig + '_' + args.model
    os.makedirs(args.save_fig, exist_ok= True)
    os.makedirs(args.public_fig, exist_ok= True)
    os.makedirs(args.public_fig + "_tmp", exist_ok= True)

    if args.model == 'ensemble':
        model = []
        model_tmp = DeepLab(backbone='resnet')
        model_tmp.decoder.last_conv[8] = nn.Conv2d(256, args.num_classes, kernel_size=(1,1), stride=(1,1))
        model_tmp.load_state_dict(torch.load('./checkpoint/yolo_0.95/deeplab_full/model_best.pth'))
        model_tmp.eval()
        model_tmp = model_tmp.cuda()
        model_tmp = tta.SegmentationTTAWrapper(model_tmp, tta.aliases.d4_transform(), merge_mode='mean')
        model.append(model_tmp)
        model_tmp = DeepLab(backbone='resnet')
        model_tmp.decoder.last_conv[8] = nn.Conv2d(256, args.num_classes, kernel_size=(1,1), stride=(1,1))
        model_tmp.load_state_dict(torch.load('./checkpoint/yolo_0.95/deeplab_1280_900/model_best.pth'))
        model_tmp.eval()
        model_tmp = model_tmp.cuda()
        model_tmp = tta.SegmentationTTAWrapper(model_tmp, tta.aliases.d4_transform(), merge_mode='mean')
        model.append(model_tmp)
        # model_tmp = DeepLab(backbone='resnet')
        # model_tmp.decoder.last_conv[8] = nn.Conv2d(256, args.num_classes, kernel_size=(1,1), stride=(1,1))
        # model_tmp.load_state_dict(torch.load('./checkpoint/yolo_3/deeplab_v2/model_best.pth'))
        # model_tmp.eval()
        # model_tmp = model_tmp.cuda()
        # model_tmp = tta.SegmentationTTAWrapper(model_tmp, tta.aliases.d4_transform(), merge_mode='mean')
        # model.append(model_tmp)
        # model_tmp = DeepLab(backbone='resnet')
        # model_tmp.decoder.last_conv[8] = nn.Conv2d(256, args.num_classes, kernel_size=(1,1), stride=(1,1))
        # model_tmp.load_state_dict(torch.load('./checkpoint/yolo_0.8/deeplab_v2/model_best.pth'))
        # model_tmp.eval()
        # model_tmp = model_tmp.cuda()
        # model_tmp = tta.SegmentationTTAWrapper(model_tmp, tta.aliases.d4_transform(), merge_mode='mean')
        # model.append(model_tmp)
        # model_tmp = DeepLab(backbone='resnet')
        # model_tmp.decoder.last_conv[8] = nn.Conv2d(256, args.num_classes, kernel_size=(1,1), stride=(1,1))
        # model_tmp.load_state_dict(torch.load('./checkpoint/yolo_1/deeplab_v2/model_best.pth'))
        # model_tmp.eval()
        # model_tmp = model_tmp.cuda()
        # model_tmp = tta.SegmentationTTAWrapper(model_tmp, tta.aliases.d4_transform(), merge_mode='mean')
        # model.append(model_tmp)
        # config = get_config(args)
        # model_tmp = SwinUnet(config, img_size=args.img_size, num_classes=2)
        # model_tmp.load_state_dict(torch.load('./checkpoint/yolo_1/swinUnet/model_best.pth'))
        # model_tmp.eval()
        # model_tmp = model_tmp.cuda()
        # # model_tmp = tta.SegmentationTTAWrapper(model_tmp, tta.aliases.d4_transform(), merge_mode='mean')
        # model.append(model_tmp)
        # model_tmp = SwinUnet(config, img_size=args.img_size, num_classes=2)
        # model_tmp.load_state_dict(torch.load('./checkpoint/yolo_0.95/swinUnet/model_best.pth'))
        # model_tmp.eval()
        # model_tmp = model_tmp.cuda()
        # # model_tmp = tta.SegmentationTTAWrapper(model_tmp, tta.aliases.d4_transform(), merge_mode='mean')
        # model.append(model_tmp)
    else:
        if args.model == 'swinUnet':
            """Swin_Unet"""
            config = get_config(args)
            model = SwinUnet(config, img_size=args.img_size, num_classes=2)
            model.load_from(config)
        elif args.model == 'deeplab':
            model = DeepLab(backbone='resnet')
            model.decoder.last_conv[8] = nn.Conv2d(256, args.num_classes, kernel_size=(1,1), stride=(1,1))
        
        if args.load:
            print("Load pretrained model!!")
            model.load_state_dict(torch.load(args.load))
        model = model.cuda()
        model.eval()
        model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')

    # valid(args, model)
    test(args, model)


