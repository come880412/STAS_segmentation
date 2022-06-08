import os
from re import I
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import albumentations as A
import tifffile as tiff
import json
import cv2
import torch
import argparse
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.model_selection import train_test_split

label_dict = {0:"background", 1:"stas"}


transforms_test =   transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.827, 0.621, 0.769), (0.168, 0.302, 0.190)),])

class cancer_seg_semi_data(Dataset):
    def __init__(self, args, data, isTrain=True):
        self.data = data
        self.semi = args.semi
        self.root = args.root
        self.isTrain = isTrain
        self.image_size = args.img_size

        self.data_info = []

        if isTrain:
            self.transform = A.Compose([
                                A.RandomCrop(width=self.image_size[0], height=self.image_size[1]),
                                A.HorizontalFlip(p=0.5),
                                A.VerticalFlip(p=0.5),
                                # A.RandomRotate90(p=0.5),
                                # A.Transpose(p=0.5),
                                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.3),
                                A.ShiftScaleRotate(p=0.5)
                            ])
        else:
            self.transform = None

        for image_name in self.data:
            points = []
            label_name = image_name[:-4] + '.json'

            with open(os.path.join(self.root, 'Train_Annotations_SEG', label_name)) as f:
                info = json.load(f)
            label_info = info['shapes']
            for label in label_info:
                tmp_points = np.array(label['points']).astype(np.int)
                points.append(tmp_points)
            image_path = os.path.join(self.root, 'Train_Images', image_name[:-4] + '.jpg')

            self.data_info.append([image_path, points, 'normal'])
        if isTrain:
            for image_name in os.listdir(os.path.join(self.root, 'Public_Image')):
                image_path = os.path.join(self.root, 'Public_Image', image_name)
                mask_path = os.path.join('./semi', image_name[:-4] + '.png')

                self.data_info.append([image_path, mask_path, 'semi'])
        
    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, index):
        image_path, points, train_type = self.data_info[index]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # (H, W, 3)

        height, width, _  = image.shape

        if train_type == 'normal':
            mask = np.zeros((height, width, 1))
            for pts in points:
                cv2.fillPoly(mask, [pts], color=1)
        else:
            mask = cv2.imread(points, 0)
            mask = mask[:,:,np.newaxis] / 255.0
        
        # self.visualization(image, mask)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        image = image.transpose((2, 0, 1)) # (3, H, W)
        mask = mask.transpose((2, 0, 1)) # (1, H, W)
        
        image = torch.from_numpy((image.copy()))
        image = image.float() / 255.0
        mean = torch.as_tensor([0.827, 0.621, 0.769], dtype=image.dtype, device=image.device)
        std = torch.as_tensor([0.168, 0.302, 0.190], dtype=image.dtype, device=image.device)
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        image.sub_(mean).div_(std)
        mask = torch.from_numpy((mask.copy()))

        return image, mask

class cancer_seg_data(Dataset):
    def __init__(self, args, data, isTrain=True):
        self.data = data
        self.root = args.root
        self.isTrain = isTrain
        self.image_size = args.img_size

        self.data_info = []

        if isTrain:
            self.transform = A.Compose([
                                A.RandomCrop(width=self.image_size[0], height=self.image_size[1]),
                                A.HorizontalFlip(p=0.3),
                                A.VerticalFlip(p=0.3),
                                # A.RandomRotate90(p=0.3),
                                # A.Transpose(p=0.3),
                                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.3),
                                A.ShiftScaleRotate(p=0.3)
                            ])
        else:
            self.transform = None

        for image_name in self.data:
            points = []
            label_name = image_name[:-4] + '.json'

            with open(os.path.join(self.root, 'Train_Annotations_SEG', label_name)) as f:
                info = json.load(f)
            label_info = info['shapes']
            for label in label_info:
                tmp_points = np.array(label['points']).astype(np.int)
                points.append(tmp_points)
            image_path = os.path.join(self.root, 'Train_Images', image_name[:-4] + '.jpg')

            self.data_info.append([image_path, points])
        
    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, index):
        image_path, points = self.data_info[index]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # (H, W, 3)

        height, width, _  = image.shape

        mask = np.zeros((height, width, 1))
        for pts in points:
            cv2.fillPoly(mask, [pts], color=1)

        # self.visualization(image, mask)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        image = image.transpose((2, 0, 1)) # (3, H, W)
        mask = mask.transpose((2, 0, 1)) # (1, H, W)
        
        image = torch.from_numpy((image.copy()))
        image = image.float() / 255.0
        mean = torch.as_tensor([0.827, 0.621, 0.769], dtype=image.dtype, device=image.device)
        std = torch.as_tensor([0.168, 0.302, 0.190], dtype=image.dtype, device=image.device)
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        image.sub_(mean).div_(std)
        mask = torch.from_numpy((mask.copy()))

        return image, mask
    def visualization(self, image, mask):
        transform = A.RandomCrop(width=1280, height=900)(image=image, mask=mask)
        image_crop, mask_crop = transform["image"], transform["mask"]

        transform = A.HorizontalFlip(p=1)(image=image_crop, mask=mask_crop)
        hor_image, hor_mask = transform["image"], transform["mask"]

        transform = A.VerticalFlip(p=1)(image=image_crop, mask=mask_crop)
        ver_image, ver_mask = transform["image"], transform["mask"]

        transform = A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1)(image=image_crop, mask=mask_crop)
        bright_image, bright_mask = transform["image"], transform["mask"]

        transform = A.RandomRotate90(p=1)(image=image_crop, mask=mask_crop)
        rotate_image, rotate_mask = transform["image"], transform["mask"]

        transform = A.Transpose(p=1)(image=image_crop, mask=mask_crop)
        transpose_image, transpose_mask = transform["image"], transform["mask"]

        transform = A.ShiftScaleRotate(p=1)(image=image_crop, mask=mask_crop)
        shift_image, shift_mask = transform["image"], transform["mask"]

        plt.figure(figsize=(18,12))
        plt.subplot(4,4,1)
        plt.title("image_ori")
        plt.imshow(image)
        plt.axis('off')
        plt.subplot(4,4,2)
        plt.title("mask_ori")
        plt.imshow(mask)
        plt.axis('off')
        plt.subplot(4,4,3)
        plt.title("image_cropped")
        plt.imshow(image_crop)
        plt.axis('off')
        plt.subplot(4,4,4)
        plt.title("mask_cropped")
        plt.imshow(mask_crop)
        plt.axis('off')
        plt.subplot(4,4,5)
        plt.title("image_vertial_flip")
        plt.imshow(ver_image)
        plt.axis('off')
        plt.subplot(4,4,6)
        plt.title("mask_vertial_flip")
        plt.imshow(ver_mask)
        plt.axis('off')
        plt.subplot(4,4,7)
        plt.title("image_horizontal_flip")
        plt.imshow(hor_image)
        plt.axis('off')
        plt.subplot(4,4,8)
        plt.title("mask_horizontal_flip")
        plt.imshow(hor_mask)
        plt.axis('off')
        plt.subplot(4,4,9)
        plt.title("image_rotate")
        plt.imshow(rotate_image)
        plt.axis('off')
        plt.subplot(4,4,10)
        plt.title("mask_rotate")
        plt.imshow(rotate_mask)
        plt.axis('off')
        plt.subplot(4,4,11)
        plt.title("image_rotateshift")
        plt.imshow(shift_image)
        plt.axis('off')
        plt.subplot(4,4,12)
        plt.title("mask_rotateshift")
        plt.imshow(shift_mask)
        plt.axis('off')
        plt.subplot(4,4,13)
        plt.title("image_bright")
        plt.imshow(bright_image)
        plt.axis('off')
        plt.subplot(4,4,14)
        plt.title("mask_bright")
        plt.imshow(bright_mask)
        plt.axis('off')
        plt.subplot(4,4,15)
        plt.title("image_transpose")
        plt.imshow(transpose_image)
        plt.axis('off')
        plt.subplot(4,4,16)
        plt.title("mask_transpose")
        plt.imshow(transpose_mask)
        plt.axis('off')
        plt.show()

class cancer_public_seg_data(Dataset):
    def __init__(self, root):
        self.root = root
        self.data_info = []
        for data in ['Public_Image']:
            for image_name in os.listdir(os.path.join(self.root, data)):
                self.data_info.append([os.path.join(self.root, data, image_name), image_name])

    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, index):
        image_path, image_name = self.data_info[index]

        image = Image.open(image_path).convert("RGB")
        return transforms_test(image), image_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train the UNet on images and target masks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root', type=str, default='../../dataset', help='Number of epochs')
    parser.add_argument('--epochs', type=int,default=5, help='Number of epochs')
    parser.add_argument('--img_size', type=int,default=896, help='size of image')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--load', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', type=float, default=0.1,help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--split_seed', type=int, default=2022, help='')
    args = parser.parse_args()
    data_list = os.listdir(os.path.join(args.root, 'Train_Images'))
    train_data_list, valid_data_list = train_test_split(data_list, random_state=args.split_seed, test_size=args.validation)

    mean = torch.as_tensor([0.827, 0.621, 0.769])
    std = torch.as_tensor([0.168, 0.302, 0.190])
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)

    valid_data = cancer_seg_data(args, train_data_list, isTrain=True)
    valid_loader = DataLoader(valid_data, batch_size=1, num_workers=0, shuffle=True, pin_memory=True)
    for image, mask in valid_loader:
        mask = mask[:,0,:,:]
        image = image.cpu().detach()
        image = ((image * std) + mean) * 255
        image = image.numpy()
        image = image.astype(np.uint8)
        # for batch_size in range(len(image)):
        #     plt.figure(figsize=(18,12))
        #     plt.subplot(1,2,1)
        #     plt.title("image")
        #     plt.imshow(image[batch_size].transpose((1, 2, 0)))
        #     plt.subplot(1,2,2)
        #     plt.title("label")
        #     plt.imshow(mask[batch_size])
        #     plt.show()

    # data_loader = DataLoader(data, batch_size=2, shuffle=True, drop_last=False)
    # iter_data = iter(data_loader)
    # image, mask = iter_data.next()
    # print(image.shape, mask.shape)