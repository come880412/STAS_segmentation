import json

import os
import cv2
import numpy as np

import albumentations as A
import ttach as tta

import torch

if __name__ == '__main__':
    dataset = '../../dataset'
    save_dir = './visualization'

    os.makedirs(save_dir, exist_ok=True)

    # image_path = os.path.join(dataset, 'Train_Images', '00000000.jpg')
    image_path = os.path.join(dataset, 'Public_Image', 'Public_00000011.jpg')
    annotation = os.path.join(dataset, 'Train_Annotations_SEG', '00000000.json')

    with open(annotation) as f:
        data = json.load(f)

    image = cv2.imread(image_path)

    image_name = data["imagePath"]
    img_width = data["imageWidth"]
    img_height = data["imageHeight"]

    label_info = data['shapes']
    
    mask = np.zeros((image.shape[0], image.shape[1], 1))
    for label in label_info:
        label_name = label['label']
        pts = np.array(label['points']).astype(np.int)
        cv2.fillPoly(mask, [pts], color=1)
    mask *= 255
    # transform = A.Compose([
    #             A.RandomCrop(width=self.image_size, height=self.image_size),
    #             A.HorizontalFlip(p=0.5),
    #             A.RandomBrightnessContrast(p=0.2),
            # ])
    cv2.imwrite(os.path.join(save_dir, 'image_ori.png'), image)
    cv2.imwrite(os.path.join(save_dir, 'mask_ori.png'), mask)

    # transform = A.RandomCrop(width=1280, height=900)(image=image, mask=mask)
    # image_crop, mask_crop = transform["image"], transform["mask"]
    # cv2.imwrite(os.path.join(save_dir, 'image_crop.png'), image_crop)
    # cv2.imwrite(os.path.join(save_dir, 'mask_crop.png'), mask_crop)

    # transform = A.HorizontalFlip(p=1)(image=image, mask=mask)
    # transformed_image, transformed_mask = transform["image"], transform["mask"]
    # cv2.imwrite(os.path.join(save_dir, 'image_hor.png'), transformed_image)
    # cv2.imwrite(os.path.join(save_dir, 'mask_hor.png'), transformed_mask)

    # transform = A.VerticalFlip(p=1)(image=image, mask=mask)
    # transformed_image, transformed_mask = transform["image"], transform["mask"]
    # cv2.imwrite(os.path.join(save_dir, 'image_ver.png'), transformed_image)
    # cv2.imwrite(os.path.join(save_dir, 'mask_ver.png'), transformed_mask)

    # transform = A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1)(image=image, mask=mask)
    # transformed_image, transformed_mask = transform["image"], transform["mask"]
    # cv2.imwrite(os.path.join(save_dir, 'image_bright.png'), transformed_image)
    # cv2.imwrite(os.path.join(save_dir, 'mask_bright.png'), transformed_mask)

    # transform = A.Rotate(limit=45, always_apply=True, p=1.0)(image=image, mask=mask)
    # roate_image, roate_mask = transform["image"], transform["mask"]
    # cv2.imwrite(os.path.join(save_dir, 'image_rotate.png'), roate_image)
    # cv2.imwrite(os.path.join(save_dir, 'mask_rotate.png'), roate_mask)

    transform = A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.0, rotate_limit=0.0, p=1)(image=image, mask=mask)
    transformed_image, transformed_mask = transform["image"], transform["mask"]
    cv2.imwrite(os.path.join(save_dir, 'image_shifts.png'), transformed_image)
    cv2.imwrite(os.path.join(save_dir, 'mask_shifts.png'), transformed_mask)

    # transform = A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.1, rotate_limit=0.0, p=1)(image=image, mask=mask)
    # transformed_image, transformed_mask = transform["image"], transform["mask"]
    # cv2.imwrite(os.path.join(save_dir, 'image_scale.png'), transformed_image)
    # cv2.imwrite(os.path.join(save_dir, 'mask_scale.png'), transformed_mask)
    
    
    # cv2.imshow('Extracted Image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(image_name, img_width, img_height)
