# STAS_segmentation
Competition URL: https://tbrain.trendmicro.com.tw/Competitions/Details/22 (public 30th, private 2th)

# Getting started
- Clone this repo to your local
``` bash
git clone https://github.com/come880412/STAS_segmentation
cd STAS_segmentation
```

## Computer Equipment
- System: Ubuntu20.04
- Pytorch version: Pytorch 1.7 or higher
- Python version: Python 3.7
- Testing:  
CPU: AMR Ryzen 7 4800H with Radeon Graphics
RAM: 32GB  
GPU: NVIDIA GeForce RTX 1660Ti 6GB  

- Training (TWCC):  
CPU: Intel(R) Xeon(R) Gold 6154 CPU @ 3.00GHz \
RAM: 180 GB \
GPU: Tesla V100 32 GB * 4

## Packages
Please read the "requirement.txt" for the details.

## Download & preprocess dataset
- You should prepare the dataset from [here](https://tbrain.trendmicro.com.tw/Competitions/Details/22), and put the dataset on the folder `../dataset`. After doing so, please use the following command to do data preprocessing.
``` bash
python3 preprocessing.py 
```
- Note: please modify the dataset path on the script `preprocessing.py`.

## Download pretrained models
- Please download the pretrained models from [here](https://drive.google.com/drive/folders/1sVYVzNTvX3deE8hJf2qTnKQ2Na8b3pRu?usp=sharing), and put the models on the folder `./checkpoint`.

## Inference
- After downloading the pretrained models and preparing the datasets, you could use the following command to test the best results on the public/private leaderboard.
``` bash
python3 test.py --root path/to/dataset/Public_image --model ./checkpoint/deeplab_1280_900/model.pth --threshold 0.35
```
- The result will be saved on the folder `./publicFig_deeplab` automatically.

## Training
- You should download the COCO pretrained models on [1]. And put the model on the folder `./pretrained`. After that, please use the following script to train the best model used in this competition.
``` bash
bash train.sh
```

# Reference
[1] https://github.com/jfzhang95/pytorch-deeplab-xception
