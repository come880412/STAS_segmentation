# ---------Deeplab-----------
python main.py --yolo_root lung_cancer_0.95/images \
               --epochs 100 --warmup_epochs 10 --model deeplab --save_model ./checkpoint/yolo_0.8 \
               --batch_size 16 --workers 8 --scheduler linearwarmup --device 1 --val_freq 2 --img_size 1280 900

# python main.py --yolo_root lung_cancer_0.9/images \
#                --epochs 100 --warmup_epochs 10 --model deeplab --save_model ./checkpoint/yolo_0.8 \
#                --batch_size 16 --workers 8 --scheduler linearwarmup --device 1 --val_freq 2 --img_size 1280 900

# python main.py --yolo_root lung_cancer_0.8/images \
#                --epochs 100 --warmup_epochs 10 --model deeplab --save_model ./checkpoint/yolo_0.8 \
#                --batch_size 16 --workers 8 --scheduler linearwarmup --device 1 --val_freq 2 --img_size 1280 900


# ---------Swinunet-----------
# python main.py --yolo_root lung_cancer_0.95/images --lr 0.05 \
#                --epochs 100 --warmup_epochs 25 --model swinUnet --save_model ./checkpoint/yolo_0.95 \
#                --batch_size 8 --workers 8 --scheduler cosine --device 1 --val_freq 2 --img_size 896 896

# python main.py --yolo_root lung_cancer_0.9/images --lr 0.05 \
#                --epochs 100 --warmup_epochs 25 --model swinUnet --save_model ./checkpoint/yolo_0.95 \
#                --batch_size 8 --workers 8 --scheduler cosine --device 1 --val_freq 2 --img_size 896 896

# python main.py --yolo_root lung_cancer_0.8/images --lr 0.05 \
#                --epochs 100 --warmup_epochs 25 --model swinUnet --save_model ./checkpoint/yolo_0.95 \
#                --batch_size 8 --workers 8 --scheduler cosine --device 1 --val_freq 2 --img_size 896 896
 