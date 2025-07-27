# Pre-Requirements

### Download dataset
```
kaggle competitions download -c uadamage-demining-competition
unzip uadamage-demining-competition.zip
```

### Install mmdetection
Assuming:
NVIDIA-SMI 575.64.03 CUDA Version: 12.9

```
conda create -n mmdet330 python=3.10 -y
conda activate mmdet330

pip install torch==2.1.0 torchvision==0.16 --index-url https://download.pytorch.org/whl/cu121
pip install openmim
mim install "mmengine==0.10.7"
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
mim install "mmdet>=3.3.0"  
```


### Run convert dataset

Convertation from the competition format to coco format for mmdetection train
```
python csv_to_coco_drop_background.py \
       --src ./train \
       --dst ./train_coco \
       --val-ratio 0.15 \
       --seed 42
```

### Run train

Convertation from the competition format to coco format for mmdetection train
```
python train.py dino_config_flat.py 
```