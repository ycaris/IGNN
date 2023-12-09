# IGNN
Code repo based on the Paper "Cross-Scale Internal Graph Neural Network for Image Super-Resolution" &nbsp; [[paper]](https://proceedings.neurips.cc/paper/2020/file/23ad3e314e2a2b43b4c720507cec0723-Paper.pdf). Modified for Medical Imaging Task 


## Prepare datasets
1 Download training dataset and test datasets from [here](https://drive.google.com/file/d/1fFBCXkUIgHkjqWiCeW7w-1TYHE0A2ZZF/view?usp=sharing) and [Kaggle Website](https://www.kaggle.com/code/mayank1101sharma/image-super-resolution-on-chest-x-ray-images/input). 

2 Crop training dataset DIV2K to sub-images and preprocess X-ray Dataset.
```
python ./datasets/prepare_DIV2K_subimages.py
python ./preprocess/crop.py
python ./preprocess/resize.py
```

## Dependencies and Installation
The code is tested with Python 3.7, PyTorch 1.13.1 and Cuda 11.7

You could install the required packages with the requirements.txt file.

'''
conda create --name <ignn> --file <requirements.txt>
'''

2 Install PyInn.
```
pip install git+https://github.com/szagoruyko/pyinn.git@master
```
3 Install matmul_cuda.
```
bash install.sh
```
4 Install other dependencies.
```
pip install -r requirements.txt
```

## Training
Use the following command to train the network:

```
python runner.py
        --gpu \
        --phase 'train'\
        --dataroot \
        --out 
```
Use the following command to resume training the network:


You can also use the following simple command with different settings in config.py:

```
python runner.py
```

## Testing
Use the following command to test the network on benchmark datasets (w/ GT):
```
python runner.py \
        --gpu \
        --phase 'test'\
        --weights \
        --dataroot \
        --testname \
        --out 
```


## Citation

```
@inproceedings{zhou2020cross,
title={Cross-scale internal graph neural network for image super-resolution},
author={Zhou, Shangchen and Zhang, Jiawei and Zuo, Wangmeng and Loy, Chen Change},
booktitle={Advances in Neural Information Processing Systems},
year={2020}
}
```
