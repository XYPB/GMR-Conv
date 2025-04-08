# GMR-Conv: An Efficient Rotation and Reflection Equivariant Convolution Kernel Using Gaussian Mixture Rings

[![PyPI version](https://img.shields.io/pypi/v/GMR-Conv.svg)](https://pypi.org/project/GMR-Conv/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE) [![arXiv:2504.02819](https://img.shields.io/badge/arXiv-2504.02819-B31B1B.svg)](https://arxiv.org/abs/2504.02819)

This is the official implementation of paper "GMR-Conv: An Efficient Rotation and Reflection Equivariant Convolution Kernel Using Gaussian Mixture Rings". It is also a following up of our previous research [SRE-Conv](https://github.com/XYPB/SRE-Conv).

*[Yuexi Du](https://xypb.github.io/), Jiazhen Zhang, [Nicha C. Dvornek](https://www.hellonicha.com/), [John A. Onofrey](https://medicine.yale.edu/profile/john-onofrey/)*

*Yale University*

![teaser](assets/teaser_stats_wide1.png)

For any question related to the implementation, please open issue in this repo, we will respond to you ASAP. For other question about the paper, please contact the author directly.

## News

- **April. 2025** The code and PyPi package is officially released! The pre-print is also available on [arXiv](https://arxiv.org/abs/2504.02819) now.

## Abstract

> Symmetry, where certain features remain invariant under geometric transformations, can often serve as a powerful prior in designing convolutional neural networks (CNNs). While conventional CNNs inherently support translational equivariance, extending this property to rotation and reflection has proven challenging, often forcing a compromise between equivariance, efficiency, and information loss. In this work, we introduce Gaussian Mixture Ring Convolution (GMR-Conv), an efficient convolution kernel that smooths radial symmetry using a mixture of Gaussian-weighted rings. This design mitigates discretization errors of circular kernels, thereby preserving robust rotation and reflection equivariance without incurring computational overhead. We further optimize both the space and speed efficiency of GMR-Conv via a novel parameterization and computation strategy, allowing larger kernels at an acceptable cost. Extensive experiments on eight classification and one segmentation datasets demonstrate that GMR-Conv not only matches conventional CNNs' performance but can also surpass it in applications with orientation-less data. GMR-Conv is also proven to be more robust and efficient than the state-of-the-art equivariant learning methods. Our work provides inspiring empirical evidence that carefully applied radial symmetry can alleviate the challenges of information loss, marking a promising advance in equivariant network architectures.


## Installation

We provide both the PyPI package for [GMR-Conv](https://pypi.org/project/GMR-Conv/) and the code to reproduce the experiment results in this repo.

To install and directly use the GMR-Conv, please run the following command:
```bash
pip install GMR-Conv
```

The minimal requirement for the `GMR-Conv` package is:
```bash
"scipy>=1.9.0",
"numpy>=1.22.0",
"torch>=1.8.0",
"pillow>=9.2.0",
"monai>=0.7.0", # for UNet implementation
```
These packages will be installed automatically when installing the `GMR-Conv`.

**Note**: Using lower version of torch and numpy should be fine given that we didn't use any new feature in the new torch version, but we do suggest you to follow the required dependencies. If you have to use the different version of torch/numpy, you may also try to install the package from source code at [project repo](https://github.com/XYPB/GMR-Conv).

## Usage

Our GMR-Conv is implemented with the same interface as conventional torch convolutional layer. It can be used easily in any modern deep learning CNN implemented in PyTorch.

```python
import torch
from GMR_Conv import GMR_Conv2d

x = torch.randn(2, 3, 32, 32)
# create a 2D GMR-Conv of size 3x3 and padding 1
gmr_conv = GMR_Conv2d(3, 16, 3, 1, 1)
y = gmr_conv(x)
x_rot = torch.rot90(x, 1, (2, 3))
y_rot = gmr_conv(x_rot)
# check equivariance under 90-degree rotation
print(torch.allclose(torch.rot90(y, 1, (2, 3)), y_rot, atol=1e-6))
```

For more detail about the specific argument for our GMR-Conv, please refer to [here](https://github.com/XYPB/GMR-Conv/blob/0f0cc8a15a1647688a478ff0864998624c13e98c/src/GMR_Conv/gmr_conv.py#L51-L72).

We have also provided [GMR-ResNet](https://github.com/XYPB/GMR-Conv/blob/main/src/GMR_Conv/gmr_resnet.py), [GMR-ResNet3D](https://github.com/XYPB/GMR-Conv/blob/main/src/GMR_Conv/gmr_resnet_3d.py), and [GMR-UNet](https://github.com/XYPB/GMR-Conv/blob/main/src/GMR_Conv/gmr_unet.py) in this repo, you may also use it as regular ResNet but with equivariance.

```python
import torch
from GMR_Conv import gmr_resnet18

x = torch.randn(2, 3, 32, 32)
# use "gmr_conv_size" argument to specify kernel size at each stage.
gmr_r18 = gmr_resnet18(gmr_conv_size=[9, 9, 5, 5], num_classes=10)
output = gmr_r18(x)
x_rot = torch.rot90(x, 1, [2, 3])
output_rot = gmr_r18(x_rot)
# You would expect the model to give equivariant output under rotated input
print(torch.allclose(output, output_rot, atol=1e-4))
```

For general CNN implemented in PyTorch, you may use ``convert_to_gmr_conv`` function to convert it from regular CNN to equivariant CNN using ``GMR_Conv2d``.

```python
import torch
import torchvision.models.resnet as resnet
from GMR_Conv import convert_to_gmr_conv

model = resnet.resnet18()
gmr_model = convert_to_gmr_conv(model)
```


## Reproducing Experiment

We also provides the code to reproduce our experimental results on the classification dataset in this repo.

### Environment

Besides the `GMR_Conv` package, you may also need to install the following packages:
```bash
glob2
datetime
pytz
matplotlib
pandas
tqdm
wandb
h5py==3.7.0
torch-geometric==2.6.0
scikit-learn==1.1.3
```

You also need to install the cosine-annealing scheduler with warmup as following:
```bash
pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
```

The specific PyTorch version we used is `1.13.0` with CUDA `11.7`.

### Datasets

Our experiments includes the following datasets and the corresponding download links:

|Dataset|Link|
|:---|:---:|
|CIFAR-10/100|PyTorch Automatic Download|
|NWPU-10/VHR-10|[Download Here](https://github.com/chaozhong2010/VHR-10_dataset_coco)|
|MTARSI|[Download Here](https://www.kaggle.com/datasets/aqibriaz/mtarsidataset)|
|NCT-CRC-100k|[Download Here](https://zenodo.org/records/1214456)|
|PatchCamelyon|[Download Here](https://www.kaggle.com/datasets/andrewmvd/metastatic-tissue-classification-patchcamelyon)|
|ImageNet-1k|[Download Here](https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description)|
|ModelNet10/40|PyTorch Automatic Download|

After downloading these dataset, please put them under the `./data` folder as following:

```bash
./data
├── MTARSI_1
├── ModelNet10
├── ModelNet40
├── NCT-CRC-HE-100K
├── NWPU_VHR-10_dataset
├── PatchCamelyon
└── imagenet-object-localization-challenge
```

You may also need to pre-process the NWPU-VHR-10 dataset to extract objects using `./datasets/VHR_preprocess.py`, see its [original repo](https://github.com/chaozhong2010/VHR-10_dataset_coco) for more information.

### Logging

We use `wandb` for training monitoring, all the logging results should be placed to `./logs` dataset as default.s

### Reproduce the Classification Results

#### CIFAR-10

Run the following command to train and evaluate the model under rotation and reflection.
```bash
python main.py \
    --cifar10 \
    --epochs 250 \
    --model-type gmr_resnet18 \
    --gmr-conv-size-list 9 9 5 5 \
    -b 128 \
    --lr 2e-2 \
    --cos \
    --sgd \
    --eval-rot \
    --eval-flip \
    --train-flip-p 0 \
    --res-keep-conv1 \
    --log \
    --cudnn \
    --moco-aug \
    --translation \
    --translate-ratio 0.2 \
    --save-model
```

#### NWPU-10

Run the following command to train and evaluate the model under rotation and reflection.
```bash
python main.py \
    --vhr10 \
    --epochs 250 \
    --model-type gmr_resnet18 \
    --gmr-conv-size-list 9 9 5 5 \
    -b 128 \
    --lr 2e-2 \
    --cos \
    --sgd \
    --eval-rot \
    --eval-flip \
    --train-flip-p 0 \
    --res-keep-conv1 \
    --log \
    --cudnn \
    --moco-aug \
    --translation \
    --translate-ratio 0.2 \
    --save-model 
```

#### MTARSI

Run the following command to train and evaluate the model under rotation and reflection.
```bash
python main.py \
    --mtarsi \
    --epochs 250 \
    --model-type gmr_resnet18 \
    --gmr-conv-size-list 9 9 5 5 \
    -b 128 \
    --lr 2e-2 \
    --cos \
    --sgd \
    --eval-rot \
    --eval-flip \
    --train-flip-p 0 \
    --res-keep-conv1 \
    --log \
    --cudnn \
    --moco-aug \
    --translation \
    --translate-ratio 0.2 \
    --save-model 
```

#### NCT-CRC

Run the following command to train and evaluate the model under rotation and reflection.
```bash
python main.py \
    --nct-crc \
    --epochs 50 \
    --model-type gmr_resnet18  \
    --gmr-conv-size-list 9 9 5 5 \
    -b 24 \
    --lr 2e-2 \
    --sgd \
    --cudnn \
    --cos \
    --eval-rot \
    --eval-flip \
    --train-flip-p 0 \
    --log \
    --moco-aug \
    --translation \
    --translate-ratio 0.1 \
    --save-model
```

#### PatchCamelyon

Run the following command to train and evaluate the model under rotation and reflection.
```bash
python main.py \
    --pcam \
    --epochs 100 \
    --model-type gmr_resnet18  \
    --gmr-conv-size-list 9 9 5 5 \
    -b 64 \
    --lr 2e-2 \
    --sgd \
    --cudnn \
    --cos \
    --eval-rot \
    --eval-flip \
    --train-flip-p 0 \
    --log \
    --moco-aug \
    --translation \
    --translate-ratio 0.1 \
    --save-model
```

#### ImageNet-1k

Run the following command to train and evaluate the model under rotation and reflection.
```bash
python main.py \
    --imagenet \
    --epochs 100 
    -b 96 \
    --amp \
    --bf16 \
    --scaler \
    --model-type gmr_resnet50 \
    --gmr-conv-size-list 9 9 5 5 \
    --lr 1e-1 \
    --min-lr 1e-5 \
    --warm-up 2 \
    --cos \
    --sgd \
    --momentum 0.9 \
    --weight-decay 1e-4 \
    --moco-aug \
    --translation \
    --translate-ratio 0.2 \
    --eval-rot \
    --eval-flip \
    --train-flip-p 0 \
    --random-erase \
    --log \
    --save-model \
    --cudnn \
    --ddp \
    --world-size 4 \
    --log-interval 400 \
    --img-size 224
```

#### ModelNet10/40

Run the following command to train and evaluate the model under rotation and reflection.
```bash
python main.py \
    --modelnet10 \ # or use --modelnet40
    --epochs 100 \
    --model-type sri_r3d_9 \
    --gmr-conv-size-list 9 9 5 5 \
    -b 8 \
    --lr 2e-2 \
    --cos \
    --sgd \
    --eval-rot \
    --res-keep-conv1 \
    --log \
    --cudnn \
    --moco-aug \
    --res-inplanes 24 \
    --save-model \
    --save-best
```

For more details about the training arguments, please see `opt.py`.


## Reference

```
@article{du2025gmr,
    title={GMR-Conv: An Efficient Rotation and Reflection Equivariant Convolution Kernel Using Gaussian Mixture Rings},
    author={Du, Yuexi and Zhang, Jiazhen and Dvornek, Nicha C and Onofrey, John A},
    journal={arXiv preprint arXiv:2504.02819},
    year={2025}
}
```
