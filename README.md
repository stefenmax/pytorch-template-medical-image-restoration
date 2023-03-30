# Pytorch-template-medical-image-restoration
Perhaps the world most convenient pytorch template for medical image restoration
<div align="center">
    <img src="assets/icon.png"/>
    <h1><code>
    Pytorch Project Template
    </h1></code>
    <p>
        <img src="https://img.shields.io/github/license/stefenmax/pytorch-template-medical-image-restoration"/>
    </p>
</div>

**About PyTorch 2.0**
  * Now the master branch supports PyTorch 2.0 by default.
  * You only need to install the related environment torch2_0.yml file.

## Dependencies
* Python 3.6
* PyTorch >= 1.13.0
* numpy
* skimage
* imageio
* matplotlib
* tqdm
* tensorboardX
* wandb (Before using wandb, you may need to sign up your own account and revise entity name in trainer.py)

## Wandb demo
![wandb](https://github.com/stefenmax/pytorch-template-medical-image-restoration/blob/main/assets/wandb.gif)

## Feature

- TensorBoardX /[wandb](https://www.wandb.com/) support (suggest using wandb, tensorboard is not well tested)
- Training state and network checkpoint saving, loading
    - Training state includes not only network weights, but also optimizer, step, epoch.
    - Checkpoint includes the last and best one(based on calculation of PSNR). This could be used for inference. 
- Distributed Learning using Distributed Data Parallel is supported, can using multiple GPU for training
- Using scrpits to queue the tasks
- Allow for GPU usage on M1 mac
- Support npy and mat format

## Code Structure

- `assets` dir: Image resourses of `Pytorch Project Template`. You can remove this directory.
- `dataset` dir: dataloader and dataset codes are here. Also, put dataset in `meta` dir.
- `experiment` dir: Your experiment result will save here.
- `models` dir: Put your checkpoint here to easily test the network.
- `src` dir:
    - `data` dir : dataloader and dataset codes are here. 
    - `loss` dir is for loss function design.
    - `model` dir is for wrapping network architecture. **You can put your own network here. **
    - `trainer.py` file: this is for setting up and iterating epoch.

## Setup

### Install requirements

- python3 (3.6, 3.7, 3.8, 3.9, 3.10 is tested)
- Support the version of PyTorch(1.13), if you use older version of pytorch than that, may meet the error of GPU usage on Mac
- `conda env create -f torch1_13.yml` for install develop dependencies (this requires python 3.6 and above because of black)`


### Code init

`conda env create -f environment.yml` for install develop dependencies (this requires python 3.6 and above because of black)


## Train example code
### For png file
- `python main.py --template AAPM --save EDSR --scale 1 --reset --save_results --patch_size 64 --ext sep --n_GPUs 1 --data_range '1-10/11-12' --loss '1*L1' --dir_data '../dataset/' --batch_size 8 --epochs 100` --start_wandb
### For mat or npy file
- `using numpy
python main.py --template AAPM --save EDSR-test --scale 1 --reset --save_results --patch_size 64 --ext img --n_GPUs 1 --data_range '1-10/11-12' --loss '1*L1' --dir_data '../dataset_npy/' --batch_size 1 --epochs 100 --using_npy or --using_mat`

**Update log**
* Mar 30, 2023
  * Add FBPCONVNet and REDCNN to model dir
  * Make learning rate can decrease logarithmically
* Mar 18, 2023
  * Support torch2.0 and wandb
  * Add RIDNet to model dir


**Future scopes**
* Support dcm format data
## Inspired by

I referred [EDSR's official implementation](https://github.com/sanghyun-son/EDSR-PyTorch) when crafting this template, so don't be surprised if you find some code is similar.
