# pytorch-template-medical-image-restoration
Perhaps the world most convient pytorch template for medical image restoration
<div align="center">
    <img src="assets/icon.png"/>
    <h1><code>
        Pytorch Project Template
    </h1></code>
    <p>
        <img src="https://img.shields.io/github/license/stefenmax/pytorch-template-medical-image-restoration"/>
    </p>
</div>

## Feature

- TensorBoardX support
- Training state and network checkpoint saving, loading
    - Training state includes not only network weights, but also optimizer, step, epoch.
    - Checkpoint includes the last and best one(based on calculation of PSNR). This could be used for inference. 
- Distributed Learning using Distributed Data Parallel is supported, can using multiple GPU for training
- Using scrpits to queue the tasks
- Allow for GPU usage on M1 mac

## Code Structure

- `assets` dir: icon image of `Pytorch Project Template`. You can remove this directory.
- `config` dir: directory for config files
- `dataset` dir: dataloader and dataset codes are here. Also, put dataset in `meta` dir.
- `model` dir: `model.py` is for wrapping network architecture. `model_arch.py` is for coding network architecture.
- `tests` dir: directory for `pytest` testing codes. You can check your network's flow of tensor by fixing `tests/model/net_arch_test.py`. 
Just copy & paste `Net_arch.forward` method to  `net_arch_test.py` and add `assert` phrase to check tensor.
- `utils` dir:
    - `train_model.py` and `test_model.py` are for train and test model once.
    - `utils.py` is for utility. random seed setting, dot-access hyper parameter, get commit hash, etc are here. 
    - `writer.py` is for writing logs in tensorboard / wandb.
- `trainer.py` file: this is for setting up and iterating epoch.

## Setup

### Install requirements

- python3 (3.6, 3.7, 3.8 is tested)
- Support the latest version of PyTorch
- `conda env create -f environment.yml` for install develop dependencies (this requires python 3.6 and above because of black)`

### Config

- Config is written in option file
- `name` is train name you run.
- `working_dir` is root directory for saving checkpoints, logging logs.
- `device` is device mode for running your model. You can choose `cpu` or `cuda`
- `data` field
    - Configs for Dataloader.
    - glob `train_dir` / `test_dir` with `file_format` for Dataloader.
    - If `divide_dataset_per_gpu` is true, origin dataset is divide into sub dataset for each gpu. 
    This could mean the size of origin dataset should be multiple of number of using gpu.
    If this option is false, dataset is not divided but epoch goes up in multiple of number of gpus.
- `train`/`test` field
    - Configs for training options.
    - `random_seed` is for setting python, numpy, pytorch random seed.
    - `num_epoch` is for end iteration step of training.
    - `optimizer` is for selecting optimizer. Only `adam optimizer` is supported for now.
    - `dist` is for configuring Distributed Data Parallel.
        - `gpus` is the number that you want to use with DDP (`gpus` value is used at `world_size` in DDP).
        Not using DDP when `gpus` is 0, using all gpus when `gpus` is -1.
        - `timeout` is seconds for timeout of process interaction in DDP.
        When this is set as `~`, default timeout (1800 seconds) is applied in `gloo` mode and timeout is turned off in `nccl` mode.
- `model` field
    - Configs for Network architecture and options for model.
    - You can add configs in yaml format to config your network.
- `log` field
    - Configs for logging include tensorboard / wandb logging. 
    - `summary_interval` and `checkpoint_interval` are interval of step and epoch between training logging and checkpoint saving.
    - checkpoint and logs are saved under `working_dir/chkpt_dir` and `working_dir/trainer.log`. Tensorboard logs are saving under `working_dir/outputs/tensorboard`
- `load` field
    - loading from wandb server is supported
    - `wandb_load_path` is `Run path` in overview of run. If you don't want to use wandb load, this field should be `~`.
    - `network_chkpt_path` is path to network checkpoint file.
    If using wandb loading, this field should be checkpoint file name of wandb run.
    - `resume_state_path` is path to training state file.
    If using wandb loading, this field should be training state file name of wandb run.

### Code lint

1. `conda env create -f environment.yml` for install develop dependencies (this requires python 3.6 and above because of black)


## Train
### For png file
- `python main.py --template AAPM --save EDSR --scale 1 --reset --save_results --patch_size 64 --ext sep --n_GPUs 1 --data_range '1-1923/1924-1933' --loss '1*L1' --dir_data '../dataset/' --batch_size 8 --epochs 100`
### For mat or npy file
- `using numpy
python main.py --template AAPM --save EDSR-test --scale 1 --reset --save_results --patch_size 64 --ext img --n_GPUs 1 --data_range '1-5/6-7' --loss '1*L1' --dir_data '../dataset_npy/' --batch_size 1 --epochs 100 --using_npy or --using_mat`

## Inspired by

I referred [EDSR's official implementation](https://github.com/sanghyun-son/EDSR-PyTorch) when crafting this template, so don't be surprised if you find some code is similar.
