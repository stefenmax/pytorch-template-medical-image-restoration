



# DenseNet and Deconvolution (linux)

DD-Net always peformed similar to UNet and DnCNN. The training time was the longest out of other 2 models. It took aprox. 24 hours to train the network.

paper: https://ieeexplore.ieee.org/document/8331861
github: https://github.com/zzc623/DD_Net

Authors write "The DD-Net was trained by the Adam algorithm [54].
The learning rate was initially set at 10−4 and slowly decreased
continuously down to 10−5. The size of mini-batch was 5. DD-
Net was implemented using Tensorflow [55] on a personal
workstation with Intel Core i5-7400 CPU and 16GB RAM.
A GPU card (Nvidia GTX Titan X) accelerated the training
process. All the convolution and deconvolution filters were
initialized with random Gaussian distributions with zero mean
and 0.01 standard deviation."

![ddnet_paper](/Users/np/Desktop/ddnet/screenshots/ddnet_paper.png)

**Results**

The average PSNR and SSIM values over 354 test images are displayed in the table below. 

Learning rate 10^-4

40 epochs

| Low Dose Image | UNet                                                         | DnCNN                                               |                DDNet                |
| :------------- | :----------------------------------------------------------- | --------------------------------------------------- | :---------------------------------: |
| sparseview_60  | **Avg PSNR: 33.28	                        Avg SSIM: 0.8858** | Avg PSNR: 32.30               Avg SSIM: 0.8560      | Avg PSNR: 32.96	Avg SSIM: 0.8797 |
| sparseview_90  | **Avg PSNR: 35.42	                     Avg SSIM: 0.9038** | Avg PSNR: 35.13               Avg SSIM: 0.8892      | Avg PSNR: 35.29	Avg SSIM: 0.9011 |
| sparseview_180 | Avg PSNR: 39.48	                    Avg SSIM: 0.9319      | **Avg PSNR: 39.77               Avg SSIM: 0.9341**  | Avg PSNR: 39.55	Avg SSIM: 0.9322 |
| ldct_7e4       | Avg PSNR: 41.78	                     Avg SSIM: 0.9429     | **Avg PSNR: 42.00	            Avg SSIM: 0.9444** | Avg PSNR: 41.84	Avg SSIM: 0.9431 |
| ldct_1e5       | Avg PSNR: 42.11	                    Avg SSIM: 0.9441      | **Avg PSNR: 42.32	            Avg SSIM: 0.9456** | Avg PSNR: 42.23	Avg SSIM: 0.9448 |
| ldct_2e5       | Avg PSNR: 42.69	                     Avg SSIM: 0.9466     | **Avg PSNR: 42.87	            Avg SSIM: 0.9477** | Avg PSNR: 42.77	Avg SSIM: 0.9471 |



```bash
def denseblock(input):
  # - L1
  num_filters = 16
  d2_1 = BatchNormalization()(input)
  d2_1 = Activation('relu')(d2_1)
  d2_1 = Conv2D(num_filters*4, (1, 1), padding='same', use_bias=True, strides=(1, 1))(d2_1)

  d2_1 = BatchNormalization()(d2_1)
  d2_1 = Activation('relu')(d2_1)
  d2_1 = Conv2D(num_filters, (5, 5), padding='same', use_bias=True, strides=(1, 1))(d2_1)

  d2_1 = concatenate([input, d2_1])

  # - L2
  d2_2 = BatchNormalization()(d2_1)
  d2_2 = Activation('relu')(d2_2)
  d2_2 = Conv2D(num_filters*4, (1, 1), padding='same', use_bias=True, strides=(1, 1))(d2_2)

  d2_2 = BatchNormalization()(d2_2)
  d2_2 = Activation('relu')(d2_2)
  d2_2 = Conv2D(num_filters, (5, 5), padding='same', use_bias=True, strides=(1, 1))(d2_2)

  d2_2 = concatenate([input, d2_1, d2_2])

  # - L3
  d2_3 = BatchNormalization()(d2_2)
  d2_3 = Activation('relu')(d2_3)
  d2_3 = Conv2D(num_filters*4, (1, 1), padding='same', use_bias=True, strides=(1, 1))(d2_3)

  d2_3 = BatchNormalization()(d2_3)
  d2_3 = Activation('relu')(d2_3)
  d2_3 = Conv2D(num_filters, (5, 5), padding='same', use_bias=True, strides=(1, 1))(d2_3)

  d2_3 = concatenate([input, d2_1, d2_2, d2_3])

  # - L4
  d2_4 = BatchNormalization()(d2_3)
  d2_4 = Activation('relu')(d2_4)
  d2_4 = Conv2D(num_filters*4, (1, 1), padding='same', use_bias=True, strides=(1, 1))(d2_4)

  d2_4 = BatchNormalization()(d2_4)
  d2_4 = Activation('relu')(d2_4)
  d2_4 = Conv2D(num_filters, (5, 5), padding='same', use_bias=True, strides=(1, 1))(d2_4)

  d2_4 = concatenate([input, d2_1, d2_2, d2_3, d2_4])
  return d2_4

def dd_net(ldct_img, is_training= True):
    net = ldct_img
    num_filter = 16
    # ---A1 Layer-----------------------
    h_conv1 = Conv2D(16, (7, 7), padding='same', use_bias=True, strides=(1, 1))(net)
    a1 = MaxPooling2D((3, 3), strides=(2, 2), padding='same') (h_conv1)
    # images 256 X 256
    d1 = denseblock(a1)
    
    a1 = BatchNormalization()(d1)
    a1 = Activation('relu')(a1)
    h_conv1_T = Conv2D(16, (1, 1), strides=(1, 1), use_bias=True) (a1)
    
    # ----A2 Layer---------------------
    a2 = MaxPooling2D((2, 2),strides=(2, 2), padding='same') (h_conv1_T)
    # images 128 X 128 d
    d2 = denseblock(a2)
    
    a2 = BatchNormalization()(d2)
    a2 = Activation('relu')(a2)
    h_conv2_T = Conv2D(16, (1, 1), strides=(1, 1), use_bias=True) (a2)
    # images 128 X 128
    
    # # ----A3 Layer----------------------
    a3 = MaxPooling2D((2, 2), strides=(2, 2), padding='same') (h_conv2_T)
    # images 64 X 64
    d3 = denseblock(a3)
    
    a3 = BatchNormalization()(d3)
    a3 = Activation('relu')(a3)
    h_conv3_T = Conv2D(16, (1, 1), strides=(1, 1), use_bias=True) (a3)
    
    # ----A4 Layer----------------------
    a4 = MaxPooling2D((2, 2), strides=(2, 2), padding='same') (h_conv3_T)
    # images 32 X 3
    d4 = denseblock(a4)
    
    a4 = BatchNormalization()(d4)
    a4 = Activation('relu')(a4)
    h_conv4_T = Conv2D(16, (1, 1), strides=(1, 1), use_bias=True) (a4)
    
    # #----B1 Layer-----------------------
    b1 = UpSampling2D((2, 2), interpolation="nearest") (h_conv4_T)
    # images 64 X 64
    b1 = concatenate([b1, h_conv3_T])
    
    b1 = Conv2DTranspose(num_filter*2, (5, 5), padding='same', strides=(1, 1)) (b1)
    b1 = Activation('relu')(b1)
    b1 = BatchNormalization()(b1)
    
    b1 = Conv2DTranspose(16, (1, 1), padding='same', strides=(1, 1)) (b1)
    b1 = Activation('relu')(b1)
    b1 = BatchNormalization()(b1)
    
    # #----B2 Layer-----------------------
    b2 = UpSampling2D((2, 2), interpolation="nearest") (b1)
    # images 128 X 128
    b2 = concatenate([b2, h_conv2_T])
    
    b2 = Conv2DTranspose(num_filter*2, (5, 5), padding='same', strides=(1, 1)) (b2)
    b2 = Activation('relu')(b2)
    b2 = BatchNormalization()(b2)
    
    b2 = Conv2DTranspose(16, (1, 1), padding='same', strides=(1, 1)) (b2)
    b2 = Activation('relu')(b2)
    b2 = BatchNormalization()(b2)
    
    #----B3 Layer------------------------conv6
    b3 = UpSampling2D((2, 2),interpolation="nearest") (b2)
    # images 256 X 256
    b3 = concatenate([b3, h_conv1_T])
    
    b3 = Conv2DTranspose(num_filter*2, (5, 5), padding='same', strides=(1, 1)) (b3)
    b3 = Activation('relu')(b3)
    b3 = BatchNormalization()(b3)
    
    b3 = Conv2DTranspose(16, (1, 1), padding='same', strides=(1, 1)) (b3)
    b3 = Activation('relu')(b3)
    b3 = BatchNormalization()(b3)
    
    #----B4 Layer-------------------------
    b4 = UpSampling2D((2, 2),interpolation="nearest") (b3)
    # images 512 X 512
    b4 = concatenate([b4, h_conv1])
    b4 = Conv2DTranspose(num_filter*2, (5, 5),padding='same', strides=(1, 1)) (b4)
    b4 = Activation('relu')(b4)
    # b4 = BatchNormalization()(b4)
    
    output_img = Conv2DTranspose(1, (1, 1), strides=(1, 1)) (b4)
    # output_img = Activation('relu')(output_img) # in paper but DIDN'T CONVERGE
    # ------ end B4 layer
    
    denoised_image = Subtract()([net, output_img])
    return denoised_image


```

Each training process took approximately 24 hours.

#### How to create a python environment and train

```
(base) [npovey@ka ~]$ conda create -n keras-gpu python=3.6 numpy scipy keras-gpu
(base) [npovey@ka unet4]$ conda activate keras-gpu
(keras-gpu) [npovey@ka unet4]$ pip install pandas
(keras-gpu) [npovey@ka unet4]$ pip install Pillow
(keras-gpu) [npovey@ka unet4]$ pip install matplotlib
(keras-gpu) [npovey@ka unet4]$ python main.py 
```

#### How to train 

In main.py  (line 77)  use train.

```python
#parser.add_argument('--phase', dest='phase', default='test', help='test')
parser.add_argument('--phase', dest='phase', default='train', help ='train')
```

#### How to test

Change from train to test in main.py  line 77

```python
#parser.add_argument('--phase', dest='phase', default='test', help='test')
parser.add_argument('--phase', dest='phase', default='train', help ='train')
```

#### Other useful info

Sign in into remote linux machines

```bash
nps-MacBook-Air-2:Desktop np$ ssh npovey@ka...
```

```bash
[npovey@ka dd_net]$ ls
main.py  model.py  model.py~  utils.py
```

 Got  core dumped problem as all GPUs were taken

```bash
Aborted (core dumped)
(keras-gpu) [npovey@ka dd_net]$ 
```

Check available GPUs

```bash
(keras-gpu) [npovey@ka dd_net]$ nvidia-smi -L
GPU 0: Quadro RTX 5000 (UUID: GPU-1f923d52-ea64-f463-96a4-3bece2719a8b)
GPU 1: Quadro RTX 5000 (UUID: GPU-20947179-1907-5468-1325-a3fc16f5a54e)
(keras-gpu) [npovey@ka dd_net]$ nvidia-smi
Fri Mar  6 14:25:38 2020       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.87.00    Driver Version: 418.87.00    CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Quadro RTX 5000     Off  | 00000000:67:00.0 Off |                  Off |
| 43%   68C    P2   212W / 230W |  15871MiB / 16095MiB |     94%      Default |
+-------------------------------+----------------------+----------------------+
|   1  Quadro RTX 5000     Off  | 00000000:68:00.0 Off |                  Off |
| 55%   78C    P2   221W / 230W |  15869MiB / 16092MiB |     96%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0     78799      C   python                                     15861MiB |
|    1     26617      C   python                                     15859MiB |
+-----------------------------------------------------------------------------+
WARNING: infoROM is corrupted at gpu 0000:67:00.0
(keras-gpu) [npovey@ka dd_net]$ 

```

Looks like both GPUs are taken

Trying a little bit later 

```bash
(base) [npovey@ka ~]$ nvidia-smi
Fri Mar  6 20:23:45 2020       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.87.00    Driver Version: 418.87.00    CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Quadro RTX 5000     Off  | 00000000:67:00.0 Off |                  Off |
| 43%   65C    P2    74W / 230W |  15871MiB / 16095MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   1  Quadro RTX 5000     Off  | 00000000:68:00.0 Off |                  Off |
| 35%   35C    P0    N/A /  N/A |      0MiB / 16092MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0     78799      C   python                                     15861MiB |
+-----------------------------------------------------------------------------+
WARNING: infoROM is corrupted at gpu 0000:67:00.0
(base) [npovey@ka ~]$ 
```

We can observe that GPU1 is free to run our code.

Run the code on the remote linux machine

```bash
[npovey@ka data]$ cd dd_net/
[npovey@ka dd_net]$ source activate keras-gpu
(keras-gpu) [npovey@ka dd_net]$ 
(keras-gpu) [npovey@ka dd_net]$ python main.py
```

As the code takes long time to run I recommend running it using screen

```bash

[npovey@ka dd_net]$ screen -S dd_net1
[npovey@ka dd_net]$ source activate keras-gpu
[npovey@ka dd_net]$ python main.py
(press ctlr+A, D to detach from screen)
[detached from 6921.dd_net1]

```

To copy result to a local machine and view them 

```bash
nps-MacBook-Air-2:Desktop np$ scp -r npovey@ka:/home/npovey/data/dd_net/output_dd_net_60.txt DD_Net/
npovey@ka's password: 
output_dd_net_60.txt                          100%   32KB 321.5KB/s   00:00    
nps-MacBook-Air-2:Desktop np$ 


```

##### run sparseview_60

```bash
[npovey@ka dd_net]$ screen -S dd_net1
[npovey@ka dd_net]$ source activate keras-gpu
(keras-gpu) [npovey@ka dncnn1]$ python main.py > output_ddnet_60.txt
```

##### run sparseview_90

```bash
[npovey@ka dd_net]$ screen -S dd_net1
[npovey@ka dd_net]$ source activate keras-gpu
(keras-gpu) [npovey@ka dd_net]$ python main.py > output_ddnet_90.txt

```

Got good results:

##### run sparseview_180

```bash
[npovey@ka dd_net]$ screen -S dd_net1
[npovey@ka dd_net]$ source activate keras-gpu
(keras-gpu) [npovey@ka dd_net]$ python main.py > output_ddnet_180.txt
```

##### run ldcd_7e4

```bash
[npovey@ka dd_net]$ screen -S dd_net1
[npovey@ka dd_net]$ source activate keras-gpu
(keras-gpu) [npovey@ka dd_net]$ python main.py > output_ddnet_ldct_7e4.txt
```

Epoch 1/50 Avg PSNR: 40.91373649562122

Epoch 2/50 Avg PSNR: 41.007181141963734

Epoch 3/50 Avg PSNR: 32.428249836375585

Epoch 4/50 Avg PSNR: 32.42561434856395

......

Epoch 50/50 Avg PSNR: 32.427569692000205



#####run ldcd_2e5

```bash
[npovey@ka dd_net]$ screen -S dd_net1
[npovey@ka dd_net]$ source activate keras-gpu
(keras-gpu) [npovey@ka dd_net]$ python main.py > output_ddnet_ldct_2e5.txt
```

Epoch 0: Avg PSNR: 41.967376293118136

Epoch 1: Avg PSNR: 37.859690812973035

Epoch 2: Avg PSNR: 39.812348874868206

Epoch 9: Avg PSNR: 41.398334420237376

##### Delete old data before training on a new dataset

```bash
ps aux >text.txt
[npovey@ka dd_net]$ kill -KILL 10371
[npovey@ka dd_net]$ rm -r logs/
[npovey@ka dd_net]$ rm -r checkpoints
[npovey@ka dd_net]$ rm -r ndct_train.tfrecord 
[npovey@ka dd_net]$ rm -r ldct_train.tfrecord 
[npovey@ka dd_net]$ rm -r ndct_test.tfrecord 
[npovey@ka dd_net]$ rm -r ldct_test.tfrecord 
[npovey@ka dd_net]$ rm -r __pycache__/
[npovey@ka dd_net]$ rm -r output_samples/
[npovey@ka dd_net]$ rm -r test/

[npovey@ka dd_net]$ conda activate keras-gpu
(keras-gpu) [npovey@ka dd_net]$ python main.py > output_ddnet_60_2.txt

```

##### Train

```bash
[npovey@ka dd_net]$ screen -S dd_net1
[npovey@ka dd_net]$ conda activate keras-gpu
(keras-gpu) [npovey@ka dncnn1]$ python main.py > output_ddnet_60.txt
```

```bash
[npovey@ka ddnet2]$ screen -S dd_net1
[npovey@ka ddnet2]$ conda activate keras-gpu
(keras-gpu) [npovey@ka ddnet2]$ python main.py > output_ddnet_ldct_7e4.txt
```

copy to local desktop

```bash
nps-MacBook-Air-2:Desktop np$ scp -r npovey@ka:/home/npovey/data/ddnet2/main.py .
npovey@ka's password: 
main.py                                       100% 7690   305.0KB/s   00:00    
nps-MacBook-Air-2:Desktop np$ scp -r npovey@ka:/home/npovey/data/ddnet2/model.py .
npovey@ka's password: 
model.py                                                                                                               100%   33KB 562.2KB/s   00:00    
(base) nps-MacBook-Air-2:Desktop np$ 

```

```bash
[npovey@ka ddnet2]$ screen -S dd_net1
[npovey@ka ddnet2]$ conda activate keras-gpu
(keras-gpu) [npovey@ka ddnet2]$ python main.py > output_ddnet_ldct_2e5.txt
```

```bash
[npovey@ka ddnet2]$ screen -S dd_net1
[npovey@ka ddnet2]$ conda activate keras-gpu
(keras-gpu) [npovey@ka ddnet2]$ python main.py > output_ddnet_ldct_1e5.txt
```

```bash
[npovey@ka ddnet2]$ screen -S dd_net1
[npovey@ka ddnet2]$ conda activate keras-gpu
(keras-gpu) [npovey@ka ddnet2]$ python main.py > output_ddnet_ldct_2e5.txt
```

```bash
[npovey@ka ddnet2]$ screen -S dd_net1
[npovey@ka ddnet2]$ conda activate keras-gpu
(keras-gpu) [npovey@ka ddnet2]$ python main.py > output_ddnet_sparseview_60.txt
CTRL+A D exit screens mode
```

```bash
[npovey@ka ddnet2]$ screen -S dd_net1
[npovey@ka ddnet2]$ conda activate keras-gpu
(keras-gpu) [npovey@ka ddnet2]$ python main.py > output_ddnet_sparseview_90.txt
CTRL+A D exit screens mode
```

```bash
[npovey@ka ddnet2]$ screen -S dd_net1
[npovey@ka ddnet2]$ conda activate keras-gpu
(keras-gpu) [npovey@ka ddnet2]$ python main.py > output_ddnet_sparseview_180.txt
CTRL+A D exit screens mode
```

##### To test 

Change from train to test in main.py  line 77

```python
#parser.add_argument('--phase', dest='phase', default='test', help='test')
parser.add_argument('--phase', dest='phase', default='train', help ='train')

```

```bash
[npovey@ka ddnet2]$ conda activate keras-gpu
(keras-gpu) [npovey@ka ddnet2]$ python main.py > test_180.txt
```

```bash
[npovey@ka ddnet2]$ conda activate keras-gpu
(keras-gpu) [npovey@ka ddnet2]$ python main.py > test_90.txt
```

```bash
[npovey@ka ddnet2]$ conda activate keras-gpu
(keras-gpu) [npovey@ka ddnet2]$ python main.py > test_7e4.txt
```

```bash
[npovey@ka ddnet2]$ conda activate keras-gpu
(keras-gpu) [npovey@ka ddnet2]$ python main.py > test_1e5.txt
```

```bash
[npovey@ka ddnet2]$ conda activate keras-gpu
(keras-gpu) [npovey@ka ddnet2]$ python main.py > test_2e5.txt
```



