# main.py
"""
Objective: denoise CT images

Preconditions:
--needs a directory called output_samples existing in order to execute

Postconditions:
--saves input LDCT, NDCT images along with output denoised images to PNG file in output_samples directory

"""
import time
import os
import pandas as pd
import tensorflow as tf
import pdb
from glob import glob
from PIL import Image
import numpy as np
from tensorflow.python.client import device_lib
from model import denoiser
from utils import *

# pdb.set_trace()

"""Returns flags"""

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', default=12, type=int, help='Batch size')
parser.add_argument('--logdir', default='logdir', help='Tensorboard log directory')

parser.add_argument("--ldct-train-file-path", default="/data/CT_data/images/ldct_7e4/train/*.flt", help='Path to the ldct_7e4 train images')
parser.add_argument("--ldct-test-file-path", default="/data/CT_data/images/ldct_7e4/test/*.flt", help='Path to the ldct_7e4 test images')

#parser.add_argument("--ldct-train-file-path", default="/data/CT_data/images/ldct_2e5/train/*.flt", help='Path to the ldct_2e5 train images')
#parser.add_argument("--ldct-test-file-path", default="/data/CT_data/images/ldct_2e5/test/*.flt", help='Path to the ldct_2e5 test images')

#parser.add_argument("--ldct-train-file-path", default="/data/CT_data/images/ldct_1e5/train/*.flt", help='Path to the ldct_1e5 train images')
#parser.add_argument("--ldct-test-file-path", default="/data/CT_data/images/ldct_1e5/test/*.flt", help='Path to the ldct_1e5 test images')

#parser.add_argument("--ldct-train-file-path", default="/data/CT_data/sparseview/sparseview_90/train/*.flt", help='Path to the sparseview_90 images')
#parser.add_argument("--ldct-test-file-path", default="/data/CT_data/sparseview/sparseview_90/test/*.flt", help='Path to the sparseview_90 images')

#parser.add_argument("--ldct-train-file-path", default="/data/CT_data/sparseview/sparseview_60/train/*.flt", help='Path to the sparseview_60 images')
#parser.add_argument("--ldct-test-file-path", default="/data/CT_data/sparseview/sparseview_60/test/*.flt", help='Path to the sparseview_60 images')

#parser.add_argument("--ldct-train-file-path", default="/data/CT_data/sparseview/sparseview_180/train/*.flt", help='Path to the sparseview_180 images')
#parser.add_argument("--ldct-test-file-path", default="/data/CT_data/sparseview/sparseview_180/test/*.flt", help='Path to the sparseview_180 images')

parser.add_argument('--ndct-train-file-path',dest='ndct_train_file_path', type=str, default='/data/CT_data/images/ndct/train/*.flt', help='Path to the ndct images')
parser.add_argument('--ndct-test-file-path',dest='ndct_test_file_path', type=str, default='/data/CT_data/images/ndct/test/*.flt', help='Path to the ndct images')

# parser.add_argument('--reg', type=float, default=0.01, help='L2 Regularizer Term')
# learning rate doesn't stay constant throughout the training
# parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate')
parser.add_argument('--num-epochs', default=50, type=int, help='Number of epochs to repeat the shuffle')

#parser.add_argument('--ckdir', default='models', help='Checkpoint directory')
parser.add_argument('--sample-dir', default='./output_samples', help='Sample directory')
parser.add_argument('--ckpt-dir', default='./checkpoints', help='Checkpoint directory')
parser.add_argument('--test_dir', dest='test_dir', default ='./test', help='test sample are saved here')
#parser.add_argument('--save_dir', dest='save_dir', default='./save', help='test sample are saved here')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')


#parser.add_argument('--phase', dest='phase', default='test', help='test')
parser.add_argument('--phase', dest='phase', default='train', help ='train')

# buffer size recommended to be equal to num of images
parser.add_argument('--buffer-size', default=3600, type=int, help='Buffer size to shuffle the dataset')

args = parser.parse_args()


def denoiser_train(denoiser, lr):
        ndct_train = sorted(glob(args.ndct_train_file_path))
        ldct_train = sorted(glob(args.ldct_train_file_path))

        ndct_eval_data = sorted(glob(args.ndct_test_file_path))
        ldct_eval_data = sorted(glob(args.ldct_test_file_path))

        denoiser.train(ndct_train, ldct_train, ndct_eval_data, ldct_eval_data, lr,
                    ckpt_dir=args.ckpt_dir, num_epochs = args.num_epochs,
                       sample_dir=args.sample_dir, buffer_size = args.buffer_size)

def denoiser_test(denoiser):
    ldct_list = sorted(glob(args.ldct_test_file_path))
    ndct_list = sorted(glob(args.ndct_test_file_path))
    denoiser.test(ldct_list, ndct_list, ckpt_dir=args.ckpt_dir, save_dir=args.test_dir)

# main method
def main(_):
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
    #if not os.path.exists(args.save_dir):
    #    os.makedirs(args.save_dir)

    if args.use_gpu:
        # added to control the gpu memory
        print("GPU\n")

        # uses the GPU #0 on linux machines
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]="1"
        print("\n\nExecution Environment:\n\n{}".format(device_lib.list_local_devices()))

        gpu_options =  tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.90)
        with tf.compat.v1.Session(config =  tf.compat.v1.ConfigProto(gpu_options = gpu_options)) as sess:
            model = denoiser(sess, args.batch_size)
            if args.phase == 'train':
                denoiser_train(model, lr = args.lr)
            elif args.phase == 'test':
                 denoiser_test(model)
            else:
                 print('[!]Unknown phase')
                 exit(0)
    else:
        print("CPU\n")
        with tf.compat.v1.Session() as sess:
            model = denoiser(sess, batch_size = args.batch_size)
            if args.phase == 'train':
                denoiser_train(model, lr = args.lr)
            elif args.phase == 'test':
                pdb.set_trace()
                denoiser_test(model)
            else:
                print('[!]Unknown phase')
                exit(0)

if __name__ == '__main__':
     tf.compat.v1.app.run()
