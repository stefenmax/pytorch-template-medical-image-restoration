# model.py
# U-Net architecture
# Y_ - original image, clean_image [ndct: normal dose ct]
# Y - denoised image  [Y = X - noise]
# X - noisy image [ldct: low dose ct]
# noise - our model learns noise [residual image]

import tensorflow as tf
import pdb
import os
import numpy as np
import zipfile
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
import sys
from PIL import Image
from utils import *
from shutil import copyfile
from keras.layers.convolutional import Conv2DTranspose
from keras.layers import concatenate
from keras.layers import Concatenate
from keras.layers import Conv2D, MaxPooling2D, Input, Dense
from keras.layers import Subtract,Activation,Cropping2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from glob import glob
from keras.layers import UpSampling2D, Dropout
from keras import optimizers
# pdb.set_trace()

psnr_set = set([0]) #global variable
count = 0

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




def unet2(ldct_img, is_training= True):
    """
    Defines the layer configurations and parameters in the contracting,
    expanding paths and produces the residual mapping as output
    """
    net = ldct_img
    c1 = Conv2D(8, (3, 3), padding='same') (net)
    #c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    c1 = Conv2D(8, (3, 3), padding='same', use_bias=False) (c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(16, (3, 3), padding='same', use_bias=False) (p1)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    c2 = Conv2D(16, (3, 3), padding='same', use_bias=False) (c2)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(32, (3, 3), padding='same', use_bias=False) (p2)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    c3 = Conv2D(32, (3, 3), padding='same', use_bias=False) (c3)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(64, (3, 3), padding='same', use_bias=False) (p3)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)
    c4 = Conv2D(64, (3, 3), padding='same', use_bias=False) (c4)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(128, (3, 3), padding='same', use_bias=False) (p4)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)
    c5 = Conv2D(128, (3, 3), padding='same', use_bias=False) (c5)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(64, (3, 3), padding='same', use_bias=False) (u6)
    c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6)
    c6 = Conv2D(64, (3, 3), padding='same', use_bias=False) (c6)
    c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), padding='same',use_bias=False) (u7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)
    c7 = Conv2D(32, (3, 3), padding='same', use_bias=False) (c7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), padding='same', use_bias=False) (u8)
    c8 = BatchNormalization()(c8)
    c8 = Activation('relu')(c8)
    c8 = Conv2D(16, (3, 3), padding='same', use_bias=False) (c8)
    c8 = BatchNormalization()(c8)
    c8 = Activation('relu')(c8)

    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(8, (3, 3), padding='same', use_bias=False) (u9)
    c9 = BatchNormalization()(c9)
    c9 = Activation('relu')(c9)
    c9 = Conv2D(8, (3, 3), padding='same', use_bias=False) (c9)
    c9 = BatchNormalization()(c9)
    c9 = Activation('relu')(c9)
    output_img = Conv2D(1, (1, 1), padding='same') (c9)
    denoised_image = Subtract()([net, output_img])
    return denoised_image


def unet_original(ldct_img, is_training= True):
    """
    Defines the layer configurations and parameters in the contracting,
    expanding paths and produces the residual mapping as output
    """
    net = ldct_img
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (net)
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)
    output_img = Conv2D(1, (3, 3), padding='same') (c9)
    denoised_image = keras.layers.Subtract()([net, output_img])

    return denoised_image

def dncnn(ldct_img, is_training= True):
    net = ldct_img
    # first convolution doesn't have batch normalization
    c1 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu') (net)
    
    c1 = Conv2D(filters=64, kernel_size=3, padding='same', use_bias = False) (c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    
    c1 = Conv2D(filters=64, kernel_size=3, padding='same', use_bias = False) (c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)

    c1 = Conv2D(filters=64, kernel_size=3, padding='same' , use_bias = False) (c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)

    c1 = Conv2D(filters=64, kernel_size=3, padding='same', use_bias = False) (c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)

    c1 = Conv2D(filters=64, kernel_size=3, padding='same', use_bias = False) (c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1) 
   
    c1 = Conv2D(filters=64, kernel_size=3, padding='same', use_bias = False) (c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)

    c1 = Conv2D(filters=64, kernel_size=3, padding='same', use_bias = False) (c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)    
    
    c1 = Conv2D(filters=64, kernel_size=3, padding='same', use_bias = False) (c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)

    c1 = Conv2D(filters=64, kernel_size=3, padding='same', use_bias = False) (c1)    
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)    

    c1 = Conv2D(filters=64, kernel_size=3, padding='same', use_bias = False) (c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)

    c1 = Conv2D(filters=64, kernel_size=3, padding='same', use_bias = False) (c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)

    c1 = Conv2D(filters=64, kernel_size=3, padding='same', use_bias = False) (c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    
    c1 = Conv2D(filters=64, kernel_size=3, padding='same', use_bias = False) (c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    
    c1 = Conv2D(filters=64, kernel_size=3, padding='same', use_bias = False) (c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)

    c1 = Conv2D(filters=64, kernel_size=3, padding='same', use_bias = False) (c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    
    output_img = Conv2D(filters=1, kernel_size=3, padding='same') (c1)
    denoised_image = keras.layers.Subtract()([net, output_img])
    return denoised_image

def create_dataset(CT_type_TFRecord, seed=1):
    """
    Extracts TFRecords and converts them to usable tensor format
    Generates patched training data if desired (comment/uncomment last 2 lines in this method)
    """

    def parse_fn(record):
        """extracts single TFRecord and parses its feature dictionary"""
        features = {
            'filename': tf.FixedLenFeature([], tf.string),
            'rows': tf.FixedLenFeature([], tf.int64),
            'cols': tf.FixedLenFeature([], tf.int64),
            'channels': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
        }
        example= tf.parse_single_example(record, features)   #bind empty feature dictionary with extracted TFRecord
        CT_img= tf.decode_raw(example['image'], out_type=tf.float32)
        CT_img= tf.reshape(CT_img, [512,512])
        CT_img= tf.expand_dims(CT_img, axis=-1)

        img_shape= tf.stack([example['rows'], example['cols'], example['channels']])
        filename= example['filename']
        return CT_img

    # def create_patches(record, seed=1):
    #     patches= []
    #     temp_record= tf.image.crop_to_bounding_box(record, 0, 100, 512, 412)       # reducing sample area of patches (getting rid of empty space on left of 512*512 img)
    #     for i in range(500):
    #         patch= tf.random_crop(temp_record, [128,128,1], seed=seed) #### RANDOM CROP IS A BIG PROBLEM!! Will cause mismatch in ldct/ndct data...
    #         patches.append(patch)
    #     patches= tf.stack(patches)
    #     assert patches.get_shape().dims == [500, 128, 128, 1]
    #     return patches
    if CT_type_TFRecord[-14:-9] == 'valdn':
        return tf.data.TFRecordDataset(CT_type_TFRecord).map(parse_fn)

    #--use this for patch-based learning--
    #return tf.data.TFRecordDataset(CT_type_TFRecord).map(parse_fn).map(create_patches).apply(tf.data.experimental.unbatch())

    #--use this for full-image learning--
    return tf.data.TFRecordDataset(CT_type_TFRecord).map(parse_fn)

def generateTFRecords(filenames, CT_type):
    """
    takes input binary CT image files and writes them out as TFRecords
    """

    def convert_img(file):
        with tf.io.gfile.GFile(file, 'rb') as fid:
            image_data= fid.read()
        filename= os.path.basename(file)
        example= tf.train.Example(features= tf.train.Features(feature= {
                  'filename': tf.train.Feature(bytes_list = tf.train.BytesList(value = [filename.encode('utf-8')])),
                  'rows': tf.train.Feature(int64_list = tf.train.Int64List(value = [512])),
                  'cols': tf.train.Feature(int64_list = tf.train.Int64List(value = [512])),
                  'channels': tf.train.Feature(int64_list = tf.train.Int64List(value = [1])),
                  'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [image_data]))
        }))
        return example
    with tf.io.TFRecordWriter('{}.tfrecord'.format(CT_type)) as writer:
        for f in filenames:
            example = convert_img(f)
            writer.write(example.SerializeToString())

def get_loss(y_pred, y_true, batch_size):
        """
        Returns the l2 loss
        """
        loss = (1.0 / batch_size) * tf.nn.l2_loss(y_true - y_pred)
        return loss

# ------------------------ denoiser object -----------------------------------#
class denoiser(object):
    def __init__(self, sess, batch_size, input_c_dim=1):
        self.sess = sess
        self.input_c_dim = input_c_dim
        self.batch_size = batch_size
        # build model
        self.Y_ = tf.compat.v1.placeholder(tf.float32, [None, None, None, self.input_c_dim])
        self.is_training = tf.compat.v1.placeholder(tf.bool, name='is_training')
        self.X = tf.compat.v1.placeholder(tf.float32, [None, None, None, self.input_c_dim])
        #self.Y = dncnn(self.X, is_training=self.is_training)
        self.Y = dd_net(self.X, is_training=self.is_training)
 
        self.loss = (1.0 / batch_size) * tf.nn.l2_loss(self.Y_ - self.Y)
        self.lr = tf.compat.v1.placeholder(tf.float32, name='learning_rate')
        self.eva_psnr = tf_psnr(self.Y, self.Y_)
        optimizer = tf.compat.v1.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
               self.train_op = optimizer.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")

    # --------------------------------train()---------------------------------*
    def train(self, ndct_train, ldct_train, ndct_eval_data, ldct_eval_data, lr, ckpt_dir, num_epochs, sample_dir, buffer_size):
        print("len(ndct_train)", len(ndct_train))
        #generate training data from ldct and ndct datasets
        if not os.path.exists('ldct_train.tfrecord'):
            generateTFRecords(ldct_train, 'ldct_train')
        if not os.path.exists('ndct_train.tfrecord'):
            generateTFRecords(ndct_train, 'ndct_train')

        #create datasets using training data
        ldct_train_dataset = create_dataset('ldct_train.tfrecord')
        ndct_train_dataset = create_dataset('ndct_train.tfrecord')
        print(ldct_train_dataset)
        print(ndct_train_dataset)

        num = num_epochs * 8
        print("num_epochs: ", num_epochs)
        #print("num", num)
        print("self.batch_size", self.batch_size)
        #train_dataset = tf.data.Dataset.zip((ldct_train_dataset, ndct_train_dataset)).repeat(num_epochs).shuffle(buffer_size).batch(self.batch_size)
        train_dataset = tf.data.Dataset.zip((ldct_train_dataset, ndct_train_dataset)).repeat(num).shuffle(buffer_size).batch(self.batch_size)
        print("len(train_dataset)",train_dataset)
        # summary_op = tf.summary.merge_all()
        iterator = train_dataset.make_initializable_iterator()

        next_element = iterator.get_next()
        self.sess.run(iterator.initializer)
        epoch = 0

        # load existing model if it exists
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            epoch = global_step+1
            print("[*] Model restore success!")
        else:
            epoch = 0
            print("[*] Not find pretrained model!")
        while True:
            try:
                # we're training the network on the training dataset
                num_batches = len(ndct_train) / self.batch_size # number of images used 3600

                #-----------------------------------------
                # learning rate is cut in half every 10 epoch
                #-----------------------------------------
                learning_rate = lr/(2**(epoch/10))


                for i in range(0, int(num_batches)):
                    print("batch: {}/{}".format(i+1,num_batches))
                    #-------------------------
                    # self.Y:  denoised image
                    # self.Y_: clean image
                    # self.X:  low dose image
                    #-------------------------
                    ldct_img, ndct_img = self.sess.run(next_element)
                    _, step_loss = self.sess.run([self.train_op, self.loss],
                                                 feed_dict={self.Y_: ndct_img, self.X: ldct_img,
                                                             self.lr: learning_rate, self.is_training : True})
                    print("Epoch: {}/{}\tLoss: {}\n".format(epoch, num_epochs, step_loss / self.batch_size))

                    # --------------------------------
                    # augment image and add to training
                    # ---------------------------------

                    ## mode 1: flipud
                    ldct_img, ndct_img = self.sess.run(next_element)
                    ldct_img_1 = data_augmentation(ldct_img, 1)
                    ndct_img_1 = data_augmentation(ndct_img, 1)
                    _, step_loss = self.sess.run([self.train_op, self.loss],
                                                 feed_dict={self.Y_ : ndct_img_1, self.X : ldct_img_1,
                                                            self.lr: learning_rate, self.is_training : True})
                    print("Epoch: {}/{}\tLoss: {}\n".format(epoch, num_epochs, step_loss / self.batch_size))

                    # # mode 2: rotate 90
                    ldct_img, ndct_img = self.sess.run(next_element)
                    ldct_img_2 = data_augmentation(ldct_img, 2)
                    ndct_img_2 = data_augmentation(ndct_img, 2)
                    _, step_loss = self.sess.run([self.train_op, self.loss],
                                                 feed_dict={self.Y_ : ndct_img_2, self.X : ldct_img_2,
                                                            self.lr: learning_rate, self.is_training : True})
                    print("Epoch: {}/{}\tLoss: {}\n".format(epoch, num_epochs, step_loss / self.batch_size))
                    
                    # # mode 3: rotate 90 and flipud
                    ldct_img, ndct_img = self.sess.run(next_element)
                    ldct_img_3 = data_augmentation(ldct_img, 3)
                    ndct_img_3 = data_augmentation(ndct_img, 3)
                    _, step_loss = self.sess.run(
                        [self.train_op, self.loss],
                        feed_dict={self.Y_ : ndct_img_3, self.X : ldct_img_3,
                                   self.lr: learning_rate, self.is_training : True})
                    print("Epoch: {}/{}\tLoss: {}\n".format(epoch, num_epochs, step_loss / self.batch_size))
                    
                    # # mode 4: rotate 180
                    ldct_img, ndct_img = self.sess.run(next_element)
                    ldct_img_4 = data_augmentation(ldct_img, 4)
                    ndct_img_4 = data_augmentation(ndct_img, 4)
                    _, step_loss = self.sess.run([self.train_op, self.loss],
                                                 feed_dict={self.Y_ : ndct_img_4, self.X : ldct_img_4,
                                                            self.lr: learning_rate, self.is_training : True})
                    print("Epoch: {}/{}\tLoss: {}\n".format(epoch, num_epochs, step_loss / self.batch_size))
                    
                    # # mode 5: rotate 180  and flipud
                    ldct_img, ndct_img = self.sess.run(next_element)
                    ldct_img_5 = data_augmentation(ldct_img, 5)
                    ndct_img_5 = data_augmentation(ndct_img, 5)
                    _, step_loss = self.sess.run([self.train_op, self.loss],
                                                 feed_dict={self.Y_ : ndct_img_5, self.X : ldct_img_5,
                                                            self.lr: learning_rate, self.is_training : True})
                    print("Epoch: {}/{}\tLoss: {}\n".format(epoch, num_epochs, step_loss / self.batch_size))
                    
                    # # mode 6: rotate 270
                    ldct_img, ndct_img = self.sess.run(next_element)
                    ldct_img_6 = data_augmentation(ldct_img, 6)
                    ndct_img_6 = data_augmentation(ndct_img, 6)
                    _, step_loss = self.sess.run([self.train_op, self.loss],
                                                 feed_dict={self.Y_ : ndct_img_6, self.X : ldct_img_6,
                                                            self.lr: learning_rate, self.is_training : True})
                    print("Epoch: {}/{}\tLoss: {}\n".format(epoch, num_epochs, step_loss / self.batch_size))

                    # # mode 7: rotate 270 and flipud
                    ldct_img, ndct_img = self.sess.run(next_element)
                    ldct_img_7 = data_augmentation(ldct_img, 7)
                    ndct_img_7 = data_augmentation(ndct_img, 7)
                    _, step_loss = self.sess.run([self.train_op, self.loss],
                                                 feed_dict={self.Y_ : ndct_img_7, self.X : ldct_img_7,
                                                            self.lr: learning_rate, self.is_training : True})
                    print("Epoch: {}/{}\tLoss: {}\n".format(epoch, num_epochs, step_loss / self.batch_size))
                    #----------- end data augmentation----------------------

                # make summary
                tf.compat.v1.summary.scalar('loss', self.loss)
                tf.compat.v1.summary.scalar('lr', self.lr)
                img =  tf.compat.v1.summary.image('denoised_image', self.Y, max_outputs=1)
                writer =  tf.compat.v1.summary.FileWriter('./logs', self.sess.graph)
                merged =  tf.compat.v1.summary.merge_all()
                summary_psnr =  tf.compat.v1.summary.scalar('eva_psnr', self.eva_psnr)

                self.evaluate(ndct_eval_data, ldct_eval_data, sample_dir, epoch, summary_merged=summary_psnr,
                     summary_writer=writer, summ_img=img)
                self.save(epoch, ckpt_dir)
                epoch = epoch + 1
            # iterator has no more data to iterate over
            except tf.errors.OutOfRangeError:
                print("Training complete")
                break

    # --------------------------evaluate() ---------------------#
    def evaluate(self, valdn_ndct, valdn_ldct, sample_dir, epoch, summary_merged, summary_writer, summ_img):
        print("[*] Evaluating...")
        global count
        global psnr_set
        print("count: ", count)
        print("psnr_set: ", psnr_set)

        if not os.path.exists('ldct_test.tfrecord'):
            generateTFRecords(valdn_ldct, 'ldct_test')
        if not os.path.exists('ndct_test.tfrecord'):
            generateTFRecords(valdn_ndct, 'ndct_test')

        # create dataset
        ldct_valdn_dataset= create_dataset('ldct_test.tfrecord')
        ndct_valdn_dataset= create_dataset('ndct_test.tfrecord')
        valdn_dataset = tf.data.Dataset.zip((ldct_valdn_dataset, ndct_valdn_dataset)).batch(1)

        iterator = valdn_dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        self.sess.run(iterator.initializer)
        psnr_sum = 0

        for i in range(0,len(valdn_ldct)):
            ldct_img, ndct_img = self.sess.run(next_element)

            #-------------------------
            # self.Y:  denoised image
            # self.Y_: clean image
            # self.X:  low dose image
            #-------------------------
            denoised_img,psnr_summary, temp_img  = self.sess.run([self.Y, summary_merged, summ_img],
                            feed_dict = {self.X: ldct_img,
                                         self.Y_:ndct_img,
                                         self.is_training: False})

            summary_writer.add_summary(psnr_summary, epoch)
            summary_writer.add_summary(temp_img, epoch)


            psnr = cal_psnr(ndct_img, denoised_img)
            psnr_sum += psnr
            print("Test img {}/{} PSNR: {}\n".format(i, len(valdn_ldct), psnr))

            #------------------------------------------------------------------
            # we are outputing 3 images in the output_samples folder
            # first: clean image, second: denoised image, third: low dose image
            # only saved the first image
            #------------------------------------------------------------------
            scalef = max(np.amax(denoised_img), np.amax(ndct_img), np.amax(ldct_img))
            denoised_img = np.clip(255 * denoised_img/scalef, 0, 255).astype('uint8')
            ndct_img = np.clip(255 * ndct_img/scalef, 0, 255).astype('uint8')
            ldct_img = np.clip(255*ldct_img/scalef, 0, 255).astype('uint8')
            # change i to any image number that we want to save
            # we can't save all images because we have 3600 images
            if i == 12:
                save_images(os.path.join(sample_dir, 'test%d_%d.png' % (i,epoch)),ldct_img, denoised_img, ndct_img)
        print("Avg PSNR: {}".format(psnr_sum / len(valdn_ndct)))
        current_psnr = psnr_sum / len(valdn_ndct)
        print("current_psnr: ", current_psnr)
        print("count: ", count)
        if(current_psnr <= max(psnr_set)):
          print("count increased by one:...........", count)
          count = count + 1
        else:
          print("new  max............... ", current_psnr)
          count = 0
        if(count == 5):
          print("Done training")
          sys.exit(0)
        psnr_set.add(current_psnr)
        print("psnr_set: ", psnr_set)
        
    # --------------------------test() ---------------------#
    # 1. Calculates average psnr for all test images
    # 2: Saves the denoised image as a float numpy array
    #    print(denoised_img.shape)
    #    (1, 1, 512, 512, 1)
    #-------------------------------------------------------#
    def test(self, test_ldct, test_ndct, ckpt_dir, save_dir):
        assert len(test_ldct) != 0, 'No testing data!'
        assert len(test_ndct) != 0, 'No testing data!'
        # pdb.set_trace()
        print("[*] Evaluating...")

        if not os.path.exists('ldct_test.tfrecord'):
            generateTFRecords(test_ldct, 'ldct_test')
        if not os.path.exists('ndct_test.tfrecord'):
            generateTFRecords(test_ndct, 'ndct_test')

        # create dataset
        ldct_test_dataset= create_dataset('ldct_test.tfrecord')
        ndct_test_dataset= create_dataset('ndct_test.tfrecord')
        test_dataset = tf.data.Dataset.zip((ldct_test_dataset, ndct_test_dataset)).batch(1)

        iterator = test_dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        self.sess.run(iterator.initializer)

        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print(" [*] Load weights SUCCESS...")
        psnr_sum = 0
        print("[*] start testing...")

        rawfiles = [open(os.path.join(save_dir, "test_{num:08d}.flt".format(num=idx)), 'wb') for idx in range (len(test_ndct))]

        for idx in range(0,len(test_ldct)):
            ldct_img, ndct_img = self.sess.run(next_element)
            denoised_img = self.sess.run([self.Y],
                                         feed_dict = {self.X: ldct_img,
                                                    self.Y_: ndct_img,
                                                    self.is_training: False})

            #--------------------------------
            #save image to a test folder
            #--------------------------------
            denoised_img = np.asarray(denoised_img)
            denoised_img.tofile(rawfiles[idx])

            # calculate PSNR
            psnr = cal_psnr(ndct_img, denoised_img)
            print("img%d PSNR: %.2f" % (idx, psnr))
            psnr_sum += psnr
        avg_psnr = psnr_sum / len(test_ndct)
        print("--- Average PSNR %.2f ---" % avg_psnr)

    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        saver = tf.compat.v1.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int( full_path.split('/')[-1].split('-')[-1])
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            return False, 0

    def save(self, iter_num, ckpt_dir, model_name='UNet-tensorflow'):
        saver = tf.compat.v1.train.Saver()
        checkpoint_dir = ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("[*] Saving model...")
        saver.save(self.sess,
                   os.path.join(checkpoint_dir, model_name),
                   global_step=iter_num)
