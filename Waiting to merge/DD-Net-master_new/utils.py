import tensorflow as tf
import numpy as np
from PIL import Image

def tf_psnr(im1, im2):
    # assert pixel value range is 0-1                                                                                                                                  
    #mse = tf.losses.mean_squared_error(labels=im2 * 255.0, predictions=im1 * 255.0)                                                                                   
    mse = tf.compat.v1.losses.mean_squared_error(labels=im2 * 255.0, predictions=im1 * 255.0)
    return 10.0 * (tf.math.log(255.0 ** 2 / mse) / tf.math.log(10.0))

def psnr_op(pred, y):
        """                                                                                                                                                            
        computes psnr between cleaned image and ground truth image                                                                                                     
        """
        # mse = tf.losses.mean_squared_error(labels=y * 255.0, predictions=pred * 255.0)                                                                               
        mse = tf.compat.v1.losses.mean_squared_error(labels=y * 255.0, predictions=pred * 255.0)
        return 10.0 * (tf.math.log(255.0 ** 2 / mse) / tf.math.log(10.0))

def save_images(path, ndct_img, denoised_img, ldct_img):
        """                                                                                                                                                            
        saves the ldct, ndct, and denoised images to PNG format in the output_samples directory                                                                        
        """
        ndct_img= np.squeeze(ndct_img)
        denoised_img= np.squeeze(denoised_img)
        ldct_img= np.squeeze(ldct_img)
        cat_img= np.concatenate([ldct_img, denoised_img, ndct_img], axis=1)
        im = Image.fromarray(cat_img.astype('uint8')).convert('L')
        im.save(path, 'png')

def cal_psnr(im1, im2):
    # assert pixel value range is 0-255 and type is uint8
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    maxval = np.amax(im1)
    psnr = 10 * np.log10(maxval ** 2 / mse)
    return psnr

def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image, axes=(-3,-2))
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image, k=1, axes=(-3,-2))
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2, axes=(-3,-2))
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2, axes=(-3,-2))
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3, axes=(-3,-2))
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3, axes=(-3,-2))
        return np.flipud(image)

def random_crop(imgs1,imgs2):
    imgs1_new = []
    imgs2_new = []
    x, y = np.random.randint(256), np.random.randint(256)
    for img1, img2 in zip(imgs1,imgs2):
        img1 = img1.copy()
        img1 = img1[y:y+256, x:x+256]
        imgs1_new.append(img1)
        img2 = img2.copy()
        img2 = img2[y:y+256, x:x+256]
        imgs2_new.append(img2)
    return imgs1_new, imgs2_new
