#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File        :   SRMwithLOSS.py
@Time        :   2022/07/05 13:19:01
@Author      :   Zhongyang Hu
@Version     :   1.0.0
@Contact     :   z.hu@uu.nl
@Publication :   
@Desc        :   SR Models and Loss functions
'''



import numpy as np
import tensorflow as tf

import pandas as pd
import matplotlib.pyplot as plt

import tensorflow_addons as tfa
from keras import backend as K 

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

from skimage.metrics import structural_similarity

##############################################################################################
#####     LOSS 
##############################################################################################

def ssim(a,b):

  im1 = tf.expand_dims(a, axis=-1)
  im2 = tf.expand_dims(b, axis=-1)

  return tf.image.ssim(im1,im2, 5, filter_size=3)


def SSIM_Loss(y_true, y_pred):

    return 1 - tf.reduce_mean(ssim(y_true, y_pred))
    #return 1 - tf.reduce_mean(structural_similarity(y_true, y_pred, full=True))


def CMAE_loss(y_true, y_pred):

    return tf.reduce_mean(tf.keras.metrics.mean_absolute_error(y_true, y_pred))


#################
def VGG_Loss(y_true, y_pred):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape = [74,74,3])
    vgg.trainable = False
    content_layers = 'block3_conv4'

    lossModel = tf.keras.models.Model([vgg.input],vgg.get_layer(content_layers).output,name = 'vgg_Layer')

    #Xt = preprocess_input(y_pred)
    #Yt = preprocess_input(y_true)
    y_pred=tf.expand_dims(y_pred, axis=-1)
    y_pred = tf.keras.layers.concatenate([y_pred,y_pred,y_pred],axis=-1)

    y_true=tf.expand_dims(y_true, axis=-1)
    y_true = tf.keras.layers.concatenate([y_true,y_true,y_true],axis=-1)
    
    vggX = lossModel.predict(y_true)
    vggY = lossModel.predict(y_pred)
    
    #return tf.reduce_mean(tf.square(vggY-vggX))
    return np.mean(np.square(np.array(vggY).flatten()-np.array(vggX).flatten()))

def Y_pool(x):
    p=tf.keras.layers.AveragePooling2D(5,padding='same')(x)
    p.trainable = False
    lr_pred = tf.keras.layers.concatenate([p,p,p],axis=-1)
    return lr_pred

def SMLT_Loss(input_lr):
    def calc_loss(y_true, y_pred):
        lr_es =  Y_pool(y_pred)
        return tf.reduce_mean(tf.square(input_lr-lr_es))
    return calc_loss

##############################################################################################
#####     SRResNet
##############################################################################################

def down_sampling_conv(input_img,Fact):
    Conv1 = tf.keras.layers.Conv2D(64, 3, padding = 'same')(input_img)
    Conv1 = tf.keras.layers.BatchNormalization()(Conv1)    
    Conv1 =tf.keras.layers.Activation(Fact)(Conv1)
    Conv1 = tf.keras.layers.Conv2D(64, 3, padding = 'same')(Conv1)
    Conv1 = tf.keras.layers.BatchNormalization()(Conv1) 
    output_img =  tf.math.add(input_img, Conv1)
    return output_img


def SRResNet(inputs_climate):
    n_d = 16 
    Fact = 'PReLU'

    #inputs_climate = tf.keras.layers.Input(input_size, name='climate')  

    Conv0 = tf.keras.layers.Conv2D(64, 3, padding = 'same')(inputs_climate)
    Conv_R =tf.keras.layers.Activation(Fact)(Conv0)
    Conv_GSC = Conv_R

    for i in range(n_d):
      Conv_R = down_sampling_conv(Conv_R,Fact)

    Conv_R = tf.math.add(Conv_GSC, Conv_R)
    
    U_Conv1 = tf.keras.layers.Conv2D(256, 3, padding = 'same')(Conv_R)  
    U_Conv1 = tf.nn.depth_to_space(U_Conv1, 2)
    U_Conv1 = tf.keras.layers.Conv2D(64, 3, padding = 'valid')(U_Conv1) # 28   
    U_Conv1 =tf.keras.layers.Activation(Fact)(U_Conv1)  
    U_Conv1 = tf.keras.layers.Conv2D(64, 3, padding = 'valid')(U_Conv1) # 26   
    U_Conv1 =tf.keras.layers.Activation(Fact)(U_Conv1)     

    U_Conv2 = tf.keras.layers.Conv2D(64*3*3, 3, padding = 'same')(U_Conv1)  
    U_Conv2 = tf.nn.depth_to_space(U_Conv2, 3)
    U_Conv2 = tf.keras.layers.Conv2D(64, 3, padding = 'valid')(U_Conv2) # 76   
    U_Conv2 =tf.keras.layers.Activation(Fact)(U_Conv2)  
    U_Conv2 = tf.keras.layers.Conv2D(64, 3, padding = 'valid')(U_Conv2) # 74   
    U_Conv2 =tf.keras.layers.Activation(Fact)(U_Conv2)    

    O_Conv = tf.keras.layers.Conv2D(1, 1,  padding = 'same')(U_Conv2)                                       
 
    model = tf.keras.models.Model(inputs = inputs_climate, outputs=O_Conv)
    #model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate = 1e-3), loss = 'mse', metrics = ['mse'])

    return model

##############################################################################################
#####     HAN 
##############################################################################################


def CSAM(x):
  res = x
  CM_Conv3D = tf.keras.layers.Conv3D(1, (3,3,3), activation='sigmoid', padding='same', input_shape=x.shape[2:])(x)
  out =  tf.math.multiply(CM_Conv3D, x)
  out = tf.math.add(out,res)
  out = tf.reshape(out, (-1, res.shape[-4],res.shape[-3],res.shape[-2]))
  return out

def LAM(x):

  res0 = x

  B, N, H, W, C = x.shape[-5], x.shape[-4], x.shape[-3], x.shape[-2], x.shape[-1]
  
  x_reshaped= tf.reshape(x, (-1, N, H*W*C))

  res1 = x_reshaped
  
  x_transpose = tf.transpose(x_reshaped)
  x_transpose = K.permute_dimensions(x_transpose, (2,0,1))

  cm = tf.matmul(x_reshaped,x_transpose)

  cm_softmax = tf.keras.activations.softmax(cm)

  out = tf.matmul(cm_softmax, res1)

  out_reshape = tf.reshape(out, (-1, N, H, W, C))

  LSK = tf.math.add(res0,out_reshape)

  LSK_reshape = tf.reshape(LSK, (-1, H, W, N*C))

  return LSK_reshape

def RCAB(x):

  H,W,C = x.shape[-3], x.shape[-2], x.shape[-1]
  res0 = x
  Conv1= tf.keras.layers.Conv2D(filters= C,kernel_size=1, strides=(1, 1), padding='valid', use_bias=True,activation='PReLU')(x)
  res1 = Conv1
  
  Conv2= tf.keras.layers.Conv2D(filters= C,kernel_size=1, strides=(1, 1), padding='valid', use_bias=True,activation='PReLU')(x)

  Pool1= tfa.layers.AdaptiveAveragePooling2D(1)(Conv2)
  sig= tf.keras.activations.sigmoid(Pool1)

  att=tf.math.multiply(sig,res1)
  add_res=tf.math.add( att, res0)

  return add_res


def HAN(input_layer):

    filter_n_head = 64
    kernel_siz_head=3
    padding_head='same'

    filter_n_RG = 128
    kernel_siz_RG=3
    padding_RG='same'

    Fact = 'PReLU'

    # -----------------------

    head_out = tf.keras.layers.Conv2D(filters= filter_n_head,kernel_size=kernel_siz_head, strides=(1, 1), padding=padding_head, use_bias=True)(input_layer)
    res_global = head_out

    # RG-l

    RG_l_res = head_out
    
    LRCAB_l = RCAB(head_out)
    LRCAB_i = RCAB(LRCAB_l)
    LRCAB_m = RCAB(LRCAB_i)

    LRCAB_Conv = tf.keras.layers.Conv2D(filters= 64, kernel_size=kernel_siz_RG, strides=(1, 1), padding=padding_RG, use_bias=True)(LRCAB_m)
    RG_l_out =  tf.math.add(LRCAB_Conv, RG_l_res)

    # RG-n

    RG_n_res = RG_l_out
    
    IRCAB_l = RCAB(RG_l_out)
    IRCAB_i = RCAB(IRCAB_l)
    IRCAB_m = RCAB(IRCAB_i)

    IRCAB_Conv = tf.keras.layers.Conv2D(filters= 64, kernel_size=kernel_siz_RG, strides=(1, 1), padding=padding_RG, use_bias=True)(IRCAB_m)
    RG_n_out =  tf.math.add(IRCAB_Conv, RG_n_res)


    # RG-N

    RG_n_res = RG_l_out
    
    IRCAB_l = RCAB(RG_l_out)
    IRCAB_i = RCAB(IRCAB_l)
    IRCAB_m = RCAB(IRCAB_i)

    IRCAB_Conv = tf.keras.layers.Conv2D(filters= 64, kernel_size=kernel_siz_RG, strides=(1, 1), padding=padding_RG, use_bias=True)(IRCAB_m)
    RG_N_out =  tf.math.add(IRCAB_Conv, RG_n_res)


    # CSAM

    CSAM_in = tf.keras.layers.Conv2D(filters= filter_n_RG, kernel_size=kernel_siz_RG, strides=(1, 1), padding=padding_RG, use_bias=True)(RG_N_out)
    CSAM_in = tf.expand_dims(CSAM_in,-1)
    CSAM_out = CSAM(CSAM_in) 

    # LAM
    LAM_in = tf.keras.layers.concatenate([tf.expand_dims(RG_l_out,1), tf.expand_dims(RG_n_out,1), tf.expand_dims(RG_N_out,1)],axis=1) 
    LAM_out = LAM(LAM_in)

    # Sum-up
    Combined = tf.keras.layers.concatenate([LAM_out, CSAM_out, res_global]) 

    # Upsampleing
    U_Conv1 = tf.keras.layers.Conv2D(256, 3, padding = 'same')(Combined)  
    U_Conv1 = tf.nn.depth_to_space(U_Conv1, 2)
    U_Conv1 = tf.keras.layers.Conv2D(64, 3, padding = 'valid')(U_Conv1) # 28   
    U_Conv1 =tf.keras.layers.Activation(Fact)(U_Conv1)  
    U_Conv1 = tf.keras.layers.Conv2D(64, 3, padding = 'valid')(U_Conv1) # 26   
    U_Conv1 =tf.keras.layers.Activation(Fact)(U_Conv1)     

    U_Conv2 = tf.keras.layers.Conv2D(64*3*3, 3, padding = 'same')(U_Conv1)  
    U_Conv2 = tf.nn.depth_to_space(U_Conv2, 3)
    U_Conv2 = tf.keras.layers.Conv2D(64, 3, padding = 'valid')(U_Conv2) # 76   
    U_Conv2 =tf.keras.layers.Activation(Fact)(U_Conv2)  
    U_Conv2 = tf.keras.layers.Conv2D(64, 3, padding = 'valid')(U_Conv2) # 74   
    U_Conv2 =tf.keras.layers.Activation(Fact)(U_Conv2)    

    O_Conv = tf.keras.layers.Conv2D(1, 1,  padding = 'same')(U_Conv2) 

    model = tf.keras.models.Model(inputs = input_layer, outputs=O_Conv)
    #model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate = 1e-3), loss=[CMAE_loss,SSIM_Loss,SMLT_Loss(input_layer)], loss_weights=[1e-2, 5.25e-0,2e-3], metrics = ['mse'])

    return model
