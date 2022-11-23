import SRM_architecture as SL
import math
import tensorflow as tf
import os
import random as python_random
import numpy as np
import Swin
import argparse

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(123)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
python_random.seed(123)

# The below set_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed
tf.random.set_seed(1234)

# setup the parser
parser = argparse.ArgumentParser(description='Training a super resolution model')
parser.add_argument('SR_Model', type=str, help='SRResNet, HAN, Swin')
parser.add_argument('LOSS_Mode', type=str, help='lOSS mode: MSE, VGG, LR, ALL')
parser.add_argument('AOIs', type=str, help='Area of Intersts')
parser.add_argument('Numver_of_Input_Variables', type=int, help='Input Variables')
parser.add_argument('Write_NPY', type=str, help='Write Numpy Array [Y/N]')
parser.add_argument('Plot_NPY', type=str, help='Plot the Written Numpy Array [Y/N]')
parser.add_argument('Batch_Size', type=int, help='Batch Size')
parser.add_argument('Training_Padding', type=int, help='Training Padding')
parser.add_argument('Standization', type=str, help='Standization [Y/N]')
parser.add_argument('Save_Format', type=str, help='Save mMdel as TF [Y/N]')
parser.add_argument('MD', type=float, help='Early Stopping Minimal Delta')
parser.add_argument('Embedding_Dim', type=int, help='Embedding Dimension for Swin')
parser.add_argument('model_version', type=int, help='Model Version')

# parse the args
args = parser.parse_args()
#----------------------------------------------------------------------------------------------------------------------------------------------------------



SRModel = args.SR_Model
loss_mode = args.LOSS_Mode
sAOI = args.AOIs
model_version = args.model_version

if args.AOIs == 'ALL':
    aoi_nr=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
if args.AOIs == 'WIS':
    aoi_nr=[1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]    
n_var=args.Numver_of_Input_Variables

if args.Write_NPY == 'T':
    np_out=True
elif args.Write_NPY == 'F':
    np_out=False
else:
    print('Error in write npy')

if args.Plot_NPY == 'T':
    plot_out=True
elif args.Plot_NPY == 'F':
    plot_out=False
else:
    print('Error in plot npy')

bs=args.Batch_Size
training_padding=args.Training_Padding

if args.Standization == 'T':
    stand_mode=True
elif args.Standization == 'F':
    stand_mode=False
else:
    print('Error in standarization')


if args.Save_Format == 'T':
    saveh5=True
elif args.Save_Format == 'F':
    saveh5=False
else:
    print('Error in save format')

saveh5=True

md =  args.MD 
emb_dim= args.Embedding_Dim

s_var=['snowmelt']*n_var


model_mode = '{}_Results_Final_{}_Vanille_Loss{}_Smlt{}_Emb{}_Bn{}_NoStand_Oct-June_Padded{}_aoi{}'.format(SRModel,loss_mode,n_var,emb_dim,bs,training_padding,sAOI,model_version)

input_layer = tf.keras.layers.Input((15,15,n_var), name='climate')  

if SRModel == 'HAN':
    SR_model=SL.HAN(input_layer)
elif SRModel == 'SRResNet':
    SR_model=SL.SRResNet(input_layer)
elif SRModel == 'Swin':
    SR_model=Swin.SwinSR((15,15,n_var),[2,2,6,2],emb_dim)
else:
    print('Unknow SR Model!')

if loss_mode == 'MSE':
    SR_model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate = 1e-3), loss='mse', metrics = ['mse'])
elif loss_mode == 'MAE':
    SR_model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate = 1e-3), loss=tf.keras.losses.MeanAbsoluteError(), metrics = ['mse'])    
elif loss_mode == 'LR_MAE':
    SR_model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate = 1e-3), loss=[tf.keras.losses.MeanAbsoluteError(), SL.SSIM_Loss,SL.SMLT_Loss(input_layer)], loss_weights=[1e-2, 5.25e-0,2e-3], metrics = ['mse'])
elif loss_mode == 'VGG_MAE':
    SR_model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate = 1e-3), loss=[tf.keras.losses.MeanAbsoluteError(), SL.SSIM_Loss,SL.VGG_Loss], loss_weights=[1e-2, 5.25e-0,2e-3], metrics = ['mse'])
elif loss_mode == 'ALL_MAE':
    SR_model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate = 1e-3), loss=[tf.keras.losses.MeanAbsoluteError(), SL.SSIM_Loss,SL.SMLT_Loss(input_layer),SL.VGG_Loss], loss_weights=[1e-2, 5.25e-0,1e-3,1e-3], metrics = ['mse'])
elif loss_mode == 'LR_MSE':
    SR_model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate = 1e-3), loss=['mse', SL.SSIM_Loss,SL.SMLT_Loss(input_layer)], loss_weights=[1e-2, 5.25e-0,2e-3], metrics = ['mse'])
elif loss_mode == 'VGG_MSE':
    SR_model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate = 1e-3), loss=['mse', SL.SSIM_Loss,SL.VGG_Loss], loss_weights=[1e-2, 5.25e-0,2e-3], metrics = ['mse'])
elif loss_mode == 'ALL_MSE':
    SR_model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate = 1e-3), loss=['mse', SL.SSIM_Loss,SL.SMLT_Loss(input_layer),SL.VGG_Loss], loss_weights=[1e-2, 5.25e-0,1e-3,1e-3], metrics = ['mse'])
elif loss_mode == 'ALL_MASE':
    SR_model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate = 1e-3), loss=['mse',tf.keras.losses.MeanAbsoluteError(), SL.SSIM_Loss,SL.SMLT_Loss(input_layer),SL.VGG_Loss], loss_weights=[5e-3,5e-3, 5.25e-0,1e-3,1e-3], metrics = ['mse'])
elif loss_mode == 'Mode1_ALL':
    SR_model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate = 2e-3), loss=[SL.CMAE_loss, SL.SSIM_Loss,SL.SMLT_Loss(input_layer),SL.VGG_Loss], loss_weights=[1e-2, 5.25e-0,1e-3,1e-3], metrics = ['mse'])
elif loss_mode == 'Mode2_ALL':
    SR_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3), loss=[SL.CMAE_loss, SL.SSIM_Loss,SL.SMLT_Loss(input_layer),SL.VGG_Loss], loss_weights=[1e-2, 5.25e-0,1e-3,1e-3], metrics = ['mse'])
elif loss_mode == 'Mode4_ALL':
    SR_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3), loss=['mse', SL.SSIM_Loss,SL.SMLT_Loss(input_layer),SL.VGG_Loss], loss_weights=[0,0,0,0,1000000], metrics = ['mse'])    
else:
    print('Unknow Loss Function!')
SR_model.summary()

# ---------------------------------- main ------------------------------------------------------------
logger_name = 'C:/Users/zh_hu/Documents/SR3M/Model/'+model_mode+'_log.csv'
if saveh5:
      model_outname='C:/Users/zh_hu/Documents/SR3M/Model/'+model_mode+'.tf'
else:
      model_outname='C:/Users/zh_hu/Documents/SR3M/Model/'+model_mode

train_Xa    = np.load('C:/Users/zh_hu/Documents/SR3M/Data/train/Train_LR.npy')
train_Ya    = np.load('C:/Users/zh_hu/Documents/SR3M/Data/train/Train_HR.npy')
train_Za    = np.load('C:/Users/zh_hu/Documents/SR3M/Data/train/Train_GEO.npy')
dev_Xa      = np.load('C:/Users/zh_hu/Documents/SR3M/Data/dev/Dev_LR.npy')
dev_Ya      = np.load('C:/Users/zh_hu/Documents/SR3M/Data/dev/Dev_HR.npy')
dev_Za      = np.load('C:/Users/zh_hu/Documents/SR3M/Data/dev/Dev_GEO.npy')


def step_decay(epoch, lr):
      drop = 0.8
      epochs_drop = 5.0
      lrate = lr * math.pow(drop,math.floor((1+epoch)/epochs_drop))
      if lrate<5e-9:
        lrate=5e-9
      return lrate

my_lr_scheduler = tf.keras.callbacks.LearningRateScheduler(step_decay)

early_stopping = tf.keras.callbacks.EarlyStopping(
          monitor = 'loss',
          min_delta=md,
          patience = 5)

history_logger=tf.keras.callbacks.CSVLogger(logger_name, separator=",", append=True)
history = SR_model.fit(train_Xa, train_Ya, epochs=30, batch_size=bs, validation_data=(dev_Xa, dev_Ya), callbacks=[history_logger,my_lr_scheduler,early_stopping]) #30/40
SR_model.save(model_outname)