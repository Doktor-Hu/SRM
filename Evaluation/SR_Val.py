from tabnanny import verbose
import tensorflow as tf
import ultility
import SRMwithLOSS as SL
import glob
import os
import numpy as np
import scipy
from scipy import stats
from skimage.metrics import structural_similarity
import pandas as pd

## ----
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

def batch_input(x, batch_size = 1): 
    train_hr_batches = []
    for it in range(int(x.shape[0] / batch_size)):
        start_idx = it * batch_size
        end_idx = start_idx + batch_size
        train_hr_batches.append(x[start_idx:end_idx])
    return train_hr_batches

def G_to_RGB(X,dim_exp=True):
    X = X.astype('float')
    if dim_exp:
        X = np.expand_dims(X, axis=-1)
    exp_out = np.concatenate([X,X,X],axis=-1) # tf.image.grayscale_to_rgb(X)
    return exp_out

def VGG_trans(X,dim_exp=True):

    out=[]

    X_RGB = G_to_RGB(X,dim_exp)
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape = [54,54,3])
    vgg.trainable = False
    content_layers = 'block3_conv4'#'block5_conv4'

    lossModel = tf.keras.models.Model([vgg.input],vgg.get_layer(content_layers).output,name = 'vgg_Layer')
    
    Batch_X=batch_input(X_RGB, batch_size = 16)  
    for batch_idx in range(len(Batch_X)):
        xt = Batch_X[batch_idx]
        vggX = lossModel.predict(xt, batch_size = 16,verbose=0) 
        out.append(vggX)
    
    return np.array(out)

def VGG_Loss(y_true, y_pred):
    return float(np.mean(np.square(np.array(VGG_trans(y_true, True)).flatten()-np.array(VGG_trans(y_pred, True)).flatten())))
## ---

#structural_similarity(img1,img2,data_range=110,filter=3)

def eval_acc(y_pred,y_true):
    RMSE_i = np.sqrt(np.mean((y_pred.flatten()-y_true.flatten())**2)) 
    #MSE_i = np.mean(np.sqrt((y_pred.flatten()-y_true.flatten())**2)) 
    R2 = scipy.stats.pearsonr(y_pred.flatten(),y_true.flatten())[0]
    MAE_i = np.mean(np.absolute(y_pred.flatten()-y_true.flatten())) 
    return RMSE_i, R2, MAE_i

def Y_pool(x):
    x=np.expand_dims(x, axis=-1)
    p=tf.keras.layers.AveragePooling2D(5,padding='same')(x)
    p.trainable = False
    lr_pred = tf.keras.layers.concatenate([p,p,p],axis=-1)
    return lr_pred

def eval_lr(input_lr, y_pred):
      lr_es =  Y_pool(y_pred)
      return float(tf.reduce_mean(tf.square(input_lr-lr_es)))

def eval_str(y_pred,y_true):
    ssim_i=[]

    for b in range(y_pred.shape[0]):
      ssim_i.append(structural_similarity(y_pred[b,:,:],y_true[b,:,:],data_range=5,filter=3)) #, full=True 110--->50 ---> 10 --->5

    return np.array(ssim_i).mean()


aoi_nr=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

s_var=['snowmelt','snowmelt','snowmelt']
n_var=len(s_var)
stand_mode=False
models = glob.glob('G:/My Drive/SRM/Experiment/Models/V6_*')

train_Xa,  train_Ya,  train_Za, val_Xa,  val_Ya,  val_Za, test_Xa, test_Ya, test_Za = ultility.overlap_evaluation(s_var, stand=stand_mode, step=2, topo=4, aoi_nr=aoi_nr, \
      SI=False, MO=['10','11','12','01','02','03','04','05','06'])

del train_Za,val_Za, test_Za
#del test_Xa, test_Ya, val_Xa, val_Ya


ACC_Summary_train = pd.DataFrame(-999, columns=['Model_Mode','RMSE','R2','MAE','SSIM','VGG','SMLT'], index = ['Bicubic'] + [os.path.basename(i)[17:-3] for i in models])
ACC_Summary_val = pd.DataFrame(-999, columns=['Model_Mode','RMSE','R2','MAE','SSIM','VGG','SMLT'], index = ['Bicubic'] + [os.path.basename(i)[17:-3] for i in models])
ACC_Summary_test = pd.DataFrame(-999, columns=['Model_Mode','RMSE','R2','MAE','SSIM','VGG','SMLT'], index = ['Bicubic'] + [os.path.basename(i)[17:-3] for i in models])

k=0

Train_EST = np.zeros((train_Xa.shape[0],54,54))
Val_EST = np.zeros((val_Xa.shape[0],54,54))
Test_EST = np.zeros((test_Xa.shape[0],54,54))

import cv2

for t in range(train_Xa.shape[0]):
  Train_EST[t,:,:]= cv2.resize(train_Xa[t,2:13,2:13,0], (54,54), interpolation = cv2.INTER_CUBIC)
for t in range(val_Xa.shape[0]):
  Val_EST[t,:,:]= cv2.resize(val_Xa[t,2:13,2:13,0], (54,54), interpolation = cv2.INTER_CUBIC)
for t in range(test_Xa.shape[0]):
  Test_EST[t,:,:]= cv2.resize(test_Xa[t,2:13,2:13,0], (54,54), interpolation = cv2.INTER_CUBIC)

ACC_Summary_train.iloc[k,0:3]=eval_acc(Train_EST, train_Ya[:,10:64,10:64])
ACC_Summary_train.iloc[k,3]=eval_str(Train_EST, train_Ya[:,10:64,10:64])
ACC_Summary_train.iloc[k,4]=VGG_Loss(train_Ya[:,10:64,10:64],Train_EST)
ACC_Summary_train.iloc[k,5]=eval_lr(train_Xa[:,2:13,2:13,:], Train_EST)

ACC_Summary_val.iloc[k,0:3]=eval_acc(Val_EST, val_Ya[:,10:64,10:64])
ACC_Summary_val.iloc[k,3]=eval_str(Val_EST, val_Ya[:,10:64,10:64])
ACC_Summary_val.iloc[k,4]=VGG_Loss(val_Ya[:,10:64,10:64],Val_EST)
ACC_Summary_val.iloc[k,5]=eval_lr(val_Xa[:,2:13,2:13,:], Val_EST)

ACC_Summary_test.iloc[k,0:3]=eval_acc(Test_EST, test_Ya[:,10:64,10:64])
ACC_Summary_test.iloc[k,3]=eval_str(Test_EST, test_Ya[:,10:64,10:64])
ACC_Summary_test.iloc[k,4]=VGG_Loss(test_Ya[:,10:64,10:64], Test_EST)
ACC_Summary_test.iloc[k,5]=eval_lr(test_Xa[:,2:13,2:13,:], Test_EST)

k+=1

for model_i in models:
    SR_model = tf.keras.models.load_model(model_i,custom_objects=None, compile=False)

    Train_EST = SR_model.predict(train_Xa)
    Val_EST = SR_model.predict(val_Xa)
    Test_EST = SR_model.predict(test_Xa)

    ACC_Summary_train.iloc[k,0:3]=eval_acc(Train_EST[:,10:64,10:64,0], train_Ya[:,10:64,10:64])
    ACC_Summary_train.iloc[k,3]=eval_str(Train_EST[:,10:64,10:64,0], train_Ya[:,10:64,10:64])
    ACC_Summary_train.iloc[k,4]=VGG_Loss(train_Ya[:,10:64,10:64],Train_EST[:,10:64,10:64,0])
    ACC_Summary_train.iloc[k,5]=eval_lr(train_Xa[:,2:13,2:13,:], Train_EST[:,10:64,10:64,0])

    ACC_Summary_val.iloc[k,0:3]=eval_acc(Val_EST[:,10:64,10:64,0], val_Ya[:,10:64,10:64])
    ACC_Summary_val.iloc[k,3]=eval_str(Val_EST[:,10:64,10:64,0], val_Ya[:,10:64,10:64])
    ACC_Summary_val.iloc[k,4]=VGG_Loss(val_Ya[:,10:64,10:64],Val_EST[:,10:64,10:64,0])
    ACC_Summary_val.iloc[k,5]=eval_lr(val_Xa[:,2:13,2:13,:], Val_EST[:,10:64,10:64,0])

    ACC_Summary_test.iloc[k,0:3]=eval_acc(Test_EST[:,10:64,10:64,0], test_Ya[:,10:64,10:64])
    ACC_Summary_test.iloc[k,3]=eval_str(Test_EST[:,10:64,10:64,0], test_Ya[:,10:64,10:64])
    ACC_Summary_test.iloc[k,4]=VGG_Loss(test_Ya[:,10:64,10:64], Test_EST[:,10:64,10:64,0])
    ACC_Summary_test.iloc[k,5]=eval_lr(test_Xa[:,2:13,2:13,:], Test_EST[:,10:64,10:64,0])

    del Train_EST, Val_EST, Test_EST

    k+=1



ACC_Summary_train.to_csv('G:/My Drive/SRM/Experiment/Acc/SRV6_ACC_Train_V5_SSIM5.csv',index_label=False)
ACC_Summary_val.to_csv('G:/My Drive/SRM/Experiment/Acc/SRV6_ACC_Val_V5_SSIM5.csv',index_label=False)
ACC_Summary_test.to_csv('G:/My Drive/SRM/Experiment/Acc/SRV6_ACC_Test_V5_SSIM5.csv',index_label=False)


#os.system("shutdown /s /t 1")

