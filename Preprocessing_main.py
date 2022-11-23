#C:/Users/zh_hu/Documents/Test/TF2/Scripts/python
# -*-coding:utf-8 -*-
'''
@File        :   Data_Preprocessing.py
@Time        :   2021/12/14 13:12:01
@Author      :   Zhongyang Hu
@Version     :   1.0.0
@Contact     :   z.hu@uu.nl
@Publication :   
@Desc        :   ARSM Main Programme
'''
### ------ Python config and packages
import os
import cv2
import ARSM
import numpy as np
import shutil
import SRconfig

###############################################################################################################################################################
##### ----------- 1. Cropping and coregistrating RACMO2 27 and 5.5 km resolution data
###############################################################################################################################################################
preprocessing_variables = ['snowmelt']
base_year=2001

##-------------------------------------------------------------------------------------------------------------------------------------------------------------
py_TF=SRconfig.py_TF  # VENV TF2
'''
Converting NetCDF to GeoTiFF
Cropping into AOIs

'''

## -- 27 km RACMO2

for var in preprocessing_variables:
    code_preprocess='Preprocessing_RACMO2.py "C:/Users/zh_hu/Documents/SR3M/Data" 27000 '+var+'.KNMI-'+str(base_year)+'.ANT27.ERAINx_RACMO2.3p2.DD.nc Y'+str(base_year)+'_ Y Y Y'
    print(py_TF+' '+code_preprocess)
    os.system(py_TF+' '+code_preprocess)

for variable in preprocessing_variables:

    files_27000_ANT = ARSM.Merge_NPY(variable,'C:/Users/zh_hu/Documents/SR3M/Data/Output/',1,27000,'Y'+str(base_year)+'_')
    files_27000_ANT.var_merge_ANT(variable+'_ANT_27000_'+str(base_year)+'.npy','Y')

    for aoi in range(1,14):
        files_27000 = ARSM.Merge_NPY(variable,'C:/Users/zh_hu/Documents/SR3M/Data/Output/',aoi,27000,'Y'+str(base_year)+'_')
        files_27000.var_merge(variable+'_AOI_'+str(aoi)+'_27000_'+str(base_year)+'.npy','Y')



## -- 5.5 km RACMO2
for var in preprocessing_variables:
    code_preprocess='Preprocessing_RACMO2.py "C:/Users/zh_hu/Documents/SR3M/Data" 5500 '+var+'.KNMI-'+str(base_year)+'.XPEN055.ERAINx_RACMO2.4.DD.nc Y'+str(base_year)+'_ Y Y Y'
    print(py_TF+' '+code_preprocess)
    os.system(py_TF+' '+code_preprocess)    


for variable in preprocessing_variables:

    files_5500_ANT = ARSM.Merge_NPY(variable,'C:/Users/zh_hu/Documents/SR3M/Data/Output/',1,5500,'Y'+str(base_year)+'_') # 1 is not taken into process
    files_5500_ANT.var_merge_ANT(variable+'_ANT_5500_'+str(base_year)+'.npy','Y')

    for aoi in range(1,14):
        files_5500 = ARSM.Merge_NPY(variable,'C:/Users/zh_hu/Documents/SR3M/Data/Output/',aoi,5500,'Y'+str(base_year)+'_')
        files_5500.var_merge(variable+'_AOI_'+str(aoi)+'_5500_'+str(base_year)+'.npy','Y')


shutil.rmtree('C:/Users/zh_hu/Documents/SR3M/Data/Output/Cache')
shutil.rmtree('C:/Users/zh_hu/Documents/SR3M/Data/Output/snowmelt')


###############################################################################################################################################################
##### ----------- 2. Training, development, testing data sets
###############################################################################################################################################################

aoi_nr=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
s_var=['snowmelt','snowmelt','snowmelt']
stand_mode=False

##-------------------------------------------------------------------------------------------------------------------------------------------------------------
file_dir = SRconfig.file_dir
train_dir = SRconfig.train_dir
dev_dir = SRconfig.dev_dir

train_Xa,  train_Ya,  train_Za,  dev_Xa, dev_Ya, dev_Za = ARSM.overlap_train(file_dir, s_var, stand=stand_mode, step=2, topo=4, aoi_nr=aoi_nr, \
      MO=['10','11','12','01','02','03','04','05','06'])


if not os.path.isdir(train_dir):
    os.mkdir(train_dir)

if not os.path.isdir(dev_dir):
    os.mkdir(dev_dir)


np.save(os.path.join(train_dir,'Train_LR.npy'),train_Xa)
np.save(os.path.join(train_dir,'Train_HR.npy'),train_Ya)
np.save(os.path.join(train_dir,'Train_GEO.npy'),train_Za)

np.save(os.path.join(dev_dir,'Dev_LR.npy'),dev_Xa)
np.save(os.path.join(dev_dir,'Dev_HR.npy'),dev_Ya)
np.save(os.path.join(dev_dir,'Dev_GEO.npy'),dev_Za)


train_Xa_U = np.zeros(train_Xa.shape)
for i in range(train_Xa.shape[0]):
    upscaled_train = cv2.resize(train_Ya[i,:,:], (15,15), interpolation = cv2.INTER_AREA)
    train_Xa_U[i,:,:,0]= upscaled_train
    train_Xa_U[i,:,:,1]= upscaled_train
    train_Xa_U[i,:,:,2]= upscaled_train

np.save(os.path.join(train_dir,'Train_ULR.npy'),train_Xa_U)
