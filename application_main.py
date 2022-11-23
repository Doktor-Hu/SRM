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
import ARSM
import numpy as np
import shutil

###############################################################################################################################################################
##### ----------- 1. Generating testing data set
###############################################################################################################################################################

#--- Attention, since the .../snowmelt/ is removed this only contains 2011 simulations
py_TF='C:/Users/zh_hu/Documents/Test/TF2/Scripts/python'  # VENV TF2
preprocessing_variables = ['snowmelt']
base_year=2011

### RACMO2 27 km Full
for var in preprocessing_variables:
    code_preprocess='Preprocessing_RACMO2.py "C:/Users/zh_hu/Documents/SR3M/Data" 27000 '+var+'.KNMI-'+str(base_year)+'.ANT27.ERAINx_RACMO2.3p2.DD.nc Y'+str(base_year)+'_ Y Y Y'
    print(py_TF+' '+code_preprocess)
    os.system(py_TF+' '+code_preprocess)

for variable in preprocessing_variables:

    files_27000_ANT = ARSM.Merge_NPY(variable,'C:/Users/zh_hu/Documents/SR3M/Data/Output/',1,27000,'Y'+str(base_year)+'_')
    files_27000_ANT.var_merge_ANT(variable+'_ANT_27000_'+str(base_year)+'.npy','Y')

    for aoi in range(1,14):
        files_27000 = ARSM.Merge_NPY(variable,'C:/Users/zh_hu/Documents/SR3M/Data/Output/',aoi,27000,'Y'+str(base_year)+'_') # This will count all avilable files
        files_27000.var_merge(variable+'_AOI_'+str(aoi)+'_27000_'+str(base_year)+'.npy','Y')


### RACMO2 5.5 km Full
for var in preprocessing_variables:
    code_preprocess='Preprocessing_RACMO2.py "C:/Users/zh_hu/Documents/SR3M/Data" 5500 '+var+'.KNMI-'+str(base_year)+'.XPEN055.ERAINx_RACMO2.4.DD.nc Y'+str(base_year)+'_ Y Y Y'
    print(py_TF+' '+code_preprocess)
    os.system(py_TF+' '+code_preprocess)    


for variable in preprocessing_variables:

    files_5500_ANT = ARSM.Merge_NPY(variable,'C:/Users/zh_hu/Documents/SR3M/Data/Output/',1,5500,'Y'+str(base_year)+'_') # 1 is not taken into process
    files_5500_ANT.var_merge_ANT(variable+'_ANT_5500_'+str(base_year)+'.npy','Y')

    for aoi in range(1,14):
        files_5500 = ARSM.Merge_NPY(variable,'C:/Users/zh_hu/Documents/SR3M/Data/Output/',aoi,5500,'Y'+str(base_year)+'_') # This will count all avilable files
        files_5500.var_merge(variable+'_AOI_'+str(aoi)+'_5500_'+str(base_year)+'.npy','Y')


shutil.rmtree('C:/Users/zh_hu/Documents/SR3M/Data/Output/Cache')
shutil.rmtree('C:/Users/zh_hu/Documents/SR3M/Data/Output/snowmelt')


###############################################################################################################################################################
##### ----------- 1.5. Padding Train
###############################################################################################################################################################
s_var=['snowmelt']*3
stand=False
topo=4
Odir='C:/Users/zh_hu/Documents/SR3M/Data/Output/Variables/'

if not os.path.isdir('C:/Users/zh_hu/Documents/SR3M/Data/apply'):
    os.mkdir('C:/Users/zh_hu/Documents/SR3M/Data/apply')


index=[(0,0),(1,0),(1,1),(2,0),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3),(4,1),(4,2),(4,3)]
AP_X, AP_Z = ARSM.overlap_ALL(s_var,stand,topo,Odir)
for AOI_NR in range(1,14):
    rs=index[AOI_NR-1][0]+1
    cs=index[AOI_NR-1][1]+1
    AOI_X = AP_X[:,(11*rs-2):(11*rs+11+2),(11*cs-2):(11*cs+11+2),:]
    np.save('C:/Users/zh_hu/Documents/SR3M/Data/apply/Input_AOI_'+str(AOI_NR)+'_Daily.npy',AOI_X)


###############################################################################################################################################################
##### ----------- 2. Applying models (AP)
###############################################################################################################################################################
model_name = 'US_Results_Final_HAN_Vanille_LossALL_MAE_Smlt3_Emb0_Bn16_NoStand_Oct-June_Padded2_aoiALL.tf'
model_in = 'solo' 
ARSM.Apply_NPY(model_name, model_in, s_var, stand=False, CSO=None,topo=4)


###############################################################################################################################################################
##### ----------- 3. Plotting results (AP)
###############################################################################################################################################################
model_name = 'US_Results_Final_HAN_Vanille_LossALL_MAE_Smlt3_Emb0_Bn16_NoStand_Oct-June_Padded2_aoiALL' #no .tf
ARSM.plot_month(model_name)
ARSM.plot_year(model_name) 

