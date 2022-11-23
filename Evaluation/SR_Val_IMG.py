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
import mapply
import datetime
import cv2


import matplotlib.pyplot as plt


aoi_nr=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

s_var=['snowmelt','snowmelt','snowmelt']
n_var=len(s_var)
stand_mode=False
models = glob.glob('G:/My Drive/SRM/Experiment/Models/*Loss*')



racmo_sim_5d5 = np.load('G:/My Drive/SRM/Experiment/Results_NPY/RACMO_5d5_Daily.npy')
racmo_sim_27 = np.load('G:/My Drive/SRM/Experiment/Results_NPY/RACMO_27_Daily.npy')
racmo_sim_5d5_bicubic = np.load('G:/My Drive/SRM/Experiment/Results_NPY/RACMO_5d5r_Daily_corr.npy')
racmo_sim_27_area = np.load('G:/My Drive/SRM/Experiment/Results_NPY/RACMO_27r_Daily_corr.npy')

t=1
fig=plt.subplot(1,4,1)
plt.imshow(racmo_sim_5d5[t,2*54:3*54,54:2*54], cmap = 'RdBu_r')
plt.clim(0,5)

fig=plt.subplot(1,4,2)
plt.imshow(racmo_sim_5d5_bicubic[t,2*54:3*54,54:2*54], cmap = 'RdBu_r')
plt.clim(0,5)

fig=plt.subplot(1,4,3)
plt.imshow(racmo_sim_27[t,2*11:3*11,11:2*11], cmap = 'RdBu_r')
plt.clim(0,5)

fig=plt.subplot(1,4,4)
plt.imshow(racmo_sim_27_area[t,2*11:3*11,11:2*11], cmap = 'RdBu_r')
plt.clim(0,5)

'''
#Only Run Once
racmo_sim_5d5 = np.load('G:/My Drive/SRM/Experiment/Results_NPY/RACMO_5d5_Daily.npy')
racmo_sim_27 = np.load('G:/My Drive/SRM/Experiment/Results_NPY/RACMO_27_Daily.npy')

racmo_sim_5d5r=np.zeros(racmo_sim_5d5.shape)
racmo_sim_27r=np.zeros(racmo_sim_27.shape)

for t in range(racmo_sim_5d5.shape[0]):
  racmo_sim_5d5r[t,:,:]= cv2.resize(racmo_sim_27[t,:,:], (216,270), interpolation = cv2.INTER_CUBIC)
  racmo_sim_27r[t,:,:]= cv2.resize(racmo_sim_5d5[t,:,:], (44,55), interpolation = cv2.INTER_AREA)

np.save('G:/My Drive/SRM/Experiment/Results_NPY/RACMO_5d5r_Daily_corr.npy',racmo_sim_5d5r)
np.save('G:/My Drive/SRM/Experiment/Results_NPY/RACMO_27r_Daily_corr.npy',racmo_sim_27r)

for model_i in models:
    model_names=os.path.basename(model_i)
    mapply.Daily_NPY(model_names, 'solo',s_var, stand=stand_mode, CSO={'CMAE_loss':SL.CMAE_loss, 'SSIM_Loss':SL.SSIM_Loss,'SMLT_Loss':SL.SMLT_Loss},topo=4)
mapply.Daily_NPY(model_names, 'solo',s_var, stand=stand_mode, CSO={'CMAE_loss':SL.CMAE_loss, 'SSIM_Loss':SL.SSIM_Loss,'SMLT_Loss':SL.SMLT_Loss},topo=4)
mapply.Daily_RACMO(True)
mapply.Daily_RACMO_27(True)

def Yearly_RACMO_resized(write_out):
  racmo_sim_5d5_bicubic = np.load('G:/My Drive/SRM/Experiment/Results_NPY/RACMO_5d5r_Daily_corr.npy')
  racmo_sim_27_area = np.load('G:/My Drive/SRM/Experiment/Results_NPY/RACMO_27r_Daily_corr.npy')

  Y5d5=np.zeros((54*5,54*4,18*12))
  Y27=np.zeros((11*5,11*4,18*12))

  t_stamps = np.arange('2001-01-01', '2019-09-01', dtype='datetime64')
  t_stamps=t_stamps.astype(datetime.datetime)
  t_stamps_years = np.array([i.year for i in t_stamps])
  t_stamps_months = np.array([i.month for i in t_stamps])

  k=0
  for y in range(2001,2019):
    s=np.where((np.array(t_stamps_years)==y) & (np.array(t_stamps_months)==7))[0][0]
    e=np.where((np.array(t_stamps_years)==(y+1)) & (np.array(t_stamps_months)==7))[0][0]
    Y5d5[:,:,k]=np.sum(racmo_sim_5d5_bicubic[int(s):int(e),:,:],axis=0)
    Y27[:,:,k]=np.sum(racmo_sim_27_area[int(s):int(e),:,:],axis=0)
    k+=1

  if write_out:
    np.save('G:/My Drive/SRM/Experiment/Results_NPY/RACMO_5d5r_Yearly_corr.npy',Y5d5)
    np.save('G:/My Drive/SRM/Experiment/Results_NPY/RACMO_27r_Yearly_corr.npy',Y27)

Yearly_RACMO_resized(True)
'''

'''
train_Xa,  train_Ya,  train_Za, val_Xa,  val_Ya,  val_Za, test_Xa, test_Ya, test_Za = ultility.overlap_evaluation(s_var, stand=stand_mode, step=2, topo=4, aoi_nr=aoi_nr, \
      SI=False, MO=['10','11','12','01','02','03','04','05','06'])

del train_Za,val_Za, test_Za

idx_model = 10
SR_model = tf.keras.models.load_model(models[idx_model],custom_objects=None, compile=False)

model_names=os.path.basename(models[idx_model])

'''

racmo_sim_5d5 = np.load('G:/My Drive/SRM/Experiment/Results_NPY/RACMO_5d5_Daily.npy')
racmo_sim_5d5_bicubic = np.load('G:/My Drive/SRM/Experiment/Results_NPY/RACMO_5d5r_Daily.npy')

SR_pred_SRResNet = np.load('G:/My Drive/SRM/Experiment/Results_NPY/V2_Results_Final_SRResNet_Vanille_LossALL_MSE_Smlt3_Emb0_Bn16_NoStand_Oct-June_Padded2_aoiall_Daily.npy')
SR_pred_HAN = np.load('G:/My Drive/SRM/Experiment/Results_NPY/V2_Results_Final_HAN_Vanille_LossALL_MSE_Smlt3_Emb0_Bn16_NoStand_Oct-June_Padded2_aoiall_Daily.npy')
SR_pred_Swin = np.load('G:/My Drive/SRM/Experiment/Results_NPY/V2_Results_Final_Swin_Vanille_LossALL_MASE_Smlt3_Emb64_Bn16_NoStand_Oct-June_Padded2_aoiall_Daily.npy')

Mask = np.zeros((54*5,54*4))
index=[(0,0),(1,0),(1,1),(2,0),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3),(4,1),(4,2),(4,3)]
k=1
for AOI_NR in range(1,14): 
    rs=index[AOI_NR-1][0]
    cs=index[AOI_NR-1][1]
    Mask[(54*rs):(54*rs+54),(54*cs):(54*cs+54)] = np.load('G:/My Drive/Paper_3/DATA/BATCH/Aux_Data/Output/mask2d/mask2d_5500_epsg3031_AOI_'+str(AOI_NR)+'.npy')
    k+=1
Mask[Mask<0.5]=0
Mask[Mask>=0.5]=1

## FIG 3
R2_img_bc = np.zeros((270, 216))
RMSE_img_bc = np.zeros((270, 216))
MAE_img_bc = np.zeros((270, 216))


R2_img_res = np.zeros((270, 216))
RMSE_img_res = np.zeros((270, 216))
MAE_img_res = np.zeros((270, 216))


R2_img_han = np.zeros((270, 216))
RMSE_img_han = np.zeros((270, 216))
MAE_img_han = np.zeros((270, 216))


R2_img_swin = np.zeros((270, 216))
RMSE_img_swin = np.zeros((270, 216))
MAE_img_swin = np.zeros((270, 216))



for i in range(270):
    for j in range(216):
        if Mask[i,j]!=0:
            R2_img_bc[i,j]= scipy.stats.pearsonr(racmo_sim_5d5_bicubic[:,i,j],racmo_sim_5d5[:,i,j])[0]**2
            RMSE_img_bc[i,j]= (np.square(racmo_sim_5d5_bicubic[:,i,j]-racmo_sim_5d5[:,i,j]).mean())**0.5
            MAE_img_bc[i,j]= abs(racmo_sim_5d5_bicubic[:,i,j]-racmo_sim_5d5[:,i,j]).mean()

            R2_img_res[i,j]= scipy.stats.pearsonr(SR_pred_SRResNet[:,i,j],racmo_sim_5d5[:,i,j])[0]**2
            RMSE_img_res[i,j]= (np.square(SR_pred_SRResNet[:,i,j]-racmo_sim_5d5[:,i,j]).mean())**0.5
            MAE_img_res[i,j]= abs(SR_pred_SRResNet[:,i,j]-racmo_sim_5d5[:,i,j]).mean()

            R2_img_han[i,j]= scipy.stats.pearsonr(SR_pred_HAN[:,i,j],racmo_sim_5d5[:,i,j])[0]**2
            RMSE_img_han[i,j]= (np.square(SR_pred_HAN[:,i,j]-racmo_sim_5d5[:,i,j]).mean())**0.5
            MAE_img_han[i,j]= abs(SR_pred_HAN[:,i,j]-racmo_sim_5d5[:,i,j]).mean()

            R2_img_swin[i,j]= scipy.stats.pearsonr(SR_pred_Swin[:,i,j],racmo_sim_5d5[:,i,j])[0]**2
            RMSE_img_swin[i,j]= (np.square(SR_pred_Swin[:,i,j]-racmo_sim_5d5[:,i,j]).mean())**0.5
            MAE_img_swin[i,j]= abs(SR_pred_Swin[:,i,j]-racmo_sim_5d5[:,i,j]).mean()


import matplotlib

import cmocean
import cmocean.cm as cmo


cbs=  False
fig = plt.figure(figsize=(24,16))
   
plt.subplot(3, 4, 1)
plt.imshow(R2_img_bc, cmap = cmo.haline)
plt.clim(0,1)
plt.xticks([])
plt.yticks([])
plt.title('Bicubic-MSE', fontweight='bold', size='large')
plt.imshow(Mask, cmap=matplotlib.colors.ListedColormap(['lightsteelblue', 'none']))
plt.text(100,80, 'Weddell Sea', color='blue',rotation=0,fontsize=12,fontstyle='italic')
plt.text(42,266, 'Bellingshausen', color='blue',rotation=310,fontsize=12,fontstyle='italic')
plt.text(30,230, 'Sea', color='blue',rotation=310,fontsize=12,fontstyle='italic')
plt.text(180,30, '(a)', color='black',fontsize=15,weight=3)

plt.subplot(3, 4, 2)
plt.imshow(R2_img_res , cmap = cmo.haline)
plt.clim(0,1)
plt.xticks([])
plt.yticks([])
plt.title('SRResNet-MSE', fontweight='bold', size='large')
plt.imshow(Mask, cmap=matplotlib.colors.ListedColormap(['lightsteelblue', 'none']))
#plt.text(110,100, 'Weddell Sea', color='blue',rotation=330,fontsize=12,fontstyle='italic')
plt.text(180,30, '(b)', color='black',fontsize=15,weight=3)

plt.subplot(3, 4, 3)
plt.imshow(R2_img_han , cmap = cmo.haline)
plt.clim(0,1)
plt.xticks([])
plt.yticks([])
plt.title('HAN-MSE', fontweight='bold', size='large')
plt.imshow(Mask, cmap=matplotlib.colors.ListedColormap(['lightsteelblue', 'none']))
#plt.text(110,100, 'Weddell Sea', color='blue',rotation=330,fontsize=12,fontstyle='italic')
plt.text(180,30, '(c)', color='black',fontsize=15,weight=3)

plt.subplot(3, 4, 4)
plt.imshow(R2_img_swin, cmap = cmo.haline)
plt.clim(0,1)
if cbs:
  cbar=plt.colorbar(fraction=0.02)
  cbar.set_label('$R^2$',fontsize=15, color='black',weight=2, labelpad=40, rotation=270)
  cbar.ax.tick_params(labelsize=15) 
plt.xticks([])
plt.yticks([])
plt.title('Swin Transformer-MSE', fontweight='bold', size='large')
#plt.xlabel('$R^2$',fontsize=15, color='black',weight=2, labelpad=20)
plt.imshow(Mask, cmap=matplotlib.colors.ListedColormap(['lightsteelblue', 'none']))
#plt.text(110,100, 'Weddell Sea', color='blue',rotation=330,fontsize=12,fontstyle='italic')
plt.text(180,30, '(d)', color='black',fontsize=15,weight=3)


# ------------------------------------------------------------------------------
plt.subplot(3, 4, 5)
plt.imshow(RMSE_img_bc, cmap = cmo.amp)
plt.clim(0,2)
plt.xticks([])
plt.yticks([])
plt.imshow(Mask, cmap=matplotlib.colors.ListedColormap(['lightsteelblue', 'none']))
#plt.text(110,100, 'Weddell Sea', color='blue',rotation=330,fontsize=12,fontstyle='italic')
plt.text(180,30, '(e)', color='black',fontsize=15,weight=3)

plt.subplot(3, 4, 6)
plt.imshow(RMSE_img_res, cmap = cmo.amp)
plt.clim(0,2)
plt.xticks([])
plt.yticks([])
plt.imshow(Mask, cmap=matplotlib.colors.ListedColormap(['lightsteelblue', 'none']))
#plt.text(110,100, 'Weddell Sea', color='blue',rotation=330,fontsize=12,fontstyle='italic')
plt.text(180,30, '(f)', color='black',fontsize=15,weight=3)

plt.subplot(3, 4, 7)
plt.imshow(RMSE_img_han, cmap = cmo.amp)
plt.clim(0,2)
plt.xticks([])
plt.yticks([])
plt.imshow(Mask, cmap=matplotlib.colors.ListedColormap(['lightsteelblue', 'none']))
#plt.text(110,100, 'Weddell Sea', color='blue',rotation=330,fontsize=12,fontstyle='italic')
plt.text(180,30, '(g)', color='black',fontsize=15,weight=3)

plt.subplot(3, 4, 8)
plt.imshow(RMSE_img_swin, cmap = cmo.amp)
plt.clim(0,2)
if cbs:
  cbar = plt.colorbar( ticks=[0, 0.5, 1,1.5,2],fraction=0.02)
  cbar.ax.set_yticklabels(['0', '0.5', '1.0','1.5', r'$\geq$'+'2.0'])
  cbar.set_label('RMSE [mm.w.e. per day]',fontsize=15, color='black',weight=2, labelpad=20, rotation=270)
  cbar.ax.tick_params(labelsize=15) 
plt.xticks([])
plt.yticks([])
#plt.xlabel('RMSE [mm.w.e]',fontsize=15, color='black',weight=2, labelpad=10)

plt.imshow(Mask, cmap=matplotlib.colors.ListedColormap(['lightsteelblue', 'none']))
#plt.text(110,100, 'Weddell Sea', color='blue',rotation=330,fontsize=12,fontstyle='italic')
plt.text(180,30, '(h)', color='black',fontsize=15,weight=3)

# ------------------------------------------------------------------------------
plt.subplot(3, 4, 9)
plt.imshow(MAE_img_bc, cmap = cmo.amp)
plt.clim(0,1)
plt.xticks([])
plt.yticks([])
plt.imshow(Mask, cmap=matplotlib.colors.ListedColormap(['lightsteelblue', 'none']))
#plt.text(110,100, 'Weddell Sea', color='blue',rotation=330,fontsize=12,fontstyle='italic')
plt.text(180,30, '(i)', color='black',fontsize=15,weight=3)

plt.subplot(3, 4, 10)
plt.imshow(MAE_img_res, cmap = cmo.amp)
plt.clim(0,1)
plt.xticks([])
plt.yticks([])
plt.imshow(Mask, cmap=matplotlib.colors.ListedColormap(['lightsteelblue', 'none']))
#plt.text(110,100, 'Weddell Sea', color='blue',rotation=330,fontsize=12,fontstyle='italic')
plt.text(180,30, '(j)', color='black',fontsize=15,weight=3)

plt.subplot(3, 4, 11)
plt.imshow(MAE_img_han, cmap = cmo.amp)
plt.clim(0,1)
plt.xticks([])
plt.yticks([])
plt.imshow(Mask, cmap=matplotlib.colors.ListedColormap(['lightsteelblue', 'none']))
#plt.text(110,100, 'Weddell Sea', color='blue',rotation=330,fontsize=12,fontstyle='italic')
plt.text(180,30, '(k)', color='black',fontsize=15,weight=3)

plt.subplot(3, 4, 12)
plt.imshow(MAE_img_swin, cmap = cmo.amp)
plt.clim(0,1)
if cbs:
  cbar=plt.colorbar( ticks=[0, 0.2, 0.4,0.6,0.8,1.0],fraction=0.02)
  cbar.ax.set_yticklabels(['0.0','0.2', '0.4', '0.6','0.8', r'$\geq$'+'1.0'])
  cbar.set_label('MAE [mm.w.e. per day]',fontsize=15, color='black',weight=2, labelpad=20, rotation=270)
  cbar.ax.tick_params(labelsize=15) 
plt.xticks([])
plt.yticks([])
#plt.xlabel('MAE [mm.w.e]',fontsize=15, color='black',weight=2, labelpad=10)
plt.imshow(Mask, cmap=matplotlib.colors.ListedColormap(['lightsteelblue', 'none']))
#plt.text(110,100, 'Weddell Sea', color='blue',rotation=330,fontsize=12,fontstyle='italic')
plt.text(180,30, '(l)', color='black',fontsize=15,weight=3)

fig.subplots_adjust(wspace=-0.63)
fig.subplots_adjust(hspace=0.02)

plt.savefig('C:/Users/zh_hu/Desktop/ARM5/Acc_stat_V5.png', dpi=300,transparent=True) # Turn on cbs=True and add _cb

## FIG 4


from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pandas as pd

QGIS_MCB=pd.read_csv('C:/Users/zh_hu/Desktop/ARM5/Figures/Melt_White.csv',header=None)
colorarray=[]
for i in range(20):
  temp=list(QGIS_MCB.iloc[i,:].values/255)
  colorarray.append(temp)
cmap_meltD = ListedColormap(colorarray)
cmap_meltC= LinearSegmentedColormap.from_list('testCmap', colors=colorarray, N=256)


import matplotlib.gridspec as gridspec
import xarray as xr
import rasterio

racmo_sim_Y_5d5 = np.load('G:/My Drive/Paper_3/output_R5_Yearly_Masked.npy')
racmo_sim_Y_5d5r = np.load('G:/My Drive/SRM/Experiment/Results_NPY/RACMO_5d5r_Yearly_corr.npy')
racmo_sim_Y_27 = np.load('G:/My Drive/Paper_3/output_R27_Yearly.npy')
racmo_sim_Y_27r = np.load('G:/My Drive/SRM/Experiment/Results_NPY/RACMO_27r_Yearly_corr.npy')

QSCAT = xr.open_rasterio('G:/My Drive/Paper_3/DATA/QSCAT_AP_GRID2.tif')

SR_pred_Y_SRResNet = np.load('G:/My Drive/SRM/Experiment/Results_NPY/V2_Results_Final_SRResNet_Vanille_LossALL_MSE_Smlt3_Emb0_Bn16_NoStand_Oct-June_Padded2_aoiall_Yearly.npy')
SR_pred_Y_HAN = np.load('G:/My Drive/SRM/Experiment/Results_NPY/V2_Results_Final_HAN_Vanille_LossALL_MSE_Smlt3_Emb0_Bn16_NoStand_Oct-June_Padded2_aoiall_Yearly.npy')
SR_pred_Y_Swin = np.load('G:/My Drive/SRM/Experiment/Results_NPY/V2_Results_Final_Swin_Vanille_LossALL_MASE_Smlt3_Emb64_Bn16_NoStand_Oct-June_Padded2_aoiall_Yearly.npy')


colname = [r'RACMO2$\rm{_{27km}}$',r'RACMO2$\rm{_{27 \leftarrow 5.5km}}$',r'RACMO2$\rm{_{5.5km}}$','Bicubic','SRResNet','HAN','Swin Tran.', 'QuikSCAT']

fig = plt.figure(figsize=(25,25.5))
gs = gridspec.GridSpec(8, 8)
fig.subplots_adjust(wspace=0.01)
fig.subplots_adjust(hspace=0.01)

input_images = [ racmo_sim_Y_27, racmo_sim_Y_27r, racmo_sim_Y_5d5, racmo_sim_Y_5d5r, SR_pred_Y_SRResNet, SR_pred_Y_HAN, SR_pred_Y_Swin, QSCAT ]

for idx_p in range(8):
  for idx_y in range(8):  
    ax = plt.subplot(gs[idx_y,idx_p])
    if idx_p==7:
      r = int(335/5)
      c = int(268/4)
      plt.imshow((input_images[idx_p][idx_y+2][2*r:3*r,c:2*c].data), cmap = cmo.thermal)
    elif idx_p<2:
      plt.imshow(input_images[idx_p][2*11:3*11,11:2*11,idx_y], cmap = cmo.thermal)  
    else:
      plt.imshow(input_images[idx_p][2*54:3*54,54:2*54,idx_y], cmap = cmo.thermal)
    plt.clim(0,400)
    plt.xticks([])
    plt.yticks([])
    if idx_y==0:
      ax.set_title(colname[idx_p], fontweight='bold',fontsize=20)
    if idx_p==0:
      ax.set_ylabel("Year {}".format(idx_y+2001), rotation=90, fontweight='bold',fontsize=20)

plt.savefig('C:/Users/zh_hu/Desktop/ARM5/TS_yearlyMelt_V3_cmo.png', dpi=300,transparent=True) 

fig = plt.figure(figsize=(20,50))
plt.imshow((input_images[idx_p][idx_y+2][2*r:3*r,c:2*c].data), cmap = cmo.thermal)
plt.clim(0,400)
cbar=plt.colorbar(orientation='horizontal', pad=0.04, aspect=36)
cbar.ax.set_xticks([0,50,100,150,200,250,300,350,400])
cbar.ax.set_xticklabels(['0','50','100','150','200', '250', '300','350', r'$\geq$'+'400'])
cbar.set_label('Annual Surface Melt [mm.w.e. per year]',fontsize=30, color='black', labelpad=20, rotation=0, fontweight='bold')
cbar.ax.tick_params(labelsize=30,length=10, width=1) 

plt.savefig('C:/Users/zh_hu/Desktop/ARM5/TS_yearlyMelt_V3_cmo_cb.png', dpi=300,transparent=True) 


