import os
from pyexpat.errors import XML_ERROR_ABORTED
from unittest import result
import mapply
import SRM
import Plot_NPY
import tensorflow as tf
import numpy as np
import datetime
import tqdm

s_var=['snowmelt','snowmelt','snowmelt']
n_var=len(s_var)
'''
#model_mode = 'SRDRN_vanille_ori_corr_smlt_smlt_smlt_bn16_noStand_Oct-June_step2_topo4_aoiall'
model_mode = 'HAN_vanille_corr_smlt_smlt_smlt_bn16_noStand_Oct-June_step2_topo4_aoiall'
model_outname='G:/My Drive/SRM/Experiment/Models/'+model_mode  '
'''
stand_mode=False
#model_names=os.path.basename(model_outname)

IDs=np.reshape(np.array(list(range(0,88464))),(304,291))
x,y = int(np.where(IDs==24492)[0]),int(np.where(IDs==24492)[1])
xs = x-77
ys = y-44

xe = x + 11 * 19
ye = y + 11 * 21



fn = 'C:/Users/zh_hu/Documents/ARSM/BATCH/Output/Variables/snowmelt_Res27000_ALL_2011.npy'

def ANT27_Monthly(ANT_27_fn, model_name,outname, CSO = None):
    model = tf.keras.models.load_model('G:/My Drive/SRM/Experiment/Models/'+model_name,custom_objects=None, compile=False)
    t_stamps = np.arange('2001-01-01', '2019-09-01', dtype='datetime64')
    data = np.load(ANT_27_fn)*86400
    print('Data loaded!')
    data_sub = data[:,xs:xe,ys:ye]
    Ts, Hs, Ws = data_sub.shape
    #np.save('G:/My Drive/SRM/Experiment/Results_NPY/RACMO27_sub_Monthly_ANT.npy', data_sub)
    del data_sub
    T,W,H = data.shape
    #output_img = np.zeros((Hs//11*54,Ws//11*54, 12))
    output_img = np.zeros((Hs//11*54,Ws//11*54))

    t_stamps=t_stamps.astype(datetime.datetime)
    t_stamps_months = np.array([i.month for i in t_stamps])
    t_stamps_years = np.array([i.year for i in t_stamps])

    k=0
    for y in tqdm.tqdm(range(2001,2019)):
        for m in range(1,13):
            print('YM',y,m)
            
            tloc=np.where((t_stamps_months==m) & (t_stamps_years==y))[0]
            print(tloc)

            for i in range(Hs//11):
                for j in range(Ws//11):

                    print('ij',i,j)
            
                    I,J = (xs + i*11), (ys + j*11)
                    raw = data[tloc,(I-2):(I+13),(J-2):(J+13)]

                    LR_in=np.zeros((len(tloc),15,15,3))
                    for b in range(3):
                        LR_in[:,:,:,b]=raw

                    del raw    

                    HR_out=model.predict(LR_in, verbose=0)
                    #print(k,':',HR_out.shape,np.sum(HR_out[tloc,10:64,10:64,0],axis=0).shape)

                    #output_img[(i*54):(i*54+54),(j*54):(j*54+54),k]=np.sum(HR_out[:,10:64,10:64,0],axis=0)
                    output_img[(i*54):(i*54+54),(j*54):(j*54+54)]=np.sum(HR_out[:,10:64,10:64,0],axis=0)
            
            k+=1

            np.save('G:/My Drive/SRM/Experiment/Results_NPY/Test/'+outname+'_Monthly_ANT_year{}_month{}.npy'.format(y,m),output_img)

            #output_img[:,(i*54):(i*54+54),(j*54):(j*54+54)] = HR_out

    #return data_sub, output_img

#model_names='V2_Results_Final_SRResNet_Vanille_LossALL_MSE_Smlt3_Emb0_Bn16_NoStand_Oct-June_Padded2_aoiall.tf'
model_names='US_Results_Final_SRResNet_Vanille_LossALL_MASE_Smlt3_Emb0_Bn16_NoStand_Oct-June_Padded2_aoiWIS.tf'
outname='SRResNet_s_noWIS'
ANT27_Monthly(fn, model_names,outname, CSO=None)

#model_names='V2_Results_Final_HAN_Vanille_LossALL_MSE_Smlt3_Emb0_Bn16_NoStand_Oct-June_Padded2_aoiall.tf'
model_names='US_Results_Final_HAN_Vanille_LossALL_MASE_Smlt3_Emb0_Bn16_NoStand_Oct-June_Padded2_aoiWIS.tf'
outname='HAN_s_noWIS'
ANT27_Monthly(fn, model_names,outname, CSO=None)

#model_names='V2_Results_Final_Swin_Vanille_LossALL_MASE_Smlt3_Emb64_Bn16_NoStand_Oct-June_Padded2_aoiall.tf'
model_names='US_Results_Final_Swin_Vanille_LossALL_MASE_Smlt3_Emb64_Bn16_NoStand_Oct-June_Padded2_aoiWIS.tf'
outname='SWIN_s_noWIS'
ANT27_Monthly(fn, model_names,outname, CSO=None)


model_names='V2_Results_Final_SRResNet_Vanille_LossALL_MASE_Smlt3_Emb0_Bn16_NoStand_Oct-June_Padded2_aoiall.tf'
outname='SRResNet_s_MASE'
ANT27_Monthly(fn, model_names,outname, CSO=None)

model_names='V2_Results_Final_HAN_Vanille_LossALL_MASE_Smlt3_Emb0_Bn16_NoStand_Oct-June_Padded2_aoiall.tf'
outname='HAN_s_MASE'
ANT27_Monthly(fn, model_names,outname, CSO=None)


os.system("shutdown /s /t 1")



import rasterio
import numpy as np

IDs=np.reshape(np.array(list(range(0,88464))),(304,291))
x,y = int(np.where(IDs==24492)[0]),int(np.where(IDs==24492)[1])
xs = x-77
ys = y-44

xe = x + 11 * 19
ye = y + 11 * 21

data=rasterio.open('C:/Users/zh_hu/Documents/ARSM/BATCH/Output/v10m/v10m_Res27000_Y2012_M10_epsg3031.tif')
data_read=data.read()[0]
data_sub = data_read[xs:xe,ys:ye]


# make a copy of the geotiff metadata
new_meta = data.meta.copy()

# create a translation transform to shift the pixel coordinates
crop = rasterio.Affine.translation(ys, xs)

# prepend the pixel translation to the original geotiff transform
new_xform = data.transform * crop

# update the geotiff metadata with the new dimensions and transform
new_meta['width'] = ye-ys
new_meta['height'] = xe-xs
new_meta['count'] = 1
new_meta['transform'] = new_xform

# write the cropped geotiff to disk
with rasterio.open('C:/Users/zh_hu/Documents/test_crop.tif', "w", **new_meta) as dest:
    dest.write(data_sub.reshape(1,data_sub.shape[0], data_sub.shape[1]))

import glob

def Month_to_Yearly_TIF(model_name,year,out_name):
    TIFF = rasterio.open('C:/Users/zh_hu/Desktop/ARM5/Data/R5500.tif')

    files1 = glob.glob('G:/My Drive/SRM/Experiment/Results_NPY/Test/'+model_name+'*'+str(year)+'*')[6:]
    files2 = glob.glob('G:/My Drive/SRM/Experiment/Results_NPY/Test/'+model_name+'*'+str(year+1)+'*')[:6]

    results = np.zeros((1404,1350,12))

    for i in range(6):
        results[:,:,i]=np.load(files1[i])
        results[:,:,(i+6)]=np.load(files2[i])

    out_img = np.sum(results,axis=2)

    new_meta = TIFF.meta.copy()

    with rasterio.open(out_name, "w", **new_meta) as dest:
        dest.write(out_img.reshape(1,out_img.shape[0], out_img.shape[1]))

    
Month_to_Yearly_TIF('Swin',2001,'C:/Users/zh_hu/Desktop/ARM5/Data/Swin_s_2001_v1.tif')
Month_to_Yearly_TIF('Swin',2005,'C:/Users/zh_hu/Desktop/ARM5/Data/Swin_s_2005_v1.tif')

Month_to_Yearly_TIF('HAN',2001,'C:/Users/zh_hu/Desktop/ARM5/Data/HAN_s_2001_v1.tif')
Month_to_Yearly_TIF('HAN',2005,'C:/Users/zh_hu/Desktop/ARM5/Data/HAN_s_2005_v1.tif')

Month_to_Yearly_TIF('SRResNET',2001,'C:/Users/zh_hu/Desktop/ARM5/Data/SRResNET_s_2001_v1.tif')
Month_to_Yearly_TIF('SRResNET',2005,'C:/Users/zh_hu/Desktop/ARM5/Data/SRResNET_s_2005_v1.tif')

Month_to_Yearly_TIF('Swin_s_noWIS',2005,'C:/Users/zh_hu/Desktop/ARM5/Data/Swin_s_NoWIS_2005_v1.tif')
Month_to_Yearly_TIF('HAN_s_noWIS',2005,'C:/Users/zh_hu/Desktop/ARM5/Data/HAN_s_NoWIS_2005_v1.tif')
Month_to_Yearly_TIF('SRResNet_s_noWIS',2005,'C:/Users/zh_hu/Desktop/ARM5/Data/SRResNet_s_NoWIS_2005_v1.tif')

Month_to_Yearly_TIF('HAN_s_MASE',2005,'C:/Users/zh_hu/Desktop/ARM5/Data/HAN_s_MASE_2005_v1.tif')
Month_to_Yearly_TIF('SRResNet_s_MASE',2005,'C:/Users/zh_hu/Desktop/ARM5/Data/SRResNet_s_MASE_2005_v1.tif')


import xarray as xr

def NC_to_TIFF(fn, param, res, out_fn_var, exe=False):
    out_fn_proj=out_fn_var[:-4]+'_epsg3031.tif'
    nc_in=xr.open_dataset(fn)
    rlat_in=nc_in['rlat'].data
    rlon_in=nc_in['rlon'].data
    proj=(nc_in['rotated_pole'].attrs)['proj4_params']
    nc_in.close()
    
    code_1 ='gdal_translate NETCDF:"{}":{} -a_ullr {} {} {} {} {}'.format(fn,param,min(rlon_in),max(rlat_in),max(rlon_in),min(rlat_in),out_fn_var)
    if exe:
        print('[Prosessing Start (GDAL)]: Concert NetCDF to GeoTIFF')
        os.system(code_1)
    print(code_1)
    print()
        
    code_2 ='gdalwarp -s_srs "{}" -t_srs "EPSG:3031" -tr {} -{} -r near {} {}'.format(proj,res,res,out_fn_var,out_fn_proj)
    if exe:
        print('[Prosessing Start (GDAL)]: Reproject to EPSG: 3031')
        os.system(code_2)
    print(code_2)
    print()

NC_to_TIFF('C:/Users/zh_hu/Documents/snowmelt_RACMO2.3p2_yearly_ANT27_1979_2017.nc','snowmelt',27000,'snowmelt_yearly.tif',exe=True)