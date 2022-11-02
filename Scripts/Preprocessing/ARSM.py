import xarray as xr
import os
import sys

import json
import geopandas as gpd
import numpy as np
from matplotlib import pyplot as plt
import glob

class NC_to_TIFF:

    def __init__(self, nc_fn):
        self.nc = nc_fn

    def do_NC_to_TIFF(self, param, res, out_fn_var, exe=False):
        out_fn_proj=out_fn_var[:-4]+'_epsg3031.tif'
        nc_in=xr.open_dataset(self.nc)
        rlat_in=nc_in['rlat'].data
        rlon_in=nc_in['rlon'].data
        try:
            proj=(nc_in['rotated_pole'].attrs)['proj4_params']
        except:
            if res == 27000:
                proj = '-m 57.295779506 +proj=ob_tran +o_proj=latlon +o_lat_p=-180.0 +lon_0=10.0'
                print('No Projection Info: trying -m 57.295779506 +proj=ob_tran +o_proj=latlon +o_lat_p=-180.0 +lon_0=10.0')
            elif res == 5500:
                proj = '-m 57.295779506 +proj=ob_tran +o_proj=latlon +o_lat_p=-180.0 +lon_0=30.0'
                print('No Projection Info: trying -m 57.295779506 +proj=ob_tran +o_proj=latlon +o_lat_p=-180.0 +lon_0=30.0')
            else:
                sys.exit('[Warning] No Projection Info: Ending here!')

        nc_in.close()
    
        code_1 ='gdal_translate NETCDF:"{}":{} -a_ullr {} {} {} {} {}'.format(self.nc,param,min(rlon_in),max(rlat_in),max(rlon_in),min(rlat_in),out_fn_var)
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

class AOI_clip:
     
    def __init__(self, grid_fn, tif_fn):
        self.grid = gpd.read_file(grid_fn)
        self.tif = xr.open_rasterio(tif_fn)
        
    def get_coor(self, index):
        gdf = self.grid
        gdf_sub = gdf[gdf['DN']==index]
        feature = [json.loads(gdf_sub.to_json())['features'][0]['geometry']]
        coors = feature [0]['coordinates'][0]
        x1 = (max([x[0] for x in coors])+min([x[0] for x in coors]))/2
        x2 = (max([x[1] for x in coors])+min([x[1] for x in coors]))/2
        
        return x1,x2    
    
    def get_xy(self, xs, ys):
        ds = self.tif 
        y=np.where(ds['y']==ds.sel(x=xs, y=ys, method="nearest")['y'])[0]
        x=np.where(ds['x']==ds.sel(x=xs, y=ys, method="nearest")['x'])[0]
        
        return int(x), int(y)




def latlon_to_xy(tif_in, locs):
    x0, y0 = tif_in.get_coor(locs['NW'])
    x1, y1 = tif_in.get_xy(x0,y0)
    
    x2, y2 = tif_in.get_coor(locs['SE'])
    x3, y3 = tif_in.get_xy(x2,y2)
    
    return tif_in.tif[:,y1:(y3+1),x1:(x3+1)]

def latlon_to_xy_loc(tif_in, locs):
    x0, y0 = tif_in.get_coor(locs['NW'])
    x1, y1 = tif_in.get_xy(x0,y0)
    
    x2, y2 = tif_in.get_coor(locs['SE'])
    x3, y3 = tif_in.get_xy(x2,y2)
    
    return y1, (y3+1), x1, (x3+1)

def filename_YM(fn):
    fn_name_base=os.path.basename(fn).split('_')
    return fn_name_base[2][1:]+fn_name_base[3][1:].zfill(2)

class Merge_NPY:
    def __init__(self, varname, parent_dir, aoi_nr, res,prefix):
        self.varname = varname
        self.parent_dir=parent_dir
        self.aoi_nr= aoi_nr
        self.res=res
        self.prefix = prefix

    def var_merge(self,out_filename, save_data_npy):
        var_path_label = os.path.join(self.parent_dir, self.varname, self.prefix+'AOI'+str(self.aoi_nr)) #t2m
        npy_list_label = glob.glob(var_path_label+'/*Res'+str(self.res)+'*.npy')
        npy_list_label_sorted = sorted(npy_list_label, key=filename_YM)

        for files in npy_list_label_sorted:
            print(files)

        print('-'*20)

        init_npy_label = np.load(npy_list_label_sorted[0])
        for fn_len in range(len(npy_list_label_sorted)-1):
            temp_npy_label= np.load(npy_list_label_sorted[fn_len+1])
            init_npy_label = np.concatenate((init_npy_label, temp_npy_label), axis=0)

        Labels_smlt = init_npy_label
        if save_data_npy == 'Y':
            out_fn=os.path.join(self.parent_dir,'Variables', out_filename)
            if os.path.isdir(os.path.join(self.parent_dir,'Variables')) == False:
                os.mkdir(os.path.join(self.parent_dir,'Variables'))
            np.save(out_fn, Labels_smlt)
    