"""
This script regrid Croco ouput to a lower resolution.
The model is on a C-grid and so some interpolation are needed, 
    this requires quit a lot of memory so a HPC is highly recommended.
    
Note: I still need to verify results.

"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
import xesmf as xe
from dask.distributed import Client,LocalCluster
import time as clock
import pathlib
import os
import sys 

from src.tools import *

path_in = '/home/jacqhugo/Datlas_2025/DATA_Crocco/'
filename = 'croco_1h_inst_surf_2006-02-01-2006-02-28'
path_save = './data_regrid/'

# path_in = '/data2/nobackup/clement/Data/Lionel_coupled_run/'
# filename = 'croco_1h_inst_surf_2006-02-01-2006-02-28'
# path_save = './data_regrid/'

DASHBOARD = False           # for Dask, turn ON for debug
new_dx = 0.1                # °, new resolution

SAVE_FILE = False           # build and save the interpolated file
SHOW_DIFF = False           # show a map with before/after

CHECK_RESULTS = True        # switch to compute more precise diff
SHOW_SPACE = False          # show diff along a spatial dimension
SHOW_TIME = True            # show diff along time



if __name__ == "__main__": 
    start = clock.time()
    new_name = path_save+filename+'_'+str(new_dx)+'deg'
    
    print('====================================') 
    print('* Regrid process of Croco file')
    print('* input : '+path_in+filename+'.nc')
    print('* ouput : '+new_name+'.nc')
    print('*    new res : '+str(new_dx)+'°')
    print('====================================')
    
    global client
    client = None
    if DASHBOARD:
        # sometimes dask cluster can cause problems "memoryview is too large"
        # (writing a big netcdf file for eg, hbudget_file)
        cluster = LocalCluster(n_workers=8) # threads_per_worker=1,
        
        client = Client(cluster)
        print("Dashboard at :",client.dashboard_link)

    if pathlib.Path(new_name+'.nc').is_file():
        print('-> File is here !')
        IS_HERE = True
    else: IS_HERE=False
            
            
    if ( (SAVE_FILE and not IS_HERE ) 
            or (SHOW_DIFF and not IS_HERE) 
            or (CHECK_RESULTS and not IS_HERE) ):

        print('* Opening file')
        ds, xgrid = open_croco_sfx_file(path_in+filename+'.nc', lazy=True, chunks={'time':100})

        print('* Interpolation at mass point...')
        L_u = ['U','oceTAUX']
        L_v = ['V','oceTAUY']
        for var in L_u:
            attrs = ds[var].attrs
            ds[var] = xgrid.interp(ds[var].load(), 'x')
            ds[var].attrs = attrs
        for var in L_v:
            attrs = ds[var].attrs
            ds[var] = xgrid.interp(ds[var].load(), 'y')
            ds[var].attrs = attrs
        
        # we have variables only at rho points now
        ds = ds.rename({"lon_rho": "lon", "lat_rho": "lat"})
        ds = ds.set_coords(['lon','lat'])

        # mask area where lat and lon == 0.
        lon2D = ds.lon
        lat2D = ds.lat
        lonmin = np.round( np.nanmin(np.where(lon2D.values<0.,lon2D.values,np.nan)), 1)
        lonmax = np.round( np.nanmax(np.where(lon2D.values<0.,lon2D.values,np.nan)), 1)
        latmin = np.round( np.nanmin(np.where(lat2D.values>0.,lat2D.values,np.nan)), 1)
        latmax = np.round( np.nanmax(np.where(lat2D.values>0.,lat2D.values,np.nan)), 1)
        print('     min lon =', lonmin)
        print('     max lon =', lonmax)
        print('     min lat =', latmin)
        print('     max lat =', latmax)
        ds['lon'] = xr.where(ds.lon==0.,np.nan,ds.lon)
        ds['lat'] = xr.where(ds.lon==0.,np.nan,ds.lat)
        ds['mask'] = xr.where(lon2D==0.,0.,1.)
        # new dataset
        ds_out = xe.util.grid_2d(lonmin, lonmax, new_dx, latmin, latmax, new_dx)
        # regridder
        regridder = xe.Regridder(ds, ds_out, "bilinear", keep_attrs=True)
        
        # regriding variable
        print('* Regridding ...')
        ds_out['mask'] = regridder(ds['mask'])
        for namevar in list(ds.variables):
            if namevar not in ['lat', 'lon', 'lat_u', 'lon_u', 'lat_v', 'lon_v', 'time','mask']:
                print('     '+namevar)
                ds_out[namevar] = regridder(ds[namevar])
                # masking
                ds_out[namevar] = ds_out[namevar].where(ds_out['mask'])
        
        # replacing x and y with lon1D and lat1D
        ds_out['lon1D'] = ds_out.lon[0,:]
        ds_out['lat1D'] = ds_out.lat[:,0]
        ds_out = ds_out.swap_dims({'x':'lon1D','y':'lat1D'})
        # removing unsed dims and coordinates
        ds_out = ds_out.drop_dims(['x_b','y_b'])
        ds_out = ds_out.reset_coords(names=['lon','lat'], drop=True)
        ds_out = ds_out.rename({'lon1D':'lon','lat1D':'lat'})

        # print some stats
        print('OLD DATASET')
        print(ds)
        print('NEW DATASET')
        print(ds_out)


        ds_out.compute()
        ds_out.to_netcdf(path=new_name + 'nc',mode='w')
        ds.close()
        ds_out.close()
        
        # save regridder
        regridder.to_netcdf(path_save+'regridder_'+str(new_dx)+'deg.nc')
        
        end = clock.time()
        print('Total execution time = '+str(np.round(end-start,2))+' s')
        
    if SHOW_DIFF:
        print('* Visual plot of before/after')
        ds, xgrid = open_croco_sfx_file(path_in+filename+'.nc', lazy=True, chunks={'time':100})
        ds_out = xr.open_dataset(new_name+'.nc')
        it = 0
        
        # VERIFICATION
        # -> GLOBAL
        # before
        plt.figure(figsize=(9, 5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.pcolormesh(ds.lon_rho,ds.lat_rho,ds['temp'][it])
        ax.coastlines()
        ax.set_title('old SST')
        ax.set_extent([-80, -36, 22, 50], crs=ccrs.PlateCarree())
        # after
        plt.figure(figsize=(9, 5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.pcolormesh(ds_out.lon,ds_out.lat,ds_out['temp'][it])
        ax.coastlines()
        ax.set_title('new SST')
        ax.set_extent([-80, -36, 22, 50], crs=ccrs.PlateCarree())

        for namevar in ['SSH','MLD','U','V','temp','salt','oceTAUX','oceTAUY','Heat_flx_net','frsh_water_net','SW_rad']:
            plt.figure(figsize=(9, 5))
            ax = plt.axes(projection=ccrs.PlateCarree())
            s = ax.pcolormesh(ds_out.lon,ds_out.lat,ds_out[namevar][it])
            plt.colorbar(s,ax=ax)
            ax.coastlines()
            ax.set_title('new '+namevar)
            ax.set_extent([-80, -36, 22, 50], crs=ccrs.PlateCarree())

    if CHECK_RESULTS:
        print('* Checking results validity')
        
        # visual plot at a specific LAT
        LAT = 38. # °N
        LON_min = -60. #°E
        LON_max = -55. #°E
        it = 0
        list_var = ['SSH','temp'] # ,'temp'
        N_CPU=15 # indice search in //
        
        # here we work on rho-located variables to avoid interpolation
        ds, xgrid = open_croco_sfx_file(path_in+filename+'.nc', lazy=True, chunks={'time':100})
        ds_i = xr.open_dataset(new_name+'.nc')
        time = ds_i.time
        print(ds_i)
        # SST, SSH, H
        
        
        
        
        # new grid
        indy_n = nearest(ds_i.lat.values,LAT)
        new_lon_min = np.round(np.min(ds_i.lon.values),2)
        new_lon_max = np.round(np.max(ds_i.lon.values),2)
        
        # old grid 
        # is lat,lon 2D so we need to find the values at similar location
        olddx = 0.02
        oldlon = ds.lon_rho.values
        oldlat = ds.lat_rho.values
        search_lon = np.arange(LON_min,LON_max,olddx)
        dictvar = {}
        L_points = []
        
        if SHOW_SPACE:
            # find indices in //
            for ik in range(len(search_lon)):
                LON = search_lon[ik]
                L_points.append([LON,LAT])
            L_indices = find_indices_ll(L_points, oldlon, oldlat, N_CPU=15)
            for namevar in list_var:
                dictvar[namevar] = [ds[namevar][it,indices[0][1],indices[0][0]].values for indices in L_indices] 
            
            
            # plot: slice at LAT
            fig, ax = plt.subplots(len(list_var),1,figsize = (len(list_var)*5,5),constrained_layout=True,dpi=200) 
            for k, namevar in enumerate(list_var):
                ax[k].plot(search_lon, dictvar[namevar],c='k',label='truth')
                ax[k].scatter(ds_i.lon,ds_i[namevar][it,indy_n,:],c='b',label='interp',marker='x')
                ax[k].set_xlim([LON_min,LON_max])
                ax[k].set_ylabel(namevar)
            ax[-1].set_xlabel('LON')
        if SHOW_TIME:
            # plot: slice at LAT,LON_max
            select_LON = LON_min
            indx,indy = find_indices([select_LON,LAT],oldlon,oldlat)[0]
            indx_n = nearest(ds_i.lon.values,select_LON)
            fig, ax = plt.subplots(len(list_var),1,figsize = (len(list_var)*5,5),constrained_layout=True,dpi=200) 
            for k, namevar in enumerate(list_var):
                ax[k].plot(time, ds[namevar][:,indy,indx],c='k',label='truth')
                ax[k].scatter(time, ds_i[namevar][:,indy_n,indx_n],c='b',label='interp',marker='x')
                ax[k].set_ylabel(namevar)
            ax[-1].set_xlabel('time')
            ax[0].set_title('at LON,LAT='+str(select_LON)+','+str(LAT))
        
    plt.show()