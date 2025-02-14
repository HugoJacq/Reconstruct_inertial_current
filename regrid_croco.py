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

from tools import *




path_in = '/home/jacqhugo/Datlas_2025/DATA_Crocco/'
filename = 'croco_1h_inst_surf_2006-02-01-2006-02-28'
path_save = path_in

# path_in = '/data2/nobackup/clement/Data/Lionel_coupled_run/'
# filename = 'croco_1h_inst_surf_2006-02-01-2006-02-28'
# path_save = './data_regrid/'

DASHBOARD = False
new_dx = 0.1 # °
start = clock.time()
if __name__ == "__main__":  

    global client
    client = None
    if DASHBOARD:
        # sometimes dask cluster can cause problems "memoryview is too large"
        # (writing a big netcdf file for eg, hbudget_file)
        cluster = LocalCluster(n_workers=8) # threads_per_worker=1,
        
        client = Client(cluster)
        print("Dashboard at :",client.dashboard_link)

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
    regridder = xe.Regridder(ds, ds_out, "bilinear")
    
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


    # VERIFICATION
    # -> GLOBAL
    # before
    plt.figure(figsize=(9, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.pcolormesh(lon2D,lat2D,ds['temp'][0])
    ax.coastlines()
    ax.set_title('old')
    ax.set_extent([-80, -36, 22, 50], crs=ccrs.PlateCarree())
    # after
    plt.figure(figsize=(9, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.pcolormesh(ds_out.lon,ds_out.lat,ds_out['temp'][0])
    ax.coastlines()
    ax.set_title('new')
    ax.set_extent([-80, -36, 22, 50], crs=ccrs.PlateCarree())

    for namevar in ['SSH','MLD','U','V','temp','salt','oceTAUX','oceTAUY','Heat_flx_net','frsh_water_net','SW_rad']:
        plt.figure(figsize=(9, 5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.pcolormesh(ds_out.lon,ds_out.lat,ds_out[namevar][0])
        ax.coastlines()
        ax.set_title('new')
        ax.set_extent([-80, -36, 22, 50], crs=ccrs.PlateCarree())

    ds_out.compute()
    ds_out.to_netcdf(path=path_save+filename+'_'+str(new_dx)+'deg.nc',mode='w')
    ds.close()
    ds_out.close()
    
    # save regridder
    regridder.to_netcdf(path_save+'regridder_'+str(new_dx)+'deg.nc')
    
    
    end = clock.time()
    print('Total execution time = '+str(np.round(end-start,2))+' s')
    plt.show()


