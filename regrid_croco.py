import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
import xesmf as xe
from dask.distributed import Client,LocalCluster
import time as clock

from tools import *

filename = '/home/jacqhugo/Datlas_2025/DATA_Crocco/croco_1h_inst_surf_2006-02-01-2006-02-28.nc'
DASHBOARD = False
new_dx = 1 # °
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

    #ds = xr.open_dataset(filename)
    ds, _ = open_croco_sfx_file(filename, lazy=True, chunks={'time':100})
    ds = ds.rename({"lon_rho": "lon", "lat_rho": "lat"})
    ds = ds.set_coords(['lon','lat'])

    temp = ds.temp

    # before
    plt.figure(figsize=(9, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    temp[0].plot.pcolormesh(ax=ax, x='lon', y='lat')
    ax.coastlines()
    ax.set_extent([-80, -36, 22, 50], crs=ccrs.PlateCarree())

    ds['lon'] = xr.where(ds.lon==0.,np.nan,ds.lon)
    ds['lat'] = xr.where(ds.lon==0.,np.nan,ds.lat)

    print(ds)

    ds_out = xe.util.grid_2d(-80, -30, new_dx, 20, 50, new_dx)
    regridder = xe.Regridder(ds, ds_out, "bilinear")
    temp_out = regridder(temp)
    ds_out['temp'] = temp_out
    print(ds_out['lat'].values)
    ds_out['lon1D'] = ds_out.lon[0,:]
    ds_out['lat1D'] = ds_out.lat[:,0]
    print(ds_out['lon1D'].values)
    print(ds_out['lat1D'].values)
    ds_out = ds_out.swap_dims({'x':'lon1D','y':'lat1D'})
    print(ds_out)

    plt.figure(figsize=(9, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ds_out.temp[0].plot.pcolormesh(ax=ax, x='lon1D', y='lat1D')
    ax.coastlines()
    ax.set_title('new')
    ax.set_extent([-80, -36, 22, 50], crs=ccrs.PlateCarree())

    print(temp)
    print(temp_out)
    
    # plt.scatter(ds['lon'][::2], ds['lat'][::2], s=0.01)  # plot grid locations
    # plt.ylim([-90, 90])
    # plt.xlabel("lon")
    # plt.ylabel("lat")
    end = clock.time()
    print('Total execution time = '+str(np.round(end-start,2))+' s')
    plt.show()


