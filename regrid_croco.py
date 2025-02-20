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


from src.tools import *
from src.filters import *
from src.constants import *

filename = 'croco_1h_inst_surf_2006-02-01-2006-02-28'
path_save = './data_regrid/'

#path_in = '/home/jacqhugo/Datlas_2025/DATA_Crocco/'
path_in = '/data2/nobackup/clement/Data/Lionel_coupled_run/'

DASHBOARD = True           # for Dask, turn ON for debug
N_CPU = 16                   # for //


new_dx = 0.1                # °, new resolution
method = 'conservative'     # conservative or bilinear
SAVE_FILE = True           # build and save the interpolated file

SHOW_DIFF       = False        # show a map with before/after
CHECK_RESULTS   = False        # switch to compute more precise diff
SHOW_SPACE      = False        # if CHECK_RESULTS: show diff along a spatial dimension
SHOW_TIME       = False        # if CHECK_RESULTS: show diff along time
BILI_VS_CONS    = False        # if CHECK_RESULTS: show diff bilinear - conservative


if __name__ == "__main__": 
    start = clock.time()
    new_name = path_save+filename+'_'+str(new_dx)+'deg'+'_'+method
    
    print('====================================') 
    print('* Regrid process of Croco file')
    print('* input : '+path_in+filename+'.nc')
    print('* ouput : '+new_name+'.nc')
    print('*    new res : '+str(new_dx)+'°')
    print('* method is : '+method)
    print('====================================')
    
    global client
    client = None
    if DASHBOARD:
        # sometimes dask cluster can cause problems "memoryview is too large"
        # (writing a big netcdf file for eg, hbudget_file)
        cluster = LocalCluster(n_workers=N_CPU) # threads_per_worker=1,
        
        client = Client(cluster)
        print("Dashboard at :",client.dashboard_link)

    if pathlib.Path(new_name+'.nc').is_file():
        print('-> File is here !')
        IS_HERE = True
    else: IS_HERE=False
            
            
    if ( (SAVE_FILE and not IS_HERE ) 
            or (SHOW_DIFF and not IS_HERE) 
            or (CHECK_RESULTS and not IS_HERE) ):

        print('* Opening file ...')
        ds, xgrid = open_croco_sfx_file(path_in+filename+'.nc', lazy=True, chunks={'time':100})
        
        print('* Getting land mask ...')
        ds['mask_valid'] = xr.where(~(np.isfinite(ds.SSH)),0.,1.).compute()
        
        fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=200)
        s=ax.pcolormesh(ds['mask_valid'].isel(time=0),cmap='jet')
        plt.colorbar(s,ax=ax)
        ax.set_xlabel('lon')
        ax.set_ylabel('lat')
        ax.set_title('mask_valid')
        plt.show()
        
        print('* Interpolation at mass point ...')
        
        
        # Croco is grid-C
        L_u = ['U','oceTAUX']
        L_v = ['V','oceTAUY']
        for var in L_u:
            attrs = ds[var].attrs
            ds[var] = xgrid.interp(ds[var], 'x') # .load()
            ds[var].attrs = attrs
        for var in L_v:
            attrs = ds[var].attrs
            ds[var] = xgrid.interp(ds[var], 'y') # .load()
            ds[var].attrs = attrs
        
        
        # we have variables only at rho points now
        #   so we rename the coordinates with names
        #   that xesmf understand
        ds = ds.rename({"lon_rho": "lon", "lat_rho": "lat"})
        ds = ds.set_coords(['lon','lat'])
        
        # COMPUTING AND REMOVING GEOSTROPHY ---
        # -> large scale filtering of SSH
        print('* Computing geostrophy ...')
        sigmax, sigmay = 3, 3
        ds['SSH'] = xr.where(ds['SSH']>1e5,np.nan,ds['SSH'])
        ds['SSH_LS0'] = xr.zeros_like(ds['SSH'])
        ds['SSH_LS'] = xr.zeros_like(ds['SSH'])
        # -> smoothing: spatial
        print('     2D filter')
        ds['SSH_LS0'].data = my2dfilter_over_time(ds['SSH'].values,sigmax,sigmay, len(ds.time), N_CPU, show_progress=True)
        # a warning is raised about JAX being multithreaded but its ok because this function is numpy only !

        # -> smoothing: time
        print('     time filter')

        # ds['SSH_LS'].data = mytimefilter_over_spatialXY(ds['SSH_LS0'],N_CPU)  
        ds['SSH_LS'].data = mytimefilter_over_spatialXY(ds['SSH_LS0'].values, N_CPU=N_CPU, show_progress=True)  
        #ds['SSH_LS'].data = ds['SSH_LS0'].data
        
        # mask invalid data
        ds['SSH_LS0'] = ds['SSH_LS0'].where(ds.mask_valid.data)
        ds['SSH_LS'] = ds['SSH_LS'].where(ds.mask_valid.data)
        
        
        fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=200)
        s=ax.pcolormesh(ds['SSH_LS'],cmap='jet')
        plt.colorbar(s,ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('SSH_LS')
        plt.show()
        
        # -> getting geo current from SSH
        print('     gradXY ssh')
        #   lat, lon are gridded
        glat2 = ds['lat'].where(ds['mask_valid']).values
        glon2 = ds['lon'].where(ds['mask_valid']).values
        
        fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=200)
        s=ax.pcolormesh(glat2,cmap='jet')
        plt.colorbar(s,ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('glat2')
        plt.show()
        
        #   local dlon,dlat
        dlon = (glon2[:,1:]-glon2[:,:-1]).mean()
        dlat = (glat2[1:,:]-glat2[:-1,:]).mean()
        print(dlon,dlat)
        fc = 2*2*np.pi/86164*np.sin(glat2*np.pi/180)
        gUg = ds['SSH_LS']*0.
        gVg = ds['SSH_LS']*0.
        
        # gradient metrics
        dx = dlon * xr.ufuncs.cos(xr.ufuncs.deg2rad(glat2)) * distance_1deg_equator 
        dy = ((glon2 * 0) + 1) * dlat * distance_1deg_equator
        # over time array
        for it in range(len(ds.time)):
            # this could be vectorized ...
            # centered gradient, dSSH/dx is on SSH point
            gVg[it,:,1:-1] =  grav/fc[:,1:-1] *( ds['SSH_LS'][it,:,2:].values - ds['SSH_LS'][it,:,:-2].values ) / ( dx[:,1:-1] )/2
            gUg[it,1:-1,:] = -grav/fc[1:-1,:]*( ds['SSH_LS'][it,2:,:].values  - ds['SSH_LS'][it,:-2,:].values ) / ( dy[1:-1,] )/2
        gUg[:,0,:] = gUg[:,1,:]
        gUg[:,-1,:]= gUg[:,-2,:]
        gVg[:,:,0] = gVg[:,:,1]
        gVg[:,:,-1]= gVg[:,:,-2]
        
        # adding geo current to the file
        ds['Ug'] = (ds['SSH'].dims,
                            gUg.data,
                            {'standard_name':'zonal_geostrophic_current',
                                'long_name':'zonal geostrophic current from SSH',
                                'units':'m s-1',})
        ds['Vg'] = (ds['SSH'].dims,
                            gVg.data,
                            {'standard_name':'meridional_geostrophic_current',
                                'long_name':'meridional geostrophic current from SSH',
                                'units':'m s-1',}) 
       
        print(ds.Ug)
        print(ds.Ug.values)
        fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=200)
        s=ax.pcolormesh(ds.lon,ds.lat,ds['Ug'].isel(time=0),cmap='jet')
        plt.colorbar(s,ax=ax)
        ax.set_xlabel('lon')
        ax.set_ylabel('lat')
        ax.set_title('Ug')
        plt.show()
       
        fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=200)
        s=ax.pcolormesh(ds.lon,ds.lat,ds['U'].isel(time=0),cmap='jet')
        plt.colorbar(s,ax=ax)
        ax.set_xlabel('lon')
        ax.set_ylabel('lat')
        ax.set_title('U')
        plt.show()
       
       
       # removing geostrophy
        ds['U'].data = ds['U'].values - ds['Ug'].values
        ds['V'].data = ds['V'].values - ds['Vg'].values
        ds['U'].attrs['long_name'] = 'Ageo '+ds['U'].attrs['long_name']
        ds['V'].attrs['long_name'] = 'Ageo '+ds['V'].attrs['long_name'] 
        
        print(ds.U)
        print(ds.U.values)
        
        fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=200)
        s=ax.pcolormesh(ds.lon,ds.lat,ds['U'].isel(time=0),cmap='jet')
        plt.colorbar(s,ax=ax)
        ax.set_xlabel('lon')
        ax.set_ylabel('lat')
        ax.set_title('U')
        plt.show()
        
        # fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=200)
        # s=ax.pcolormesh(ds.lon,ds.lat,ds['Ug'].isel(time=0),cmap='jet')
        # plt.colorbar(s,ax=ax)
        # ax.set_xlabel('lon')
        # ax.set_ylabel('lat')
        # ax.set_title('Ug')
        # plt.show()
        
        # removing work variables
        ds = ds.drop_vars(['SSH_LS0','SSH_LS'])
        #ds = ds.drop_vars(['temp','salt','oceTAUX','oceTAUY','Heat_flx_net','frsh_water_net','SW_rad','MLD'])
        # END COMPUTING GEOSTROPHY ---
        
        
        if method=='conservative':  
            # we need to compute the boundaries of the lat,lon.
            # here we are high res so we just remove 1 data point.
            # see https://earth-env-data-science.github.io/lectures/models/regridding.html
            ds['lon_b'] = xr.DataArray(ds.lon_u.data[1:,:], dims=["y_u", "x_u"]) # coords=[times, locs]
            ds['lat_b'] = xr.DataArray(ds.lat_v.data[:,1:], dims=["y_v", "x_v"])   
            ds = ds.set_coords(['lon_b','lat_b'])
            ds = ds.isel(x_rho=slice(0,-2),y_rho=slice(0,-2))

        # remove unused coordinates
        ds = ds.reset_coords(['lon_v','lat_v','lon_u','lat_u'],drop=True)

        # mask area where lat and lon == 0.
        lon2D = ds.lon
        lat2D = ds.lat
        lonmin = np.round( np.nanmin(np.where(lon2D.values<0.,lon2D.values,np.nan)), 1)
        lonmax = np.round( np.nanmax(np.where(lon2D.values<0.,lon2D.values,np.nan)), 1)
        latmin = np.round( np.nanmin(np.where(lat2D.values>0.,lat2D.values,np.nan)), 1)
        latmax = np.round( np.nanmax(np.where(lat2D.values>0.,lat2D.values,np.nan)), 1)
        print('* Area is:')
        print('     min lon =', lonmin)
        print('     max lon =', lonmax)
        print('     min lat =', latmin)
        print('     max lat =', latmax)
        ds['lon'] = xr.where(ds['mask_valid'])#ds.lon==0.,np.nan,ds.lon)
        ds['lat'] = xr.where(ds['mask_valid'])#ds.lat==0.,np.nan,ds.lat)
        ds['lon_b'] = xr.where(ds['mask_valid'])#ds.lon_b==0.,np.nan,ds.lon_b)
        ds['lat_b'] = xr.where(ds['mask_valid'])#ds.lat_b==0.,np.nan,ds.lat_b)
        #ds['mask_valid'] = xr.where(lon2D==0.,0.,1.)
        
        print('* Regridding ...')
        # new dataset
        ds_out = xe.util.grid_2d(lonmin, lonmax, new_dx, latmin, latmax, new_dx)       
        
        # regridder
        print('     -> compute once the operator')
        regridder = xe.Regridder(ds, ds_out, method) # bilinear conservative
        
        # regriding variables+
        # some error is raised about arrays not being C-contiguous, please ignore
        print('     -> apply the operator to variables:')
        ds_out['mask_valid'] = regridder(ds['mask_valid'])
        for namevar in list(ds.variables):
            if namevar not in ['lat', 'lon', 'lat_u', 'lon_u', 'lat_v', 'lon_v', 'time','mask_valid','lon_b','lat_b']:
                print('     '+namevar)
                ds_out[namevar] = regridder(ds[namevar])
                # masking
                ds_out[namevar] = ds_out[namevar].where(ds_out['mask_valid'])
        
        print(ds_out.U)
        print(ds_out.U.values)
        
        
        # replacing x and y with lon1D and lat1D
        ds_out['lon1D'] = ds_out.lon[0,:]
        ds_out['lat1D'] = ds_out.lat[:,0]
        ds_out = ds_out.swap_dims({'x':'lon1D','y':'lat1D'})
        
        # removing unsed dims and coordinates
        ds_out = ds_out.drop_dims(['x_b','y_b'])
        ds_out = ds_out.reset_coords(names=['lon','lat'], drop=True)
        ds_out = ds_out.rename({'lon1D':'lon','lat1D':'lat'})

        # print some stats
        print('\nOLD DATASET\n')
        print(ds)
        print('\nNEW DATASET\n')
        print(ds_out)

        print(ds.U)
        print(ds.U.values)

        print(ds_out.U)
        print(ds_out.U.values)
        raise Exception
        print('* Saving ...')
        ds_out.attrs['xesmf_method'] = method
        ds_out.compute()
        ds_out.to_netcdf(path=new_name + '.nc',mode='w')
        ds.close()
        ds_out.close()
        
        # save regridder
        regridder.to_netcdf(path_save+'regridder_'+str(new_dx)+'deg_'+method+'.nc')
        
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
        plt.figure(figsize=(9, 5),dpi=200)
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
            plt.figure(figsize=(9, 5),dpi=200)
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
        list_var = ['SSH','temp','U','V'] # ,'temp'
        N_CPU=15 # indice search in //
        
        # here we work on rho-located variables to avoid interpolation
        ds, xgrid = open_croco_sfx_file(path_in+filename+'.nc', lazy=True, chunks={'time':100})
        ds_i = xr.open_dataset(new_name+'.nc')
        time = ds_i.time
        #print(ds_i)
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
            L_indices = find_indices_ll(L_points, oldlon, oldlat, N_CPU=N_CPU)
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
        
        if BILI_VS_CONS:
            dsc = xr.open_mfdataset(path_save+'croco_1h_inst_surf_2006-02-01-2006-02-28_0.1deg_conservative.nc')
            dsb = xr.open_mfdataset(path_save+'croco_1h_inst_surf_2006-02-01-2006-02-28_0.1deg_bilinear.nc')
            
            print(dsc)
            print(dsb)
            
            
            for k, namevar in enumerate(list_var):
                fig, ax = plt.subplots(1,1,figsize = (len(list_var)*5,5),constrained_layout=True,dpi=200) 
                
                s = ax.pcolormesh(dsc.lon,dsc.lat, dsb[namevar][it,:-1,:]-dsc[namevar][it,:,:],cmap='bwr')
                plt.colorbar(s,ax=ax)
                ax.set_ylabel('lat')
                ax.set_xlabel('lon')
                ax.set_title(namevar+' bili minus cons')
            
            
            
    plt.show()