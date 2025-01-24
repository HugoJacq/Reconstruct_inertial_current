import numpy as np
import scipy as sp
import xarray as xr
from constants import *
import pathlib
from datetime import timedelta

from filters import *
from tools import *

def build_LS_files(files, box, path_save):
    """
    Builds a netcdf file with smooth variable to get large scale fields.
        initial file is global, we create a subset defined by 'box'

    INPUT:
        - files: list of path for 1 variable
        - box: list of lat lon boundaries [left,right,bottom,top]
        - path_save: where to save the netcdf file
    OUPUT:
        - a netcdf file with time and space smoothing, for currents, SSH, MLD
        
    OLD FUNCTION, NOT USED ANYMORE
    """
    N_CPU = 1
    
    # opening dataset
    ds = xr.open_mfdataset(files)
    list_var = list(ds.keys())
    nt, _, _ = np.shape(ds[list_var[0]])
    
    # checking if file is present
    name_save = path_save + 'LS_fields'
    for var in list_var:
        name_save += '_'+var
    name_save += '_' + str(box[0]) + '_' + str(box[1]) + '_' + str(box[2]) + '_' + str(box[3]) + '.nc'
    if pathlib.Path(name_save).is_file():
        print(' LS file for '+str(list_var)+' is already here')
        return None       
    
    # searching for lon,lat indexes
    glon = ds.lon
    glat = ds.lat
    ix = np.where((glon>=box[0])&(glon<=box[1]))[0]
    iy = np.where((glat>=box[2])&(glat<=box[3]))[0]
    glon = glon[ix[0]:ix[-1]+1]
    glat = glat[iy[0]:iy[-1]+1]
    # selecting data in lat lon
    ds = ds.isel(lat=slice(iy[0],iy[-1]+1),lon=slice(ix[0],ix[-1]+1))
    
    data_vars = {}
    for var in list_var:
        print('     var is',var)
        ds[var] = ds[var].load() # here we load only a subset of the data
        standard_name = ds[var].attrs['standard_name']
        long_name = ds[var].attrs['long_name']
        units = ds[var].attrs['units']
        # filtering out erronous values
        ds[var] = xr.where(ds[var]>1e5,np.nan,ds[var])
        
        ds[var+'0'],ds[var+'f'] = xr.zeros_like(ds[var]),xr.zeros_like(ds[var])
        # spatial filtering
        print('     - spatial filter')
        sigmax = 3
        sigmay = 3
        ds[var+'0'].data = my2dfilter_over_time(ds[var].values,sigmax,sigmay, nt, N_CPU, ns=2)
        # time filtering
        print('     - time filter')
        ds[var+'f'] = mytimefilter(ds[var+'0'])  
        
        
        data_vars[var] = (ds[var].dims,
                                ds[var].data,
                                {'standard_name':standard_name,
                                    'long_name':long_name,
                                    'units':units,})
        data_vars[var+'_LS'] = (ds[var].dims,
                                ds[var+'f'].data,
                                {'standard_name':standard_name+'_large_scale',
                                    'long_name':long_name+' large scale',
                                    'units':units,})
        
        # geostrophic current from LS fields
        if var=='SSH':
            print('     - geostrophic currents')
            dlon=(glon[1]-glon[0]).values
            dlat=(glat[1]-glat[0]).values
            glon2,glat2 = np.meshgrid(glon,glat)
            fc = 2*2*np.pi/86164*np.sin(glat2*np.pi/180)
            gUg=ds['SSHf']*0.
            gVg=ds['SSHf']*0.
            for it in range(nt):
                gVg[it,:,1:-1] = grav/fc[:,1:-1]*(ds['SSHf'][it,:,2:].data-ds['SSHf'][it,:,:-2].data)/(dlon*110000*np.cos(glat2[:,1:-1]*np.pi/180))/2
                gUg[it,1:-1,:] = -grav/fc[1:-1,:]*(ds['SSHf'][it,2:,:].data-ds['SSHf'][it,:-2,:].data)/(dlat*110000)/2
            gUg[:,0,:]=gUg[:,1,:]
            gUg[:,-1,:]=gUg[:,-2,:]
            gVg[:,:,0]=gVg[:,:,1]
            gVg[:,:,-1]=gVg[:,:,-2]
            
            data_vars['Ug'] = (ds[var].dims,
                                gUg.data,
                                {'standard_name':'zonal_geostrophic_current',
                                    'long_name':'zonal geostrophic current from SSH',
                                    'units':'m s-1',})
            data_vars['Vg'] = (ds[var].dims,
                                gVg.data,
                                {'standard_name':'meridional_geostrophic_current',
                                    'long_name':'meridional geostrophic current from SSH',
                                    'units':'m s-1',})
    

    # ds2 = xr.open_dataset('Interp_1D_LON-24.8_LAT45.2.nc')
    # print('Ug are equal ?:',np.array_equal(ds2['Ug'],gUg))
    # raise Exception
    coords=ds.coords
    ds_LS = xr.Dataset(data_vars=data_vars,coords=coords,
                        attrs={'input_files':files,'box':str(box),'file_created_by':'build_LS_files'})            
    print(ds_LS)
    
    
    ds_LS.to_netcdf(path=name_save,mode='w')
    ds_LS.close()
    
def interp_at_model_t_1D(list_files, dt, point_loc, N_CPU, path_save, method='linear'):
    """
    Source model is : MITgcm or Crocco
    
    Builds a netcdf file with variable interpolated at the timestep dt.
        initial file is global, we create a subset defined by 'box'

    INPUT:
        - list_files: list of path for the files [filesUV,filesH,filesW,filesD]
        - dt : time step to interpolate at
        - point_loc : tuple of [lon,lat] where to work
        - N_CPU: do the spatial filter in // if > 1
        - path_save: where to save the netcdf file
        - method : interpolation method
    OUPUT:
        - a netcdf file with time interpolated at ['ir','jr'], for 
          currents, SSH, MLD. Stress is added as C.U10**2
    """   
    # checking if file is present
    name_save = path_save
    name_save += 'Interp_1D_LON'+str(point_loc[0])+'_LAT'+str(point_loc[1])+'.nc'
    if pathlib.Path(name_save).is_file():
        print('     interpolated file 1D is already here')
        return None
    
    # opening datasets
    ds = xr.open_mfdataset(list_files)
    nt = len(ds.time)
    
    # getting a reduced size area to compute large scale current
    # Area is +- 1° around 'point_loc'
    ds = ds.sel(lon=slice(point_loc[0]-1,point_loc[0]+1),
                lat=slice(point_loc[1]-1,point_loc[1]+1) )
    
    # large scale filtering of SSH
    sigmax, sigmay = 3, 3
    ds['SSH'] = xr.where(ds['SSH']>1e5,np.nan,ds['SSH'])
    gSSH_LS0 = my2dfilter_over_time(ds['SSH'].values,sigmax,sigmay, nt, N_CPU, ns=2)
    gSSH_LS = mytimefilter(gSSH_LS0)  
    
    # getting geo current from SSH
    glon2,glat2 = np.meshgrid(ds.lon.values,ds.lat.values)
    fc = 2*2*np.pi/86164*np.sin(glat2*np.pi/180)
    dlon = (ds.lon[1]-ds.lon[0]).values
    dlat = (ds.lat[1]-ds.lat[0]).values
    gUg = gSSH_LS*0.
    gVg = gSSH_LS*0.
    for it in range(nt):
        gVg[it,:,1:-1] =  grav/fc[:,1:-1]*( gSSH_LS[it,:,2:]-gSSH_LS[it,:,:-2] ) / ( dlon*110000*np.cos(glat2[:,1:-1]*np.pi/180))/2
        gUg[it,1:-1,:] = -grav/fc[1:-1,:]*( gSSH_LS[it,2:,:]-gSSH_LS[it,:-2,:] ) / ( dlat*110000)/2
    gUg[:,0,:] = gUg[:,1,:]
    gUg[:,-1,:]= gUg[:,-2,:]
    gVg[:,:,0] = gVg[:,:,1]
    gVg[:,:,-1]= gVg[:,:,-2]
     
    ds['Ug'] = (ds['SSU'].dims,
                        gUg,
                        {'standard_name':'zonal_geostrophic_current',
                            'long_name':'zonal geostrophic current from SSH',
                            'units':'m s-1',})
    ds['Vg'] = (ds['SSU'].dims,
                        gVg,
                        {'standard_name':'meridional_geostrophic_current',
                            'long_name':'meridional geostrophic current from SSH',
                            'units':'m s-1',}) 
     
    # selecting the location in 1D (time)
    ds = ds.sel(lon=point_loc[0],lat=point_loc[1])
    LON = ds.lon.values # scalar
    LAT = ds.lat.values # scalar

    # removing geostrophy
    ds['SSU'].data = ds['SSU'].values - ds['Ug'].values
    ds['SSV'].data = ds['SSV'].values - ds['Vg'].values
    
    # new time vector
    time = np.arange(ds.time.values[0], ds.time.values[-1], timedelta(seconds=dt),dtype='datetime64[ns]')
    ds_i = ds.interp({'time':time},method=method) # linear interpolation in time
        
    # stress as: C x wind**2
    gTx = ds_i.geo5_u10m
    gTy = ds_i.geo5_v10m
    gTAx = 8e-6*np.sign(gTx)*gTx**2
    gTAy = 8e-6*np.sign(gTy)*gTy**2
    
    # adding the stress to the variables
    ds_i['TAx'] = (ds_i.geo5_u10m.dims,
                        gTAx.data,
                        {'standard_name':'eastward_wind_stress_from_CdUU',
                            'long_name':'Eastward wind stress at 10m above water',
                            'units':'m2 s-2',})
    ds_i['TAy'] = (ds_i.geo5_v10m.dims,
                        gTAy.data,
                        {'standard_name':'eastward_wind_stress_from_CdUU',
                            'long_name':'Eastward wind stress at 10m above water',
                            'units':'m2 s-2',})
    
    # rename MLD with its proper name
    ds_i['MLD'] = ds_i['KPPhbl']
    ds_i = ds_i.drop_vars(['KPPhbl'])

    # saving
    print('     saving ...')
    ds_i.attrs['interp_method'] = method
    ds_i.attrs['produced_by'] = 'interp_at_model_t_1D'
    ds_i.attrs['model_dt'] = dt
    ds_i.attrs['location (LON,LAT)'] = (LON,LAT)
    
    ds_i.to_netcdf(path=name_save,mode='w')
    ds.close()
    ds_i.close()
    