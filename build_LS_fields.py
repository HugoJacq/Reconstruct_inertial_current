import numpy as np
import scipy as sp
import xarray as xr
from netCDF4 import Dataset
from constants import *
from numba import jit
from joblib import Parallel, delayed
import time as clock
import pathlib
from datetime import datetime, timedelta

@jit(nopython=True, fastmath=True)
def meshgrid(x, y, indexing='ij'):
    xx = np.empty(shape=(x.size, y.size), dtype=x.dtype)
    yy = np.empty(shape=(x.size, y.size), dtype=y.dtype)
    if indexing=='ij':
        for i in range(x.size):
            for j in range(y.size):
                xx[i,j] = i  # change to x[k] if indexing xy
                yy[i,j] = j  # change to y[j] if indexing xy
    else:
        for i in range(x.size):
            for j in range(y.size):
                xx[i,j] = x[j]  # change to x[k] if indexing xy
                yy[i,j] = y[i]  # change to y[j] if indexing xy
    return xx, yy

def my2dfilter(s,sigmax,sigmay, ns=2):
    """
    Spatial 2D filter, using gaussian kernel.
    This is used to get large scale trend of SSH

    INPUT:
        - s : signal to smooth
        - sigmax : for gaussian kernel, std in x direction
        - sigmay : for gaussian kernel, std in y direction
        - ns : width of the gaussian filter, in number of std
    OUTPUT:
        - smoothed signal 'sf', same dimensions as 's'
    """

    x, y = np.meshgrid(np.arange(-int(ns*sigmax), int(ns*sigmax)+1), np.arange(-int(ns*sigmay), int(ns*sigmay)+1), indexing='ij')
    #x, y = meshgrid(np.arange(-int(ns*sigmax), int(ns*sigmax)+1), np.arange(-int(ns*sigmay), int(ns*sigmay)+1), indexing='ij')
    cf=np.exp(-(x**2/sigmax**2+y**2/sigmay**2))
    m=~np.isnan(s)*1.
    s = np.where(np.isnan(s),0,s)
    #s[np.isnan(s)]=0.
    # if type(s)=='xarray.core.dataarray.DataArray':
    #     conv2d = lambda x: sp.signal.convolve2d(x, cf, mode="same")
    #     s_sum = xr.apply_ufunc(conv2d, s)
    #     w_sum = xr.apply_ufunc(conv2d, m)
    s_sum = (sp.signal.convolve2d(s, cf, mode='same'))
    w_sum = (sp.signal.convolve2d(m, cf, mode='same'))

    sf = s*0.
    sf[w_sum!=0] = s_sum[w_sum!=0] / w_sum[w_sum!=0]
    return sf

def my2dfilter_over_time(s,sigmax,sigmay, nt, N_CPU, ns=2):
    """
    'my2dfilter' but over each time step and in parallel with Joblib
    """  
    if N_CPU==1:
        list_results = []
        for it in range(nt):
            list_results.append( my2dfilter(s[it,:,:], sigmax, sigmay) )
    else:
        list_results = Parallel(n_jobs=N_CPU)(delayed(my2dfilter)(s[it,:,:], sigmax, sigmay) for it in range(nt))
    
    return np.array(list_results)

def mytimefilter(Hf0):
    """
    This is a time smoothing operator
    
    INPUT:
        - Hf0: array 3D (time,y,x)
    OUTPUT:
        - Hf: smoothed field, same dimensions as Hf0
    """
    
    _, ny, nx = np.shape(Hf0) # shape of array
    time_conv = np.arange(-1*86400,1*86400+3600,3600) # 1D array, time boundaries for kernel (in sec)
    Hf = Hf0*0. # initialization
    
    # time kernel for convolution
    taul = 2*86400
    gl = np.exp(-taul**-2 * time_conv**2) 
    gl = (gl / np.sum(np.abs(gl)))
    # doing the convolution in time
    for ix in range(nx):
        for iy in range(ny):
            Hf[:,iy,ix] = np.convolve(Hf0[:,iy,ix],gl,'same')
    return Hf

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
    name_save += '_' + str(box[0]) + '_' + str(box[1]) + '_' + str(box[2]) + '_' + str(box[3]) + '_' + str(box[4]) + '_' + str(box[5]) + '.nc'
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
    

    
    
    coords=ds.coords
    ds_LS = xr.Dataset(data_vars=data_vars,coords=coords,
                        attrs={'input_files':files,'box':str(box),'file_created_by':'build_LS_files'})            
    print(ds_LS)
    
    
    ds_LS.to_netcdf(path=name_save,mode='w')
    ds_LS.close()
    
def interp_at_model_t_1D(dsUg, dt, ir, jr, list_files, box, path_save, method='linear'):
    """
    Builds a netcdf file with variable interpolated at the timestep dt.
        initial file is global, we create a subset defined by 'box'

    INPUT:
        - dsUg: dataset with geostrophic currents, on the same grid as in 'files'
        - dt : time step to interpolate at
        - list_files: list of path for 1 variable [filesUV,filesH,filesW,filesD]
        - box: list of lat lon boundaries [left,right,bottom,top]
        - path_save: where to save the netcdf file
        - method : interpolation method
    OUPUT:
        - a netcdf file with time interpolated at ['ir','jr'], for 
          currents, SSH, MLD. Stress is added as C.U10**2
    """   
    
    # opening datasets
    ds = xr.open_mfdataset(list_files)
    gUg = dsUg.Ug
    gVg = dsUg.Vg
  
    # searching for lon,lat indexes
    glon = ds.lon
    glat = ds.lat
    ix = np.where((glon>=box[0])&(glon<=box[1]))[0]
    iy = np.where((glat>=box[2])&(glat<=box[3]))[0]
    glon = glon[ix[0]:ix[-1]+1]
    glat = glat[iy[0]:iy[-1]+1]
    
    # selecting data in lat lon
    ds = ds.isel(lat=slice(iy[0],iy[-1]+1),lon=slice(ix[0],ix[-1]+1))
    ds = ds.isel(lon=ir,lat=jr)
    print('     location is:',ds.lon.values,ds.lat.values)
    LON = ds.lon.values
    LAT = ds.lat.values
    
    # checking if file is present
    name_save = path_save
    name_save += 'Interp_1D_LON'+str(LON)+'_LAT'+str(LAT)+'.nc'
    if pathlib.Path(name_save).is_file():
        print(' Interpolated file 1D is already here')
        return None
    

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
                        {'standard_name':'eastward_wind_stress',
                            'long_name':'Eastward wind stress at 10m above water',
                            'units':'m2 s-2',})
    ds_i['TAy'] = (ds_i.geo5_v10m.dims,
                        gTAy.data,
                        {'standard_name':'eastward_wind_stress',
                            'long_name':'Eastward wind stress at 10m above water',
                            'units':'m2 s-2',})
    
    # rename MLD with its proper name
    ds_i['MLD'] = ds_i['KPPhbl']
    ds_i = ds_i.drop_vars(['KPPhbl'])

    # saving
    ds_i.attrs['interp_method'] = method
    ds_i.attrs['produced_by'] = 'interp_at_model_t_1D'
    ds_i.attrs['model_dt'] = dt
    ds_i.attrs['location (LON,LAT)'] = (LON,LAT)
    
    ds_i.to_netcdf(path=name_save,mode='w')
    ds.close()
    ds_i.close()
    