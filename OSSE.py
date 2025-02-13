import numpy as np
import scipy as sp
import xarray as xr
from xgcm import Grid as xGrid
from constants import *
import pathlib
from datetime import timedelta
#import xcroco as xc
import matplotlib.pyplot as plt
import warnings
import dask.array as da
from memory_profiler import profile

from filters import *
from tools import *
#from adapt_croco_to_xgcm import *
 
class Model_source_OSSE:
    """
    This class adapts model outputs to be used with the python scripts
    """
    
    def __init__(self, SOURCE, liste_all_files):
        self.source = SOURCE
        
        if SOURCE=='MITgcm':
            self.dataset = xr.open_mfdataset(liste_all_files)
            self.nameLon_u = 'lon'
            self.nameLon_v = 'lon'
            self.nameLon_rho = 'lon'
            self.nameLat_u = 'lat'
            self.nameLat_v = 'lat'
            self.nameLat_rho = 'lat'
            self.nameSSH = 'SSH'
            self.nameU = 'U'
            self.nameV = 'V'
            self.nameTime = 'time'
            self.nameOceTaux = 'oceTAUX'
            self.nameOceTauy = 'oceTAUY'
            self.dataset = self.dataset.rename({'KPPhbl':'MLD',
                            'SSU':'U',
                            'SSV':'V'})
            self.gridtype = 'centered'
        elif SOURCE=='Croco':
            size_chunk = 200
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                # warning are about chunksize, because not all chunk have the same size.
                self.dataset = xr.open_mfdataset(
                                    liste_all_files, 
                                    chunks={'time_counter': -1,
                                            'x_rho': size_chunk,
                                            'y_rho': size_chunk,
                                            'y_u': size_chunk, 
                                            'x_u': size_chunk,
                                            'y_v': size_chunk, 
                                            'x_v': size_chunk,})
            self.nameLon_u = 'lon_u'
            self.nameLon_v = 'lon_v'
            self.nameLon_rho = 'lon_rho'
            self.nameLat_u = 'lat_u'
            self.nameLat_v = 'lat_v'
            self.nameLat_rho = 'lat_rho'
            self.nameSSH = 'SSH'
            self.nameU = 'U'
            self.nameV = 'V'
            self.nameTime = 'time'
            self.nameOceTaux = 'oceTAUX'
            self.nameOceTauy = 'oceTAUY'
            self.gridtype = 'C'
            # rename redundant dimensions
            _dims = (d for d in ['x_v', 'y_u', 'x_w', 'y_w'] if d in self.dataset.dims)
            for d in _dims:
                self.dataset = self.dataset.rename({d: d[0]+'_rho'})
            
            # removing used variables
            self.dataset = self.dataset.drop_vars(['bvstr','bustr','ubar','vbar','hbbl','h',
                                                   'time_instant','time_instant_bounds','time_counter_bounds'])
            
            
            # renaming
            self.dataset = self.dataset.rename({'zeta':'SSH',
                                                'sustr':'oceTAUX',
                                                'svstr':'oceTAUY',
                                                'shflx':'Heat_flx_net',
                                                'swflx':'frsh_water_net',
                                                'swrad':'SW_rad',
                                                'hbl':'MLD',
                                                'u':'U',
                                                'v':'V',
                                                'time_counter':'time'})
            #   here we remove some variable
            #   because else it cannot write the whole file
            #   but they can be accessed with the original file.
            #   and we dont need them at every dt, and also no need to have the smoothed version.
            #self.dataset = self.dataset.drop_vars(['MLD','temp','salt','frsh_water_net','Heat_flx_net','SW_rad'])
            
            if 'nav_lat_rho' in self.dataset.variables:
                self.dataset = self.dataset.rename({
                                                'nav_lat_rho':'lat_rho',
                                                'nav_lon_rho':'lon_rho',
                                                'nav_lat_u':'lat_u',
                                                'nav_lat_v':'lat_v',
                                                'nav_lon_u':'lon_u',
                                                'nav_lon_v':'lon_v'})
            # building xgcm grid
            coords={'x':{'center':'x_rho',  'right':'x_u'}, 
                    'y':{'center':'y_rho', 'right':'y_v'}}    
            self.grid = xGrid(self.dataset, 
                coords=coords,
                boundary='extend')
            
    def get_name_dim(self):
        return ( self.nameLon_u, self.nameLon_v, self.nameLon_rho,
                self.nameLat_u, self.nameLat_v, self.nameLat_rho,
                self.nameSSH, self.nameU, self.nameV, self.nameTime,
                self.nameOceTaux,self.nameOceTauy)
        
   
class Model_source_3D:
    """
    This class adapts 3D model outputs to be used with the python scripts
    """
    
    def __init__(self, SOURCE, liste_all_files):
        self.source = SOURCE 
        if SOURCE=='Croco':
            size_chunk = 200
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                # warning are about chunksize, because not all chunk have the same size.
                self.dataset = xr.open_mfdataset(
                                    liste_all_files, 
                                    chunks={'time_counter': -1,
                                            'x_rho': size_chunk,
                                            'y_rho': size_chunk,
                                            'y_u': size_chunk, 
                                            'x_u': size_chunk,
                                            'y_v': size_chunk, 
                                            'x_v': size_chunk,})
            print(self.dataset)
            # WIP to plot Croco profiles ...
            raise Exception
        
            self.nameLon_u = 'lon_u'
            self.nameLon_v = 'lon_v'
            self.nameLon_rho = 'lon_rho'
            self.nameLat_u = 'lat_u'
            self.nameLat_v = 'lat_v'
            self.nameLat_rho = 'lat_rho'
            self.nameSSH = 'SSH'
            self.nameU = 'U'
            self.nameV = 'V'
            self.nameTime = 'time'
            self.nameOceTaux = 'oceTAUX'
            self.nameOceTauy = 'oceTAUY'
            self.gridtype = 'C'
            # rename redundant dimensions
            _dims = (d for d in ['x_v', 'y_u', 'x_w', 'y_w'] if d in self.dataset.dims)
            for d in _dims:
                self.dataset = self.dataset.rename({d: d[0]+'_rho'})
            
            # removing used variables
            self.dataset = self.dataset.drop_vars(['bvstr','bustr','ubar','vbar','hbbl','h',
                                                   'time_instant','time_instant_bounds','time_counter_bounds'])
            
            
            # renaming
            self.dataset = self.dataset.rename({'zeta':'SSH',
                                                'sustr':'oceTAUX',
                                                'svstr':'oceTAUY',
                                                'shflx':'Heat_flx_net',
                                                'swflx':'frsh_water_net',
                                                'swrad':'SW_rad',
                                                'hbl':'MLD',
                                                'u':'U',
                                                'v':'V',
                                                'time_counter':'time'})
            #   here we remove some variable
            #   because else it cannot write the whole file
            #   but they can be accessed with the original file.
            #   and we dont need them at every dt, and also no need to have the smoothed version.
            #self.dataset = self.dataset.drop_vars(['MLD','temp','salt','frsh_water_net','Heat_flx_net','SW_rad'])
            
            if 'nav_lat_rho' in self.dataset.variables:
                self.dataset = self.dataset.rename({
                                                'nav_lat_rho':'lat_rho',
                                                'nav_lon_rho':'lon_rho',
                                                'nav_lat_u':'lat_u',
                                                'nav_lat_v':'lat_v',
                                                'nav_lon_u':'lon_u',
                                                'nav_lon_v':'lon_v'})
            # building xgcm grid
            coords={'x':{'center':'x_rho',  'right':'x_u'}, 
                    'y':{'center':'y_rho', 'right':'y_v'}}    
            self.grid = xGrid(self.dataset, 
                coords=coords,
                boundary='extend')
            
   
   
def interp_at_model_t_1D(model_source, dt, point_loc, N_CPU, path_save, method='linear'):
    """
    Source model is : MITgcm or Crocco
    
    Builds a netcdf file with variable interpolated at the timestep dt.
        initial file is global, we create a subset defined by 'box'

    INPUT:
        - model_source: object of class Model_source_OSSE (dataset and name of coords)
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
    name_save += model_source.source + '_Interp_1D_LON'+str(point_loc[0])+'_LAT'+str(point_loc[1])+'.nc'
        
    if pathlib.Path(name_save).is_file():
        print('     interpolated file 1D is already here')
        return None
    
    
    
    # name of variable are different with SOURCE
    (nameLon_u, nameLon_v, nameLon_rho,
     nameLat_u, nameLat_v, nameLat_rho, 
     nameSSH, nameU, nameV, nameTime,
     nameOceTaux,nameOceTauy)= model_source.get_name_dim()
    
    # opening datasets
    ds = model_source.dataset
    nt = len(ds[nameTime])
    
    # getting a reduced size area to compute large scale current
    # Area is +- 1° around 'point_loc'if MITgcm
    # else if Croco, for 40 its about +- 0.5° around 'point_loc'
    if model_source.source=='MITgcm':
        # every array on rho grid
        ds = ds.sel({nameLon_rho:slice(point_loc[0]-1,point_loc[0]+1),
                    nameLat_rho:slice(point_loc[1]-1,point_loc[1]+1)})
        indx = nearest(ds[nameLon_rho].values,point_loc[0])
        indy = nearest(ds[nameLat_rho].values,point_loc[1])
        LON = ds[nameLon_rho][indx].values # scalar
        LAT = ds[nameLat_rho][indy].values # scalar   
    elif model_source.source == 'Croco':     
        indx,indy = find_indices(point_loc,ds.lon_rho.values,ds.lat_rho.values)[0]
        LON = ds[nameLon_rho].values[indy,indx] # scalar
        LAT = ds[nameLat_rho].values[indy,indx] # scalar
        latmin = np.amin( xr.where(ds.lat_rho==0.,np.nan,ds.lat_rho) )
        latmax = np.amax( xr.where(ds.lat_rho==0.,np.nan,ds.lat_rho) )
        lonmin = np.amin( xr.where(ds.lon_rho==0.,np.nan,ds.lon_rho) )
        lonmax = np.amax( xr.where(ds.lon_rho==0.,np.nan,ds.lon_rho) )

        dindx = 40
        dindy = dindx
        
        
        # smaller domain for lighter computations
        ds = ds.isel(   x_rho=slice( indx-dindx,indx+dindx),
                        y_rho=slice( indy-dindy,indy+dindy),
                        x_u=slice( indx-dindx,indx+dindx),
                        #y_u=slice( indy-dindy,indy+dindy),
                        #x_v=slice( indx-dindx,indx+dindx),
                        y_v=slice( indy-dindy,indy+dindy) )
        
        # building xgcm grid
        coords={'x':{'center':'x_rho',  'right':'x_u'}, 
                'y':{'center':'y_rho', 'right':'y_v'}}    
        grid = xGrid(ds, 
              coords=coords,
              boundary='extend')
    

    # large scale filtering of SSH
    sigmax, sigmay = 3, 3
    ds[nameSSH] = xr.where(ds[nameSSH]>1e5,np.nan,ds[nameSSH])
    ds['SSH_LS0'] = xr.zeros_like(ds[nameSSH])
    ds['SSH_LS'] = xr.zeros_like(ds[nameSSH])
    
    print('     2D filter')
    ds['SSH_LS0'].data = my2dfilter_over_time(ds[nameSSH].values,sigmax,sigmay, nt, N_CPU, ns=2)
    
    print('     time filter')
    ds['SSH_LS'].data = mytimefilter(ds['SSH_LS0'])  
    
    # getting geo current from SSH
    # finite difference, SSH goes from rho -> rho
    print('     gradXY ssh')
    if model_source.source=='MITgcm':
        # lat, lon are 1D
        glon2,glat2 = np.meshgrid(ds[nameLon_rho].values,ds[nameLat_rho].values)
        dlon = (ds[nameLon_rho][1]-ds[nameLon_rho][0]).values
        dlat = (ds[nameLat_rho][1]-ds[nameLat_rho][0]).values
    elif model_source.source=='Croco':
        # lat, lon are gridded
        glat2 = ds[nameLat_rho].values
        glon2 = ds[nameLon_rho].values
        # local dlon,dlat
        dlon = (glon2[:,1:]-glon2[:,:-1]).mean()
        dlat = (glat2[1:,:]-glat2[:-1,:]).mean()
      
    fc = 2*2*np.pi/86164*np.sin(glat2*np.pi/180)
    gUg = ds['SSH_LS']*0.
    gVg = ds['SSH_LS']*0.
    # gradient metrics
    dx = dlon * xr.ufuncs.cos(xr.ufuncs.deg2rad(glat2)) * distance_1deg_equator 
    dy = ((glon2 * 0) + 1) * dlat * distance_1deg_equator
    # over time array
    for it in range(nt):
        # this could be vectorized ...
        # centered gradient, dSSH/dx is on SSH point
        gVg[it,:,1:-1] =  grav/fc[:,1:-1] *( ds['SSH_LS'][it,:,2:].values - ds['SSH_LS'][it,:,:-2].values ) / ( dx[:,1:-1] )/2
        gUg[it,1:-1,:] = -grav/fc[1:-1,:]*( ds['SSH_LS'][it,2:,:].values  - ds['SSH_LS'][it,:-2,:].values ) / ( dy[1:-1,] )/2
    gUg[:,0,:] = gUg[:,1,:]
    gUg[:,-1,:]= gUg[:,-2,:]
    gVg[:,:,0] = gVg[:,:,1]
    gVg[:,:,-1]= gVg[:,:,-2]
    # adding geo current to the file
    ds['Ug'] = (ds[nameSSH].dims,
                        gUg.data,
                        {'standard_name':'zonal_geostrophic_current',
                            'long_name':'zonal geostrophic current from SSH',
                            'units':'m s-1',})
    ds['Vg'] = (ds[nameSSH].dims,
                        gVg.data,
                        {'standard_name':'meridional_geostrophic_current',
                            'long_name':'meridional geostrophic current from SSH',
                            'units':'m s-1',}) 
    
    # selecting the location in 1D (time)
    if model_source.source == 'MITgcm':
        # this line reduce X and Y dimension to 1 point
        ds = ds.sel({nameLon_rho:point_loc[0],
                    nameLat_rho:point_loc[1]})
    elif model_source.source == 'Croco':
        # For Croco, we need to do manually the change of cell position
        # 1) interp at mass point
        # 2) select the point corresponding to 'point_loc' (LON,LAT)
        # variables at other points:
        for var in [nameU,nameOceTaux]:
            attrs = ds[var].attrs
            ds[var] = grid.interp(ds[var], 'x')
            ds[var].attrs = attrs
        for var in [nameV,nameOceTauy]:
            attrs = ds[var].attrs
            ds[var] = grid.interp(ds[var], 'y')
            ds[var].attrs = attrs
        # remove non-rho coordinates
        ds = ds.drop_vars({'lat_u','lat_v','lon_u','lon_v'})
        # variables already on rho points:
        ds = ds.isel(x_rho=dindx,y_rho=dindy)   
        ds = ds.rename({'lon_rho':'lon','lat_rho':'lat'})
    
    # removing geostrophy
    ds[nameU].data = ds[nameU].values - ds['Ug'].values
    ds[nameV].data = ds[nameV].values - ds['Vg'].values
    ds[nameU].attrs['long_name'] = 'Ageo '+ds[nameU].attrs['long_name']
    ds[nameV].attrs['long_name'] = 'Ageo '+ds[nameV].attrs['long_name']
    
    # new time vector
    print('     interp on model time vector')
    time = np.arange(ds[nameTime].values[0], ds[nameTime].values[-1], timedelta(seconds=dt),dtype='datetime64[ns]')
    ds_i = ds.interp({'time':time},method=method) # linear interpolation in time
        
    # adding the stress to the variables
    # stress as: C x wind**2
    if model_source.source == 'MITgcm':
        gTx = ds_i.geo5_u10m
        gTy = ds_i.geo5_v10m
        gTAx = 1e-3*np.sign(gTx)*gTx**2 # Tau/rho = Cd*U**2 :  CD~1e-3
        gTAy = 1e-3*np.sign(gTy)*gTy**2
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

    # saving
    print('     saving ...')
    ds_i.attrs['interp_method'] = method
    ds_i.attrs['produced_by'] = 'interp_at_model_t_1D'
    ds_i.attrs['model_dt'] = dt
    ds_i.attrs['location (LON,LAT)'] = (LON,LAT)
    ds_i.attrs['model_source'] = model_source.source
    
    ds_i.to_netcdf(path=name_save,mode='w')
    ds.close()
    ds_i.close()
    print('     done !')
    
def interp_at_model_t_2D(model_source, dt, LON_bounds, LAT_bounds, N_CPU, path_save, method='linear'):
    """
    Source model is : Crocco
    
    Builds a netcdf file with variable interpolated at the timestep dt.
        initial file is North Atlanctic region, we create a subset defined by LON_bounds, LAT_bounds

    INPUT:
        - model_source: object of class Model_source_OSSE (dataset and name of coords)
        - dt : time step to interpolate at
        - LON_bounds : longitude bounds of the spatial domain
        - LAT_bounds : latitutde bounds of the spatial domain
        - N_CPU: do the spatial filter in // if > 1
        - path_save: where to save the netcdf file
        - method : interpolation method
    OUPUT:
        - a netcdf file with time interpolated on a 2D grid comprised in between LON_bounds and LAT_bounds, 
            for ageo currents, geo current, SSH, MLD, stress, fluxes
            
    Note : 
        - tested for model_source.source=='Croco' (C grid)
        - output is not interpolate on 'dt' timestep, 
            the interpolation is done on the fly during model integration
            (memory constrain)
    """
    final_time_chunk = 500
    
    # checking if file is present
    name_save = path_save
    name_save += model_source.source + '_Interp_2D_LON'+str(LON_bounds[0])+'_'+str(LON_bounds[1])+'_LAT'+str(LAT_bounds[0])+'_'+str(LAT_bounds[1])
    if pathlib.Path(name_save+'.nc').is_file() or pathlib.Path(name_save+'.zarr').is_dir():
        print('     interpolated file 2D is already here')
        return None # exit 
    
    # name of variable are different with SOURCE
    (nameLon_u, nameLon_v, nameLon_rho,
     nameLat_u, nameLat_v, nameLat_rho, 
     nameSSH, nameU, nameV, nameTime,
     nameOceTaux,nameOceTauy)= model_source.get_name_dim()
    
    # opening datasets
    ds = model_source.dataset
    
    nt = len(ds[nameTime])
    
    if len(ds[nameLon_rho].shape)==1:
        # gridded on lat,lon: lat lon are 1D
        indxmin = nearest(ds[nameLon_rho].values,LON_bounds[0])
        indxmax = nearest(ds[nameLon_rho].values,LON_bounds[1])
        indymin = nearest(ds[nameLat_rho].values,LAT_bounds[0])
        indymax = nearest(ds[nameLat_rho].values,LAT_bounds[1])
        C1 = float(ds[nameLon_rho].values[indxmin]),float(ds[nameLat_rho].values[indymin])
        C2 = float(ds[nameLon_rho].values[indxmax]),float(ds[nameLat_rho].values[indymin])
        C3 = float(ds[nameLon_rho].values[indxmin]),float(ds[nameLat_rho].values[indymax])
        C4 = float(ds[nameLon_rho].values[indxmax]),float(ds[nameLat_rho].values[indymax])
        
        # every array on rho grid
        ds = ds.sel({nameLon_rho:slice(LON_bounds[0],LON_bounds[0]),
                    nameLat_rho:slice(LAT_bounds[0],LAT_bounds[1])})
        LON = ds[nameLon_rho][indxmin:indxmax].values 
        LAT = ds[nameLat_rho][indymin:indymax].values 
        
    elif len(ds[nameLon_rho].shape)==2:
        # gridded on x,y: lat lon are 2D
        indices = find_indices_gridded_latlon(ds[nameLon_rho].values, LON_bounds, ds[nameLat_rho].values, LAT_bounds)
        indxmin, indxmax, indymin, indymax = indices
        C1 = float(ds[nameLon_rho].values[indymin,indxmin]),float(ds[nameLat_rho].values[indymin,indxmin])
        C2 = float(ds[nameLon_rho].values[indymin,indxmax]),float(ds[nameLat_rho].values[indymin,indxmax])
        C3 = float(ds[nameLon_rho].values[indymax,indxmin]),float(ds[nameLat_rho].values[indymax,indxmin])
        C4 = float(ds[nameLon_rho].values[indymax,indxmax]),float(ds[nameLat_rho].values[indymax,indxmax])
        
        latmin = np.amin( xr.where(ds.lat_rho==0.,np.nan,ds.lat_rho) )
        latmax = np.amax( xr.where(ds.lat_rho==0.,np.nan,ds.lat_rho) )
        lonmin = np.amin( xr.where(ds.lon_rho==0.,np.nan,ds.lon_rho) )
        lonmax = np.amax( xr.where(ds.lon_rho==0.,np.nan,ds.lon_rho) )     
        
        # smaller domain
        ds = ds.isel(x_rho=slice(indxmin,indxmax),
                    y_rho=slice(indymin,indymax),
                    x_u=slice(indxmin,indxmax),
                    y_v=slice(indymin,indymax) )
        
        # building xgcm grid
        coords={'x':{'center':'x_rho',  'right':'x_u'}, 
                'y':{'center':'y_rho', 'right':'y_v'}}    
        grid = xGrid(ds, 
              coords=coords,
              boundary='extend')
    
    
    
    print('LOOKING AT:')
    print(str(LON_bounds[0])+','+str(LAT_bounds[1])+'------'+str(LON_bounds[1])+','+str(LAT_bounds[1]))
    print('|                        |')
    print('|                        |')
    print('|                        |')
    print(str(LON_bounds[0])+','+str(LAT_bounds[0])+'------'+str(LON_bounds[1])+','+str(LAT_bounds[0]))
    print('FOUND:')
    print(str(C3)+'------'+str(C4))
    print('|                                        |')
    print('|                                        |')
    print('|                                        |')
    print('|                                        |')
    print(str(C1)+'------'+str(C2))
    
    # large scale filtering of SSH
    sigmax, sigmay = 3, 3
    ds[nameSSH] = xr.where(ds[nameSSH]>1e5,np.nan,ds[nameSSH])
    ds['SSH_LS0'] = xr.zeros_like(ds[nameSSH])
    ds['SSH_LS'] = xr.zeros_like(ds[nameSSH])
    print('     -> 2D spatial filter')
    ds['SSH_LS0'].data = my2dfilter_over_time(ds[nameSSH].values,sigmax,sigmay, nt, N_CPU, ns=2)
    
    print('     -> time filter')
    ds['SSH_LS'].data = mytimefilter_over_spatialXY(ds['SSH_LS0'], N_CPU)
    ds = ds.drop_vars('SSH_LS0') # remove this as we dont need it anymore
    
    # getting geo current from SSH
    # finite difference, SSH goes from rho -> rho
    print('     -> gradXY ssh')
    if len(ds[nameLon_rho].shape)==1:
        # lat, lon are 1D
        glon2,glat2 = np.meshgrid(ds[nameLon_rho].values,ds[nameLat_rho].values)
        dlon = (ds[nameLon_rho][1]-ds[nameLon_rho][0]).values
        dlat = (ds[nameLat_rho][1]-ds[nameLat_rho][0]).values
    elif len(ds[nameLon_rho].shape)==2:
        # lat, lon are gridded
        glat2 = ds[nameLat_rho].values
        glon2 = ds[nameLon_rho].values
        # local dlon,dlat
        dlon = (glon2[:,1:]-glon2[:,:-1]).mean()
        dlat = (glat2[1:,:]-glat2[:-1,:]).mean()
      
    fc = 2*2*np.pi/86164*np.sin(glat2*np.pi/180)
    gUg = ds['SSH_LS']*0.
    gVg = ds['SSH_LS']*0.
    # gradient metrics
    dx = dlon * xr.ufuncs.cos(xr.ufuncs.deg2rad(glat2)) * distance_1deg_equator 
    dy = ((glon2 * 0) + 1) * dlat * distance_1deg_equator
    # over time array
    for it in range(nt):
        # this could be vectorized ...
        # centered gradient, dSSH/dx is on SSH point
        gVg[it,:,1:-1] =  grav/fc[:,1:-1] *( ds['SSH_LS'][it,:,2:].values - ds['SSH_LS'][it,:,:-2].values ) / ( dx[:,1:-1] )/2
        gUg[it,1:-1,:] = -grav/fc[1:-1,:]*( ds['SSH_LS'][it,2:,:].values  - ds['SSH_LS'][it,:-2,:].values ) / ( dy[1:-1,] )/2
    gUg[:,0,:] = gUg[:,1,:]
    gUg[:,-1,:]= gUg[:,-2,:]
    gVg[:,:,0] = gVg[:,:,1]
    gVg[:,:,-1]= gVg[:,:,-2]
    
    
    # adding geo current to the file
    ds['Ug'] = (ds[nameSSH].dims,
                        gUg.data,
                        {'standard_name':'zonal_geostrophic_current',
                            'long_name':'zonal geostrophic current from SSH',
                            'units':'m s-1',})
    ds['Vg'] = (ds[nameSSH].dims,
                        gVg.data,
                        {'standard_name':'meridional_geostrophic_current',
                            'long_name':'meridional geostrophic current from SSH',
                            'units':'m s-1',}) 

    if model_source.gridtype=='C':
        print('     -> C grid: interp on rho points')
        # For Croco, we need to do manually the change of cell position
        # 1) interp at mass point
        # 2) select the point corresponding to 'point_loc' (LON,LAT)
        # variables at other points:
        for var in [nameU,nameOceTaux]:
            attrs = ds[var].attrs
            ds[var] = grid.interp(ds[var], 'x')
            ds[var].attrs = attrs
        for var in [nameV,nameOceTauy]:
            attrs = ds[var].attrs
            ds[var] = grid.interp(ds[var], 'y')
            ds[var].attrs = attrs
        # remove non-rho coordinates
        ds = ds.drop_vars({nameLat_u,nameLat_v,nameLon_u,nameLon_v})
        # all onvariables are now on rho points:  
        ds = ds.rename({nameLon_rho:'lon',nameLat_rho:'lat'})
    
    #ds = ds.load() # this trigger computation
    #print(ds)
    
    # at this point, everything should be on rho point.
    #   and also the dataset should be small.
    #   so we set the chunksize to get 1 chunk for every dims.
    # This chunking is done to be able to write in //
    # chunks = ds[nameSSH].chunks
    for varname in ds.variables:
        ds[varname] = ds[varname].chunk(chunks=100)
        
    # removing geostrophy
    ds[nameU].data = ds[nameU] - ds['Ug']
    ds[nameV].data = ds[nameV] - ds['Vg']
    ds[nameU].attrs['long_name'] = 'Ageo '+ds[nameU].attrs['long_name']
    ds[nameV].attrs['long_name'] = 'Ageo '+ds[nameV].attrs['long_name']
    
    print('     -> interp on model time vector')
    ds_i = ds
    #print(ds_i)
    # this is done during the model integration !
    # because what would be done (see below) consumes too much memory (scipy.interpn behind xr.interp)
    
    #newtime = np.arange(ds[nameTime].values[0], ds[nameTime].values[-1], timedelta(seconds=dt),dtype='datetime64[ns]')
    # ds_i = ds.interp({nameTime:newtime},method=method) 
    
    # a xarray.dataarray.interp on chunk dim is not yet supported and it seems hard to implement.
    # ongoing since 2020
    # see https://github.com/pydata/xarray/pull/4155
    # and https://github.com/pydata/xarray/issues/6799
    
    #print_memory_array(ds_i,'U')
    #print_memory_chunk(ds_i,'U')

    # adding the stress to the variables
    # stress as: C x wind**2
    if model_source.source == 'MITgcm':
        gTx = ds_i.geo5_u10m
        gTy = ds_i.geo5_v10m
        gTAx = 1e-3*np.sign(gTx)*gTx**2 # Tau/rho = Cd*U**2 :  CD~1e-3
        gTAy = 1e-3*np.sign(gTy)*gTy**2
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


    # saving
    print('     saving ...')
    ds_i.attrs['interp_method'] = method
    ds_i.attrs['produced_by'] = 'interp_at_model_t_2D'
    ds_i.attrs['model_dt'] = dt
    ds_i.attrs['input LAT_bounds'] = LAT_bounds
    ds_i.attrs['input LON_bounds'] = LON_bounds
    ds_i.attrs['true corner C1 (LON,LAT)'] = C1
    ds_i.attrs['true corner C2 (LON,LAT)'] = C2
    ds_i.attrs['true corner C3 (LON,LAT)'] = C3
    ds_i.attrs['true corner C4 (LON,LAT)'] = C4
    ds_i.attrs['model_source'] = model_source.source
    
    # writing in time chunk to fit in memory
    # if the file is huge
    #   -> this create the structure
    # ds_template = ds_i.drop_vars([var for var in ds_i.variables if var not in ['lat','lon']])
    # ds_template.to_zarr(name_save+'.zarr', compute=False) 
    #   -> this is to write variables independant of time
    # ds_template = ds_template
    # print(ds_template)
    # ds_template.to_zarr(name_save+'.zarr', mode='a')
    #   -> writing the other variables in time chunks
    # ds_i = ds_i.drop_vars(['lat','lon'])
    # Nchunk_time = len(ds_i[nameTime]) // final_time_chunk+1
    # for j in range(0,len(ds_i[nameTime]),final_time_chunk):
    #     print('     time chunk '+str(j)+'/'+str(Nchunk_time))
    #     temp = ds_i.isel(time=slice(j, j+final_time_chunk)).load()
    #     temp.to_zarr(name_save+'.zarr', region={'time': slice(j, j+final_time_chunk)}, mode='a')
    
    # if the file isnt huge
    ds_i.to_netcdf(path=name_save+'.nc',mode='w')
    
    # note on this: xarray/dask should be able to do to_netcdf in one go if dataset is chunked...
    
    ds.close()
    ds_i.close()
    print('     done !')