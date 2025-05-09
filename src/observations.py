import numpy as np
import xarray as xr

from .tools import *

class Observation1D:
    """
    Observations of currents fields for 'Unstek1D'
    
    point_loc   : [LON,LAT] location (°E,°N)
    periode_obs : interval of time between obs (s)
    dt_model    : timestep of input OSSE model (s)
    path_file   : path to regridded file with obs/forcing variables 
    """
    def __init__(self, point_loc, periode_obs, dt_model, path_file):
        
        # from dataset
        ds = xr.open_dataset(path_file)
        indx = nearest(ds.lon.values,point_loc[0])
        indy = nearest(ds.lat.values,point_loc[1])
        self.data = ds.isel(lon=indx,lat=indy)
        
        self.U,self.V = self.data.U.values,self.data.V.values
        self.dt = dt_model
        self.obs_period = periode_obs
        self.time_obs = np.arange(0., len(self.data.time)*dt_model,periode_obs)

    def get_obs(self):
        """
        OSSE of current from the coupled OA model
        """
        # HERE
        # it would be nice to have the same shape for any model output and the observations.
        # use a fill value ?
        # or see https://stackoverflow.com/questions/71692885/handle-varying-shapes-in-jax-numpy-arrays-jit-compatible
        self.Uo = self.U[::self.obs_period//self.dt]
        self.Vo = self.V[::self.obs_period//self.dt]
        return self.Uo,self.Vo
    
class Observation1D_SAVE:
    """
    Observations of currents fields for 'Unstek1D'
    
    path_file points to a hand made 1D (temporal) serie of current, wind stress extracted from a coupled model
    """
    def __init__(self, periode_obs, dt_model, path_file):
        
        # from dataset for OSSE
        self.data = xr.open_dataset(path_file)
        self.U,self.V = self.data.U.values,self.data.V.values
        self.dt = dt_model
        self.obs_period = periode_obs
        self.time_obs = np.arange(0, len(self.data.time)*dt_model,periode_obs)

    def get_obs(self):
        """
        OSSE of current from the coupled OA model
        """
        self.Uo = self.U[::self.obs_period//self.dt]
        self.Vo = self.V[::self.obs_period//self.dt]
        return self.Uo,self.Vo

class Observation2D:
    """
    Observations of currents fields for 'Unstek1D', 2D (spatial)
    
    - period_obs : float(s) time interval between observations 
    - dt_model : obs comes from OSSE, this is the time step of the OSSE.
    - path_file : file with OSSE data (regridded)
    - LON_bounds : LON min and LON max of zone
    - LAT_bounds : LAT min and LAT max of zone
    """
    def __init__(self, periode_obs, dt_model, path_file, LON_bounds, LAT_bounds):
        
        # from dataset for OSSE
        ds = xr.open_dataset(path_file)
        indxmin = nearest(ds.lon.values,LON_bounds[0])
        indxmax = nearest(ds.lon.values,LON_bounds[1])
        indymin = nearest(ds.lat.values,LAT_bounds[0])
        indymax = nearest(ds.lat.values,LAT_bounds[1])
        
        self.data = ds.isel(lon=slice(indxmin,indxmax),lat=slice(indymin,indymax))
        self.U,self.V = self.data.U.values,self.data.V.values
        self.dt = dt_model
        self.obs_period = periode_obs
        
        self.time_obs = np.arange(0, len(self.data.time)*dt_model,periode_obs)

    def get_obs(self):
        """
        OSSE of current from the coupled OA model
        """
        self.Uo = self.U[::self.obs_period//self.dt]
        self.Vo = self.V[::self.obs_period//self.dt]
        return self.Uo,self.Vo
    