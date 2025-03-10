"""
OSSE maker from Croco OA-coupled simulation
"""
import xarray as xr
import numpy as np

import tools

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
        indx = tools.nearest(ds.lon.values,point_loc[0])
        indy = tools.nearest(ds.lat.values,point_loc[1])
        self.data = ds.isel(lon=indx,lat=indy)
        
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