import numpy as np
import xarray as xr

class Observation1D:
    """
    Observations of currents fields for 'Unstek1D'
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
    
    