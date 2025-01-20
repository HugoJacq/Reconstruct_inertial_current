import numpy as np
import xarray as xr

class Forcing1D:
    """
    Forcing fields for 'Unstek1D'
    """
    def __init__(self, dt, path_file, TRUE_WIND_STRESS):
        
        # from dataset
        self.data = xr.open_dataset(path_file)
        self.U,self.V,self.MLD = self.data.SSU.values,self.data.SSV.values,self.data.MLD
        self.bulkTx,self.bulkTy,self.oceTx,self.oceTy = self.data.TAx.values,self.data.TAy.values,self.data.oceTAUX.values,self.data.oceTAUY.values
        self.fc = 2*2*np.pi/86164*np.sin(self.data.lat.values*np.pi/180) # Coriolis value at jr,ir
        self.nt = len(self.data.time)
        self.time = np.arange(0,self.nt*dt,dt)    # 1 step every dt
        self.dt = dt
        
        # wind stress
        if TRUE_WIND_STRESS:
            self.TAx,self.TAy = self.oceTx,self.oceTy
        else:
            self.TAx,self.TAy = self.bulkTx,self.bulkTy
        
    