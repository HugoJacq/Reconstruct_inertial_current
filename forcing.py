import numpy as np
import xarray as xr

from tools import *

class Forcing1D:
    """
    Forcing fields for 'Unstek1D'
    """
    def __init__(self, dt, path_file, TRUE_WIND_STRESS):
        
        # from dataset
        self.data = xr.open_dataset(path_file)
        self.U,self.V,self.MLD = self.data.U.values,self.data.V.values,self.data.MLD
        if 'TAx' in self.data.keys():
            self.bulkTx,self.bulkTy = self.data.TAx.values,self.data.TAy.values
        self.oceTx,self.oceTy = self.data.oceTAUX.values,self.data.oceTAUY.values
        self.fc = 2*2*np.pi/86164*np.sin(self.data.lat.values*np.pi/180) # Coriolis value at jr,ir
        self.nt = len(self.data.time)
        self.time = np.arange(0,self.nt*dt,dt)    # 1 step every dt
        self.dt = dt
        
        # wind stress
        if TRUE_WIND_STRESS:
            self.TAx,self.TAy = self.oceTx,self.oceTy
        else:
            if 'TAx' in self.data.keys():
                self.TAx,self.TAy = self.bulkTx,self.bulkTy
            else:
                print('     no wind in the file, I will use true windstress and not cd*U**2')
                print('         TRUE_WINDSTRESS turned to TRUE')
                self.TAx,self.TAy = self.oceTx,self.oceTy
                         
class Forcing2D:
    """
    Forcing fields for :
        - 'jUnstek1D_spatial'
        -
    """
    def __init__(self, dt, LON_bounds, LAT_bounds, path_file, TRUE_WIND_STRESS):
        
        # from dataset
        self.data = xr.open_dataset(path_file)
        
        
        # A METTRE DANS UNE FONCTION QUI SERA
        # REUTILISEE DANS 'OSSE:interp_at_model_t_2D'
        # -> voir utiliser le fichier interped pour récupérer le domaine spatial
        
        # reducing lat/lon to fit bounds
        L_corners = []
        x_rho_indmin,x_rho_indmax = len(self.data.x_rho),0
        y_rho_indmin,y_rho_indmax = len(self.data.x_rho),0
        lat_rho = self.data.lat_rho.values
        lon_rho = self.data.lon_rho.values
        for LAT in LAT_bounds:
            for LON in LON_bounds:
                L_corners.append([LON,LAT])
                indx,indy = find_indices(L_corners[-1],lon_rho,lat_rho)[0]
                if indx < x_rho_indmin: indx = x_rho_indmin
                if indx > x_rho_indmax: indx = x_rho_indmax
                if indy < y_rho_indmin: indy = y_rho_indmin
                if indy > y_rho_indmax: indy = y_rho_indmax
        
        print('LOOKING FOR:')
        print(LON_bounds[0]+','+LAT_bounds[1]+'------'+LON_bounds[1]+','+LAT_bounds[1])
        print('|            |')
        print('|            |')
        print(LON_bounds[0]+','+LAT_bounds[0]+'------'+LON_bounds[1]+','+LAT_bounds[0])
        print('FOUND:')
        print(lat_rho[x_rho_indmin,y_rho_indmax].values+'------'+lat_rho[x_rho_indmax,y_rho_indmax].values)
        print('|            |')
        print('|            |')
        print(lat_rho[x_rho_indmin,y_rho_indmin].values+'------'+lat_rho[x_rho_indmax,y_rho_indmin].values)
        
        raise Exception
        self.U,self.V,self.MLD = self.data.U.values,self.data.V.values,self.data.MLD
        if 'TAx' in self.data.keys():
            self.bulkTx,self.bulkTy = self.data.TAx.values,self.data.TAy.values
        self.oceTx,self.oceTy = self.data.oceTAUX.values,self.data.oceTAUY.values
        self.fc = 2*2*np.pi/86164*np.sin(self.data.lat.values*np.pi/180) # Coriolis value at jr,ir
        self.nt = len(self.data.time)
        self.time = np.arange(0,self.nt*dt,dt)    # 1 step every dt
        self.dt = dt
        
        # wind stress
        if TRUE_WIND_STRESS:
            self.TAx,self.TAy = self.oceTx,self.oceTy
        else:
            if 'TAx' in self.data.keys():
                self.TAx,self.TAy = self.bulkTx,self.bulkTy
            else:
                print('     no wind in the file, I will use true windstress and not cd*U**2')
                print('         TRUE_WINDSTRESS turned to TRUE')
                self.TAx,self.TAy = self.oceTx,self.oceTy
    