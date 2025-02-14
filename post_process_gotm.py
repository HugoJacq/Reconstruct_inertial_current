"""
This scripts plots the data from a GOTM simulation
"""

import xarray as xr
import matplotlib.pyplot as plt
import pathlib
import os
# my imports
from tools import *
from constants import *

# ===================================
# INPUT
workdir = 'gotm_workdir/'

point_loc = [-53.,29.5] # °E, °N
file_name = 'output_file.nc'
forcing_file = '/home/jacqhugo/Datlas_2025/DATA_Crocco/croco_1h_inst_surf_2006-02-01-2006-02-28.nc'

# PARAMETERS
dt_gotm = 60 # s
dt_croco = 3600 # s

PLOT_CURRENT = True            # plots the gotm current vs true current

dir_save = "./png_gotm/"
dpi=200
# ===================================

exp = 'Croco_LON' + str(point_loc[0])+'_'+'LAT' + str(point_loc[1])+'/'

path_save = dir_save + exp
if not pathlib.Path(path_save).is_dir():
    os.system('mkdir '+path_save)

dsout = xr.open_dataset(workdir+exp+file_name).isel(lon=0,lat=0)
# this can be long because of Cgrid that requires interpolation 
#  |
#  v
dstruth = open_croco_sfx_file_at_point_loc(forcing_file, point_loc) 

time_gotm = np.arange(0,len(dsout.time))*dt_gotm
time_croco = np.arange(0,len(dstruth.time))*dt_croco

Ugotm = dsout.u
Ucroco = dstruth.U
Taux = dstruth.oceTAUX
Tauy = dstruth.oceTAUY
#print('Ugotm',Ugotm)
#print('Ucroco',Ucroco)

# looking at surface current gotm vs Croco
if PLOT_CURRENT:
    fig, ax = plt.subplots(2,1,figsize = (10,7),constrained_layout=True,dpi=dpi)
    ax[0].plot( time_gotm/oneday, Ugotm[:,-1], c='b',label='gotm')
    ax[0].plot( time_croco/oneday, Ucroco, c='k',label='croco')
    ax[0].set_ylabel('surface current (m/s)')
    ax[0].legend()
    
    ax[1].plot(time_croco/oneday, Taux, c='limegreen',label=r'$\tau_x$')
    ax[1].plot(time_croco/oneday, Tauy, c='darkorange',label=r'$\tau_y$')
    ax[1].set_ylabel('wind stress (m2/s2)')
    ax[1].set_xlabel('time (days)')
    ax[1].legend()
    
    fig.savefig(path_save+'gotm_vs_Croco_Usfx.png')
    



dsout.close()
dstruth.close()
plt.show()