"""
This scripts plots the data from a GOTM simulation
"""

import xarray as xr
import matplotlib.pyplot as plt
import pathlib
import os
from dask.distributed import Client,LocalCluster
# my imports
from tools import *
from constants import *

# ===================================
# INPUT
DASHBOARD = False    # dask dashboard
workdir = 'gotm_workdir/'

point_loc = [-53.,29.5] # °E, °N
file_name = 'output_file.nc'
forcing_file = '/home/jacqhugo/Datlas_2025/DATA_Crocco/croco_1h_inst_surf_2006-02-01-2006-02-28.nc'

# PARAMETERS
dt_gotm = 60 # s
dt_croco = 3600 # s

PLOT_CURRENT = True            # plots the gotm current vs true current
PLOT_MLD     = False            #


dir_save = "./png_gotm/"
dpi=200
# ===================================
# This avoids infinite subprocess creation
if __name__ == "__main__":  
    print('* GOTM post process')
    client = None
    if DASHBOARD:
        # sometimes dask cluster can cause problems "memoryview is too large"
        # (writing a big netcdf file for eg, hbudget_file)
        cluster = LocalCluster(n_workers=8) # threads_per_worker=1,
        client = Client(cluster)
        print("Dashboard at :",client.dashboard_link)

    exp = 'Croco_LON' + str(point_loc[0])+'_'+'LAT' + str(point_loc[1])+'/'

    path_save = dir_save + exp
    if not pathlib.Path(path_save).is_dir():
        os.system('mkdir '+path_save)

    print('     opening files')
    dsout = xr.open_dataset(workdir+exp+file_name).isel(lon=0,lat=0)
    # this can be long because of Cgrid that requires interpolation 
    # and also files are big
    #  |
    #  v
    dstruth = open_croco_sfx_file_at_point_loc(forcing_file, point_loc, interp_var='all') 

    time_gotm = np.arange(0,len(dsout.time))*dt_gotm
    time_croco = np.arange(0,len(dstruth.time))*dt_croco

    Ugotm = dsout.u
    Ucroco = dstruth.U
    Taux = dstruth.oceTAUX
    Tauy = dstruth.oceTAUY
    Heat = dstruth.Heat_flx_net
    MLDc = dstruth.MLD

    #print('Ugotm',Ugotm)
    #print('Ucroco',Ucroco)


    RMSE = score_RMSE(Ugotm.values[::dt_croco//dt_gotm,0],Ucroco.values)

    # looking at surface current gotm vs Croco
    if PLOT_CURRENT:
        print('* PLotting currents')
        fig, ax = plt.subplots(2,1,figsize = (10,7),constrained_layout=True,dpi=dpi)
        ax[0].plot( time_gotm/oneday, Ugotm[:,-1], c='b',label='gotm')
        ax[0].plot( time_croco/oneday, Ucroco, c='k',label='croco')
        ax[0].hlines(0.,0,time_croco[-1]/oneday, colors='gray',linestyles='--')
        ax[0].set_ylim([-0.3,0.3])
        ax[0].set_ylabel('surface current (m/s)')
        ax[0].legend()
        
        ax2 = ax[1].twinx()
        ax2.plot( time_croco/oneday,Heat, c='darkorange',label=r'$H$',alpha=0.5)
        ax[1].plot(time_croco/oneday, Taux, c='limegreen',label=r'$\tau_x$')
        ax[1].plot(time_croco/oneday, Tauy, c='darkturquoise',label=r'$\tau_y$')
        ax[1].plot( time_croco[0]/oneday,Heat[0], c='darkorange',label=r'$H$',alpha=0.5) #dummy
        ax[1].hlines(0.,0,time_croco[-1]/oneday, colors='gray',linestyles='--')
        ax[1].set_ylabel('wind stress (m2/s2)')
        ax[1].set_xlabel('time (days)')
        ax[1].set_ylim([-0.6,0.6])
        ax[1].legend()
        
        ax2.set_ylabel(' net heat flux (W/m2)')
        ax2.set_ylim([-600,600])
        
        fig.suptitle('RMSE='+str(np.round(RMSE,5)))
        
        fig.savefig(path_save+'gotm_vs_Croco_Usfx.png')

    if PLOT_MLD:
        print('* PLotting MLD')
        fig, ax = plt.subplots(2,1,figsize = (10,7),constrained_layout=True,dpi=dpi)
        ax[0].plot( time_gotm/oneday, Ugotm[:,-1], c='b',label='gotm')
        ax[0].plot( time_croco/oneday, Ucroco, c='k',label='croco')
        ax[0].hlines(0.,0,time_croco[-1]/oneday, colors='gray',linestyles='--')
        ax[0].set_ylabel('surface current (m/s)')
        ax[0].legend()
        
        ax[1].plot(time_croco/oneday, Taux, c='limegreen',label=r'$\tau_x$')
        ax[1].plot(time_croco/oneday, Tauy, c='darkorange',label=r'$\tau_y$')
        ax[1].hlines(0.,0,time_croco[-1]/oneday, colors='gray',linestyles='--')
        ax[1].set_ylabel('wind stress (m2/s2)')
        ax[1].set_xlabel('time (days)')
        ax[1].legend()
        
        fig.savefig(path_save+'gotm_vs_Croco_MLD.png')



    dsout.close()
    dstruth.close()
    plt.show()