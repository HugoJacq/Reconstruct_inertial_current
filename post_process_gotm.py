"""
This scripts plots the data from a GOTM simulation
"""

import xarray as xr
import matplotlib.pyplot as plt

# ===================================
# INPUT
workdir = 'gotm_workdir/'
exp = 'Croco_LON-50.0_LAT35.0/'
file_name = 'output_file.nc'
forcing_file = 'home/jacqhugo/Datlas_2025/DATA_Croco/croco_1h_inst_surf_2006-02-01-2006-02-28.nc'


PLOT_CURRENT = True            # plots the gotm current vs true current
# ===================================


dsout = xr.open_dataset(workdir+exp+file_name)
dstruth = xr.open_dataset(forcing_file, )

