"""
This file extract Croco data to be input for the GOTM 1D column model
"""
import xarray as xr
from xgcm import Grid as xGrid
from tools import *
import warnings
import pathlib
import sys
import os
import time as clock
# =========================================
# INPUTS

# LON,LAT location where to save forcing files
# This should be in an area where lateral processes 
#   are not dominant.
point_loc = [-50.,35.]

# Croco files
fileC = '/home/jacqhugo/Datlas_2025/DATA_Crocco/croco_1h_inst_surf_2006-02-01-2006-02-28.nc'

# Output name
dir_forcing = './gotm_workdir/'
name_output = 'forcing'

# =========================================
start = clock.time()
print('* Building forcing file for GOTM simulation')

case = 'Croco_LON' + str(point_loc[0])+'_'+'LAT' + str(point_loc[1])+'/'

if not pathlib.Path(dir_forcing+case).is_dir():
    os.system('mkdir '+dir_forcing+case)

print('     Name of file is :',dir_forcing+case+name_output+'.txt')
if pathlib.Path(dir_forcing+case+name_output+'.txt').is_file():
    print('     Forcing file is already here, exiting ...')
    sys.exit(0) # exit script

# opening files
ds = open_croco_sfx_file_at_point_loc(fileC, point_loc)

time = ds.time.values.astype("datetime64[s]")  # convert to 's' to get right format for GOTM
print('     loading dataset ...')
ds = ds.load()

# saving the forcing file
# C0 = time
# C1 = wind stress x
# C2 = wind stress y
# C3 = heat flux 
# C4 = fresh water flux
# C5 = rad flux
print('     saving ...')
with open(dir_forcing+case+name_output+".txt", "w") as f:
    for it in range(len(time)):
        timestr = str(time[it])[:10]+' '+str(time[it])[11:]
        f.write(timestr+'  '+
                    str(ds.oceTAUX.values[it])+'    '+
                        str(ds.oceTAUY.values[it])+'    '+
                            str(ds.Heat_flx_net.values[it])+'    '+
                                str(ds.frsh_water_net.values[it])+'    '+
                                    str(ds.SW_rad.values[it])+'\n')
print('     done !')
print('Total execution time = '+str(np.round(clock.time()-start,2))+' s')