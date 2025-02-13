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
print('* Building forcing file for GOTM simulation')

case = 'Croco_LON' + str(point_loc[0])+'_'+'LAT' + str(point_loc[1])+'/'

if not pathlib.Path(dir_forcing+case).is_dir():
    os.system('mkdir '+dir_forcing+case)

print('     Name of file is :',dir_forcing+case+name_output+'.txt')
if pathlib.Path(dir_forcing+case+name_output+'.txt').is_file():
    print('     Forcing file is already here, exiting ...')
    #sys.exit(0) # exit script

ds, xgrid = open_croco_sfx_file(fileC)
# # opening files
# size_chunk = 200
# with warnings.catch_warnings():
#     warnings.simplefilter('ignore')
#     # warning are about chunksize, because not all chunk have the same size.
#     ds = xr.open_dataset(fileC,chunks=-1)
#                         # ,chunks={'time_counter': -1,
#                         #     'x_rho': size_chunk,
#                         #     'y_rho': size_chunk,
#                         #     'y_u': size_chunk, 
#                         #     'x_u': size_chunk,
#                         #     'y_v': size_chunk, 
#                         #     'x_v': size_chunk,})

# # removing used variables
# ds = ds.drop_vars(['bvstr','bustr','ubar','vbar','hbbl','h',
#                     'time_instant','time_instant_bounds','time_counter_bounds'])

# # rename redundant dimensions
# _dims = (d for d in ['x_v', 'y_u', 'x_w', 'y_w'] if d in ds.dims)
# for d in _dims:
#     ds = ds.rename({d: d[0]+'_rho'})

# # renaming variables
# ds = ds.rename({'zeta':'SSH',
#                 'sustr':'oceTAUX',
#                 'svstr':'oceTAUY',
#                 'shflx':'Heat_flx_net',
#                 'swflx':'frsh_water_net',
#                 'swrad':'SW_rad',
#                 'hbl':'MLD',
#                 'u':'U',
#                 'v':'V',
#                 'time_counter':'time'})
# if 'nav_lat_rho' in ds.variables:
#     ds = ds.rename({'nav_lat_rho':'lat_rho',
#                     'nav_lon_rho':'lon_rho',
#                     'nav_lat_u':'lat_u',
#                     'nav_lat_v':'lat_v',
#                     'nav_lon_u':'lon_u',
#                     'nav_lon_v':'lon_v'})



# reduce dataset size around point_loc
edge = 1 # °
LON_bounds = [point_loc[0]-edge,point_loc[0]+edge]
LAT_bounds = [point_loc[1]-edge,point_loc[1]+edge]
indices = find_indices_gridded_latlon(ds.lon_rho.values, LON_bounds, ds.lat_rho.values, LAT_bounds)
indxmin, indxmax, indymin, indymax = indices
# smaller domain
ds = ds.isel(x_rho=slice(indxmin,indxmax),
            y_rho=slice(indymin,indymax),
            x_u=slice(indxmin,indxmax),
            y_v=slice(indymin,indymax) )

# building a new xgcm grid on the reduced dataset
coords={'x':{'center':'x_rho',  'right':'x_u'}, 
        'y':{'center':'y_rho', 'right':'y_v'}}    
grid = xGrid(ds, 
    coords=coords,
    boundary='extend')

# interp at mass point
print('Interp at mass point')
for var in ['U','oceTAUX']:
    attrs = ds[var].attrs
    ds[var] = grid.interp(ds[var], 'x')
    ds[var].attrs = attrs
for var in ['V','oceTAUY']:
    attrs = ds[var].attrs
    ds[var] = grid.interp(ds[var], 'y')
    ds[var].attrs = attrs
# remove non-rho coordinates
ds = ds.drop_vars({'lat_u','lat_v','lon_u','lon_v'})
# all onvariables are now on rho points:  
ds = ds.rename({'lon_rho':'lon','lat_rho':'lat'})

# reducing dataset at the 'point_loc'
indx,indy = find_indices(point_loc,ds.lon.values,ds.lat.values)[0]
ds = ds.isel(x_rho=indx,y_rho=indy)
time = ds.time.values.astype("datetime64[s]")  # convert to 's' to get right format for GOTM
print('     computing ...')
ds = ds.load()
# ploting the forcing


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