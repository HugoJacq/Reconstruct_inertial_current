"""
This file is the main file of the study 'Inertial_current_from_MITgcm'

python reconstruct_intertial.py

by: Hugo Jacquet and Clément Ubelmann

Updates: 
    - CU 6/01/25: creation 
    - HJ 7/01/25: add comments everywhere
    - HJ 9/01/25: add build large scale variable file
"""
from model_unstek import *
from build_LS_fields import *
import time as clock

start = clock.time()

# list of files
filesUV = np.sort(glob.glob("/home/jacqhugo/Datlas_2025/DATA/U_V/llc2160_2020-11-*_SSU-SSV.nc"))
filesH = np.sort(glob.glob("/home/jacqhugo/Datlas_2025/DATA/SSH/llc2160_2020-11-*_SSH.nc"))
filesW = np.sort(glob.glob("/home/jacqhugo/Datlas_2025/DATA/v10m/llc2160_2020-11-*_v10m.nc"))
filesD = np.sort(glob.glob("/home/jacqhugo/Datlas_2025/DATA/KPPhbl/llc2160_2020-11-*_KPPhbl.nc"))

# area of interest
box= [-25, -24, 45., 46, 24000., 24010.] # bottom lat, top lat, left lon, right lon
# spatial location. We work in 1D
ir=4
jr=4

# UNSTEADY EKMAN MODEL
dt=60 # timestep
# LAYER DEFINITION
#   -> number of values = number of layers
#   -> values = turbulent diffusion coefficient
#pk=np.array([-3,-12])  # 1 layers
pk=np.array([-3,-8,-10,-12]) # 2 layers
#pk=np.array([-2,-4,-6,-8,-10,-12]) # 3 layers


# Large scale

# constants
grav = 10 # m/s2



# Building LS files
print('* Getting large scale motion ...')
path_save = './LS_fields'
#build_LS_files(filesUV,box, my2dfilter, mytimefilter, path_save)
build_LS_files(filesH, box, path_save)
#build_LS_files(filesW,box, my2dfilter, mytimefilter, path_save)
#build_LS_files(filesD,box, my2dfilter, mytimefilter, path_save)

# stress as: C x wind**2
# gTAx = 8e-6*np.sign(gTx)*gTx**2
# gTAy = 8e-6*np.sign(gTy)*gTy**2
end = clock.time()
print('Total execution time = '+str(np.round(end-start,2))+' s')
#plt.show()