import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" # slower if false
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".125"
os.environ['XLA_GPU_ENABLE_LATENCY_HIDING_SCHEDULER'] = 'false' # if true, slower
os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = "1"

print(os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"])
print(os.environ['XLA_GPU_ENABLE_LATENCY_HIDING_SCHEDULER'])
#print(os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"])    

# custom imports
from OSSE import *
from observations import *
from forcing import *
from unstek import *
from junstek import *
from inv import *
from tools import *
from scores import *
from benchmark import *

print(jax.devices())

start = clock.time()

SOURCE='Croco'
dt=60
period_obs          = 86400
TRUE_WIND_STRESS=True
point_loc_source = {'MITgcm':[-24.8,45.2], # °E,°N
                    'Croco':[-50.,35.]}

dico_pk_solution ={'MITgcm':{'[-24.8, 45.2]':
                                        {
                                        '1':{'TRUE_TAU'  :[-9.39170447, -9.21727385],
                                            'Cd.UU'     :[-8.44024616, -9.43221639],
                                            
                                            },
                                        '2':{'TRUE_TAU'  :[-9.11293915, -8.73891447, -11.15877101, -12.56505214],
                                            'Cd.UU'     :[-8.26241066,  -9.06349461, -11.31082486, -12.47552063]
                                            } 
                                        }
                                    },
                   'Croco':{ '[-50.0, 35.0]':
                                        {
                                        '1':{'TRUE_TAU'  :[-11.31980127, -10.28525189]
                                             },
                                        '2':{'TRUE_TAU'  :[-10.76035344, -9.3901326, -10.61707124, -12.66052074]
                                             },
                                        '3':{'TRUE_TAU'  :[-8,-9,-10.76035344, -9.3901326, -10.61707124, -12.66052074] # this is WIP
                                             },
                                        '4':{'TRUE_TAU'  :[-7,-7.5,-8,-9,-10.76035344, -9.3901326, -10.61707124, -12.66052074] # this is WIP
                                             }
                                        }
                                     } 
                            }
point_loc = point_loc_source[SOURCE]

path_file = SOURCE+'_Interp_1D_LON'+str(point_loc[0])+'_LAT'+str(point_loc[1])+'.nc'
forcing = Forcing1D(dt, path_file, TRUE_WIND_STRESS)   
observations = Observation1D(period_obs, dt, path_file)

print('* Benchmarking ...')       
Nexec = 20 # >1
dT = 1*86400        

NB_LAYER = [2]

# 1 layer
if 1 in NB_LAYER:
    Nl = 1
    pk = dico_pk_solution[SOURCE][str(point_loc)][str(Nl)]['TRUE_TAU']
    Lmodel = [  Unstek1D(Nl, forcing=forcing, observations=observations),
                jUnstek1D(Nl, forcing=forcing, observations=observations),
                jUnstek1D_Kt(Nl, forcing=forcing, observations=observations, dT=dT)]
    benchmark_all(pk, Lmodel, observations, Nexec)

# 2 layer
if 2 in NB_LAYER:
    Nl = 2
    pk = dico_pk_solution[SOURCE][str(point_loc)][str(Nl)]['TRUE_TAU']
    Lmodel = [  Unstek1D(Nl, forcing=forcing, observations=observations),
                jUnstek1D(Nl, forcing=forcing, observations=observations),
                jUnstek1D_Kt(Nl, forcing=forcing, observations=observations, dT=dT)]
    benchmark_all(pk, Lmodel, observations, Nexec)

# 3 layer
if 3 in NB_LAYER:
    Nl = 3
    pk = dico_pk_solution[SOURCE][str(point_loc)][str(Nl)]['TRUE_TAU']
    Lmodel = [  Unstek1D(Nl, forcing=forcing, observations=observations),
                jUnstek1D(Nl, forcing=forcing, observations=observations),
                jUnstek1D_Kt(Nl, forcing=forcing, observations=observations, dT=dT)]
    benchmark_all(pk, Lmodel, observations, Nexec)
    
# 4 layer
if 4 in NB_LAYER:
    Nl = 4
    pk = dico_pk_solution[SOURCE][str(point_loc)][str(Nl)]['TRUE_TAU']
    Lmodel = [  Unstek1D(Nl, forcing=forcing, observations=observations),
                jUnstek1D(Nl, forcing=forcing, observations=observations),
                jUnstek1D_Kt(Nl, forcing=forcing, observations=observations, dT=dT)]
    benchmark_all(pk, Lmodel, observations, Nexec)
      
end = clock.time()
print('Total execution time = '+str(np.round(end-start,2))+' s')