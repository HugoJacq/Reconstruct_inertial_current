import os
import sys
sys.path.insert(0,'..')
from model_unstek import * # old

import matplotlib.pyplot as plt
import scipy.optimize as opt
from optimparallel import minimize_parallel
import time as clock

from unstek import *
from junstek import *
from forcing import *
from observations import *
from inv import *

import jax
import jax.numpy as jnp
import optax
import optax.tree_utils as otu
jax.config.update("jax_enable_x64", True)
#jax.config.update("jax_debug_nans", True)
cpu_device = jax.devices('cpu')[0]
jax.config.update('jax_platform_name', 'cpu')
print(jax.devices())

# initial vector_k
vector_k = np.array([-3.,-12.]) # 1 layer
#pk = np.array([-3.61402696, -9.44992617]) # solution of minimization
#pk = np.array([-3.9, -9.5]) # test near the solution
#pk=np.array([-3.,-8.,-10.,-12.])
dt = 60 # s

path_file = '../Interp_1D_LON-24.8_LAT45.2.nc'
TRUE_WIND_STRESS = False
period_obs = 86400 # s, how many second between observations

MINIMIZE = True
save_iter = False # save iteration during minimize
maxiter = 100
PARALLEL_MINIMIZED = False
N_CPU = 12
PRINT_INFO = False

# Plot
dpi=200

Nl = len(vector_k)//2
forcing = Forcing1D(dt, path_file, TRUE_WIND_STRESS)   
observations = Observation1D(period_obs, dt, path_file)
Uo,Vo = observations.get_obs()
U, V = forcing.U, forcing.V


##########
# Normal #
##########
if True:
    t1 = clock.time()
    
    model = Unstek1D(Nl, forcing, observations)
    var = Variational(model, observations)
    U_0 = np.zeros((model.nl), dtype='complex')
    
    # First we test that the function work
    pk = vector_k
    print('pk',pk)
    
    # testing the functions
    if False:
        U_0 = jnp.zeros((model.nl), dtype='complex')
        _, Ca = model.do_forward(pk, U_0)
        Ua, Va = jnp.real(Ca), jnp.imag(Ca)
            
        J = var.cost(pk)
        print('J',J)
        
        dJ = var.grad_cost(pk)
        print('dJ',dJ)
    
        if False:
            print('gradient test')
            pk=np.array([-3,-12])
            eps=1e-8

            _, Ca = model.do_forward(pk, U_0)
            Ua, Va = np.real(Ca), np.imag(Ca)

            _, Ca1 = model.do_forward(pk+eps*pk, U_0)
            Ua1, Va1 = np.real(Ca1), np.imag(Ca1)
            
            print(Ua)
            print(Ua1)
            
            dCa = model.tgl(pk, pk)
            dUa, dVa = np.real(dCa), np.imag(dCa)
            print(dUa)
            
            
            print((Ua1-Ua)/eps)
            #print(np.sum((Ua1-Ua)))
            
            _, Ca = model.do_forward(pk, U_0)
            Ua, Va = np.real(Ca), np.imag(Ca)
            X=+pk

            dU = model.tgl(pk, X)
            MX = [np.real(dU),np.imag(dU)]
            Y = [Ua,Va]
            MtY =  model.adjoint(pk, Y)

            # the two next print should be equal
            print(np.sum(MX[0]*Y[0]+MX[1]*Y[1]))
            print(np.sum(X*MtY))
            # the next print should be << 1
            print( (np.abs(np.sum(MX[0]*Y[0]+MX[1]*Y[1]) - np.sum(X*MtY)))/ np.abs(np.sum(X*MtY)))
            
            raise Exception(' the gradient test is finished')
        
    # minimization to find k        
    if MINIMIZE:
        res = opt.minimize(var.cost, pk, args=(save_iter), # , args=(Uo, Vo, Ri)
                        method='L-BFGS-B',
                        jac=var.grad_cost,
                        options={'disp': True, 'maxiter': maxiter})
            
        if np.isnan(var.cost(res['x'])): # , Uo, Vo, Ri
            print('The model has crashed.')
        else:
            print(' vector K solution ('+str(res.nit)+' iterations)',res['x'])
            print(' cost function value with K solution:',var.cost(res['x'])) # , Uo, Vo, Ri
        pk = res['x']
    
    _, Ca = model.do_forward(pk,U_0)
    Ua, Va = np.real(Ca),np.imag(Ca)
    RMSE = score_RMSE(Ua, U)     
    
    # PLOT
    title = ''
    for k in range(len(pk)):
        title += 'k'+str(k)+','   
    title = title[:-1]+' = '+str(pk) + ' ('+str(np.round(RMSE,3))+')'
    
    plt.figure(figsize=(10,3),dpi=dpi)
    plt.plot(forcing.time/86400,U, c='k', lw=2, label='LLC ref')
    plt.plot(forcing.time/86400,Ua, c='g', label='Unstek')
    plt.scatter(observations.time_obs/86400,Uo, c='r', label='obs')
    plt.title(title)
    plt.xlabel('Time (days)')
    plt.ylabel('Ageo zonal current (m/s)')
    plt.legend(loc=1)
    plt.tight_layout()
    plt.savefig('NOJAX_series_reconstructed_long_'+str(model.nl)+'layers.png')
    print('-> time for no jax = ',np.round(clock.time() - t1,4) )

###############
# JAX version #
###############
"""
DEBUGING:

- jax.debug.print() for traced (dynamic) array values with jax.jit(), jax.vmap() and others
- print() for static values, such as dtypes and array shapes
"""
if True:   
    
    t1 = clock.time()
    model = jUnstek1D(Nl, forcing, observations)
    var = Variational(model, observations)
    
    pk = vector_k
    print('pk',pk)
        
    
    # testing the functions
    if False:
        
        U_0 = jnp.zeros((model.nl), dtype='complex')
        _, Ca = model.do_forward_jit(pk, U_0)
        Ua2, Va2 = jnp.real(Ca)[0], jnp.imag(Ca)[0]
        
        
        J = var.cost(pk)
        print('J',J)
        
        dJ = var.grad_cost(pk)
        print('dJ',dJ)
        

    # To do later : jax this minimization
    # maybe see: 
    #   - optimix
    #   - optax, example LBFGS
    if MINIMIZE:
        res = opt.minimize(var.cost, pk, args=(save_iter),
                            method='L-BFGS-B',
                            jac=var.grad_cost,
                            options={'disp': True, 'maxiter': maxiter})
        if np.isnan(var.cost(res['x'])): # , Uo, Vo, Ri
            print('The model has crashed.')
        else:
            print(' vector K solution ('+str(res.nit)+' iterations)',res['x'])
            print(' cost function value with K solution:',var.cost(res['x'])) # , Uo, Vo, Ri
        pk = res['x']
    
    _, Ca2 = model.do_forward_jit(pk,U_0)
    Ua2, Va2 = np.real(Ca2)[0],np.imag(Ca2)[0]
    RMSE = score_RMSE(Ua, U)  
       
    # PLOT
    title = ''
    for k in range(len(pk)):
        title += 'k'+str(k)+','   
    title = title[:-1]+' = '+str(pk) #+ ' ('+str(np.round(RMSE,3))+')'
    
    plt.figure(figsize=(10,3),dpi=dpi)
    plt.plot(forcing.time/86400,U, c='k', lw=2, label='LLC ref')
    plt.plot(forcing.time/86400,Ua2, c='g', label='Unstek')
    plt.scatter(observations.time_obs/86400,Uo, c='r', label='obs')
    plt.title(title)
    plt.xlabel('Time (days)')
    plt.ylabel('Ageo zonal current (m/s)')
    plt.legend(loc=1)
    plt.tight_layout()
    plt.savefig('JAX_series_reconstructed_long_'+str(model.nl)+'layers.png')
    print('-> time for with jax = ',np.round(clock.time() - t1,4) )
    
        
plt.show()

