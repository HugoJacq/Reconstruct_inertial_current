import numpy as np
import xarray as xr
import warnings
import matplotlib.pyplot as plt
import time as clock
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, lax
from functools import partial
import equinox as eqx
import timeit
import diffrax
from diffrax import Euler, diffeqsolve, ODETerm

# def my_interpolate(t, time_array, data_array):
#         # Custom efficient interpolation 
#         idx = jnp.searchsorted(time_array, t) - 1
#         idx = jnp.clip(idx, 0, len(time_array) - 2)
#         t0, t1 = time_array[idx], time_array[idx + 1]
#         y0, y1 = data_array[idx], data_array[idx + 1]
#         alpha = (t - t0) / (t1 - t0)
#         return y0 + alpha * (y1 - y0)


class classic_slab1D:
    def __init__(self, pk, TAx, TAy, fc, t0, nt, dt, dt_forcing):
        self.TA = jnp.asarray(TAx) + 1j*jnp.asarray(TAy)
        self.fc = jnp.asarray(fc)  
        self.pk = pk
        self.t0 = t0
        self.t1 = t0+nt*dt
        self.dt = dt
        self.dt_forcing = dt_forcing
        self.nt = nt
        
        self.do_forward_jit = jit(self.do_forward)
        
    def __one_step(self, X0, it):
        K, U = X0      
        #current_time = it*self.dt
        #TA = self.TA_t(current_time)
        
        nsubsteps = self.dt_forcing // self.dt
        itf = jnp.array(it//nsubsteps, int)

        aa = jnp.mod(it,nsubsteps)/nsubsteps
        itsup = lax.select(itf+1>=self.nt, -1, itf+1) 
        TA = (1-aa)*self.TA[itf] + aa*self.TA[itsup]
        
        U = U.at[it+1].set( U[it] + self.dt*(-1j*self.fc*U[it] 
                                           + K[0]*TA 
                                           - K[1]*U[it] ) )  
        X0 = K, U        
        return X0, X0

    def do_forward(self):
        U = jnp.zeros( self.nt, dtype='complex')
        K = jnp.exp(self.pk)
        
        X0 = K, U
        final, _ = lax.scan(self.__one_step, X0, jnp.arange(0,self.nt-1))
        _, U = final
        return jnp.real(U)
   

class classic_slab1D_eqx(eqx.Module):
    # control vector
    pk : jnp.array
    # parameters
    TA : jnp.array
    fc : jnp.array
    # run parameters
    t0 : np.int32
    t1 : np.int32
    nt : np.int32
    dt : np.int32
    dt_forcing : np.int32
    
    def __init__(self, pk, TAx, TAy, fc, t0, nt, dt, dt_forcing):
        self.fc = jnp.asarray(fc)  
        self.TA = jnp.asarray(TAx) + 1j*jnp.asarray(TAy)
        self.pk = pk
        self.t0 = t0
        self.t1 = t0+nt*dt
        self.dt = dt
        self.dt_forcing = dt_forcing
        self.nt = nt

    def __one_step(self, X0, it):
        K, U = X0  
        nsubsteps = self.dt_forcing // self.dt
        itf = jnp.array(it//nsubsteps, int)

        aa = jnp.mod(it,nsubsteps)/nsubsteps
        itsup = lax.select(itf+1>=self.nt, -1, itf+1) 
        TA = (1-aa)*self.TA[itf] + aa*self.TA[itsup]
         
        U = U.at[it+1].set( U[it] + self.dt*(-1j*self.fc*U[it] 
                                           + K[0]*TA 
                                           - K[1]*U[it] ) )  
        X0 = K, U        
        return X0, X0
    
    
    @eqx.filter_jit
    def __call__(self, difx=True):
        U = jnp.zeros( self.nt, dtype='complex')
        K = jnp.exp(self.pk)
        
        if difx:
            
            
            nsubsteps = self.dt_forcing // dt
            
            def vector_field(t, C, args):
                U,V = C
                fc, K, TAx, TAy = args
                
                # on the fly interpolation
                it = jnp.array(t//dt, int)
                itf = jnp.array(it//nsubsteps, int)
                aa = jnp.mod(it,nsubsteps)/nsubsteps
                itsup = lax.select(itf+1>=len(TAx), -1, itf+1) 
                TAx = (1-aa)*TAx[itf] + aa*TAx[itsup]
                TAy = (1-aa)*TAy[itf] + aa*TAy[itsup]
                
                # physic
                d_U = fc*V + K[0]*TAx - K[1]*U
                d_V = -fc*U + K[0]*TAy - K[1]*V
                d_y = d_U,d_V
                return d_y
            
            
            term =ODETerm(vector_field)
            saveat = diffrax.SaveAt(steps=True)
            #saveat = diffrax.SaveAt(ts=jnp.arange(self.t0,self.t1,self.dt_forcing))
            
            y0 = 0.0, 0.0
            args = self.fc, K, jnp.real(self.TA), jnp.imag(self.TA)
            
            sol = diffeqsolve(term, 
                           solver=Euler(), 
                           t0=self.t0, 
                           t1=self.t1, 
                           y0=y0, 
                           args=args, 
                           dt0=dt, 
                           saveat=saveat,
                           adjoint=diffrax.ForwardMode(),
                           max_steps=nt).ys
            U = sol[0]
            
        else:
            
            X0 = K, U
            final, _ = lax.scan(lambda carry, y: self.__one_step(carry, y), X0, jnp.arange(0,self.nt-1)) # 
            _, U = final
            U = jnp.real(U)
        return U



def benchmark(func, N=20):
    L = np.zeros(N)
    _ = func() # run once for compilation
    for k in range(N):
        time1=clock.time()
        _ = func()
        L[k] = clock.time()-time1
    return L.mean(), L.std()

point_loc = [-50.,35.]
path_regrid = '/home/jacqhugo/Datlas_2025/Reconstruct_inertial_current/data_regrid/'
name_regrid = 'croco_1h_inst_surf_2006-02-01-2006-02-28_0.1deg_conservative.nc'
ds = xr.open_mfdataset(path_regrid+name_regrid).sel(lon=point_loc[0],lat=point_loc[1],method='nearest')
t0=0
nt = 28*86400//60
# t1=28*86400
dt=60
dt_forcing = 3600.
TAx = ds.oceTAUX.values
TAy = ds.oceTAUY.values
fc = 1e-4 
pk = jnp.array([-10,-9])
    
difx = True # use diffeqsolve if True

jmodel = classic_slab1D(pk, TAx, TAy, fc, t0, nt, dt, dt_forcing)
eqxmodel = classic_slab1D_eqx(pk, TAx, TAy, fc, t0, nt, dt, dt_forcing)
eqxmodel_dfx = partial(eqxmodel,difx=difx)
N = 10



print('eqxmodel: mean, std (s)',benchmark(eqxmodel_dfx,N=N))
print('jmodel: mean, std (s)',  benchmark(jmodel.do_forward_jit,N=N))

U1 = eqxmodel(difx=difx)
U2 = jmodel.do_forward_jit()

fig, ax = plt.subplots(1,1,figsize = (10,3),constrained_layout=True,dpi=200)
ax.plot(np.arange(0,len(ds.time)*dt_forcing,dt_forcing)/86400, ds.U, c='b')
ax.plot(np.arange(t0,nt*dt,dt)/86400,U1, c='k', lw=2, label='eqxmodel(diff='+str(difx))
ax.plot(np.arange(t0,nt*dt,dt)/86400,U2, c='r', lw=2, label='jmodel',ls='--')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Ageo zonal current (m/s)')

plt.show()



# To do : compare gradient computation performance