import numpy as np
import matplotlib.pyplot as plt
import time as clock
import jax
import jax.numpy as jnp
from jax import jit, lax
import jax.tree_util as jtu
from functools import partial
import equinox as eqx
import diffrax
from diffrax import Euler, diffeqsolve, ODETerm

jax.config.update("jax_enable_x64", True)

class classic_slab1D:
    def __init__(self, TAx, TAy, fc, t0, nt, dt, dt_forcing):
        self.TA = jnp.asarray(TAx) + 1j*jnp.asarray(TAy)
        self.fc = jnp.asarray(fc)  
        self.t0 = t0
        self.t1 = t0+nt*dt
        self.dt = dt
        self.dt_forcing = dt_forcing
        self.nt = nt
        self.do_forward_jit = jit(self.do_forward)
        
    def __one_step(self, X0, it):
        K, U = X0      
        # interpolation
        nsubsteps = self.dt_forcing // self.dt
        itf = jnp.array(it//nsubsteps, int)
        aa = jnp.mod(it,nsubsteps)/nsubsteps
        itsup = lax.select(itf+1>=self.nt, -1, itf+1) 
        TA = (1-aa)*self.TA[itf] + aa*self.TA[itsup]
        # 1 time forward
        U = U.at[it+1].set( U[it] + self.dt*(-1j*self.fc*U[it] 
                                           + K[0]*TA 
                                           - K[1]*U[it] ) )  
        X0 = K, U        
        return X0, X0

    def do_forward(self, pk):
        # initialisation
        U = jnp.zeros( self.nt, dtype='complex')
        K = jnp.exp(pk)
        # time loop
        X0 = K, U
        final, _ = lax.scan(self.__one_step, X0, jnp.arange(0,self.nt-1))
        _, U = final
        return jnp.real(U), jnp.imag(U)
   
class classic_slab1D_eqx(eqx.Module):
    
    pk : jnp.array # control vector
    TA : jnp.array          # parameters
    fc : jnp.array          # |
    dt_forcing : np.int32   # |
    t0 : np.int32           # run parameters
    t1 : np.int32           # |
    nt : np.int32           # | 
    dt : np.int32           # |
    
    is_difx : bool          # use diffrax or not
    
    def __init__(self, pk, TAx, TAy, fc, t0, nt, dt, dt_forcing, is_difx):
        self.fc = jnp.asarray(fc)  
        self.TA = jnp.asarray(TAx) + 1j*jnp.asarray(TAy)
        self.pk = pk
        self.t0 = t0
        self.t1 = t0+nt*dt
        self.dt = dt
        self.dt_forcing = dt_forcing
        self.nt = nt
        self.is_difx = is_difx
        
    @eqx.filter_jit
    def __call__(self):
        K = jnp.exp(self.pk)
        nsubsteps = self.dt_forcing // self.dt
        
        if self.is_difx: # diffrax time forward

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

            sol = diffeqsolve(term=ODETerm(vector_field), 
                           solver=Euler(), 
                           t0=self.t0, 
                           t1=self.t1, 
                           y0=(0.0, 0.0), 
                           args=(self.fc, K, jnp.real(self.TA), jnp.imag(self.TA)), 
                           dt0=dt, 
                           saveat=diffrax.SaveAt(steps=True),
                           adjoint=diffrax.ForwardMode(),
                           max_steps=nt).ys
            U = sol[0]+1j*sol[1]
            
        else: # my time forward
            
            U = jnp.zeros( self.nt, dtype='complex')
            def __one_step(X0, it):
                K, U = X0  
                # interpolation
                itf = jnp.array(it//nsubsteps, int)
                aa = jnp.mod(it,nsubsteps)/nsubsteps
                itsup = lax.select(itf+1>=self.nt, -1, itf+1) 
                TA = (1-aa)*self.TA[itf] + aa*self.TA[itsup]
                # one time forward
                U = U.at[it+1].set( U[it] + self.dt*(-1j*self.fc*U[it] 
                                                + K[0]*TA 
                                                - K[1]*U[it] ) )  
                X0 = K, U        
                return X0, X0
            # time loop
            X0 = K, U
            final, _ = lax.scan(lambda carry, y: __one_step(carry, y), X0, jnp.arange(0,self.nt-1)) # 
            _, U = final
        return jnp.real(U),jnp.imag(U)

def benchmark(func, N=20):
    L = np.zeros(N)
    _ = func() # run once for compilation
    for k in range(N):
        time1=clock.time()
        result = func()
        L[k] = clock.time()-time1
    return L.mean(), L.std()


def cost(sol, obs): return jnp.nanmean( (sol[0]-obs[0])**2 + (sol[0]-obs[0])**2)
    

def cost_j(pk, jmodel, obs):
    sol = jmodel.do_forward_jit(pk)
    return cost(sol, obs)

def cost_eqx(dynamic_model, static_model, obs):
    mymodel = eqx.combine(dynamic_model, static_model)
    sol = mymodel()
    return cost(sol, obs)

def dcost_j(pk, jmodel, obs): return jax.jacfwd(cost_j)(pk, jmodel, obs)
    
def dcost_eqx(dynamic_model, static_model, obs): return eqx.filter_jacfwd(cost_eqx)(dynamic_model, static_model, obs)
    
def my_partition(mymodel):
    filter_spec = jtu.tree_map(lambda arr: False, mymodel) # keep nothing
    filter_spec = eqx.tree_at( lambda tree: tree.pk, filter_spec, replace=True) # keep only pk
    return eqx.partition(mymodel, filter_spec)
    
# MY PARAMETERS
t0 = 0
nt = 28*86400//60
dt = 60
dt_forcing = 3600.
time_forcing = jnp.arange(t0,nt*dt,dt_forcing)
TAx = 0.2*np.ones(len(time_forcing)) # step forcing
TAy = 0.0*np.ones(len(time_forcing))
fc = 1e-4 

pktarget = jnp.asarray([-8.,-13.])
pkini = jnp.array([-9.,-11.])
 

jmodel = classic_slab1D(TAx, TAy, fc, t0, nt, dt, dt_forcing)
eqxmodel = classic_slab1D_eqx(pktarget, TAx, TAy, fc, t0, nt, dt, dt_forcing, is_difx=False)
eqxmodel_dfx = classic_slab1D_eqx(pktarget, TAx, TAy, fc, t0, nt, dt, dt_forcing, is_difx=True)
N = 10

print('Forward model:')
print('     jmodel:         mean, std (s)', benchmark(partial(jmodel.do_forward_jit,pk=pktarget),N=N))
print('     eqxmodel:       mean, std (s)', benchmark(eqxmodel,N=N))
print('     eqx_dfx_model:  mean, std (s)', benchmark(eqxmodel_dfx,N=N))


# make some observations
dt_obs = 86400 # 1 per day
obs = np.nan*jnp.zeros((2,nt),dtype=np.float64)
truth = jmodel.do_forward_jit(pktarget)
for k in range(nt):
    if k%(dt_obs//dt)==0:
        obs = obs.at[0,k].set(truth[0][k]) # U
        obs = obs.at[1,k].set(truth[1][k]) # V

# new models with new control parameter pk
eqxmodel_2 = classic_slab1D_eqx(pkini, TAx, TAy, fc, t0, nt, dt, dt_forcing, is_difx=False)
eqxmodel_dfx_2 = classic_slab1D_eqx(pkini, TAx, TAy, fc, t0, nt, dt, dt_forcing, is_difx=True)
dyn_eqx, stat_eqx = my_partition(eqxmodel_2)
dyn_eqx_dfx, stat_eqx_dfx = my_partition(eqxmodel_dfx_2)

print('Cost:')
print('     jmodel:         mean, std (s)', benchmark(partial(cost_j,pk=pkini,jmodel=jmodel,obs=obs),N=N))
print('     eqxmodel:       mean, std (s)', benchmark(partial(cost_eqx,dyn_eqx,stat_eqx,obs),N=N))
print('     eqx_dfx_model:  mean, std (s)', benchmark(partial(cost_eqx,dyn_eqx_dfx,stat_eqx_dfx,obs),N=N))

print('gradient (forward):')
print('     jmodel:         mean, std (s)', benchmark(partial(dcost_j,pk=pkini,jmodel=jmodel,obs=obs),N=N))
print('     eqxmodel:       mean, std (s)', benchmark(partial(dcost_eqx,dyn_eqx,stat_eqx,obs),N=N))
print('     eqx_dfx_model:  mean, std (s)', benchmark(partial(dcost_eqx,dyn_eqx_dfx,stat_eqx_dfx,obs),N=N))



# end of issue
raise Exception


# plot to be sure its the same results
U1 = eqxmodel(difx=True)
U2 = jmodel.do_forward_jit()
fig, ax = plt.subplots(1,1,figsize = (10,3),constrained_layout=True,dpi=200)
ax.plot(np.arange(t0,nt*dt,dt)/86400,U1, c='k', lw=2, label='eqxmodel(diff=True)')
ax.plot(np.arange(t0,nt*dt,dt)/86400,U2, c='r', lw=2, label='jmodel',ls='--')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Ageo zonal current (m/s)')
plt.show()



# To do : compare gradient computation performance



# point_loc = [-50.,35.]
# path_regrid = '/home/jacqhugo/Datlas_2025/Reconstruct_inertial_current/data_regrid/'
# name_regrid = 'croco_1h_inst_surf_2006-02-01-2006-02-28_0.1deg_conservative.nc'
# ds = xr.open_mfdataset(path_regrid+name_regrid).sel(lon=point_loc[0],lat=point_loc[1],method='nearest')
# TAx = ds.oceTAUX.values
# TAy = ds.oceTAUY.values