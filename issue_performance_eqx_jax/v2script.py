import numpy as np
import os
os.environ["EQX_ON_ERROR"] = "nan"
import time as clock
import jax
import jax.numpy as jnp
from jax import lax
import equinox as eqx
import diffrax
from diffrax import Euler, diffeqsolve, ODETerm
jax.config.update("jax_enable_x64", True)
    
class classic_slab1D_eqx_only(eqx.Module):
    pk : jnp.array = jnp.asarray([-8.,-13.]) 
    TAx : jnp.array = jnp.array(0.2)      
    TAy : jnp.array = jnp.array(0.0)   
    fc : jnp.array = jnp.array(1e-4)      
    dt_forcing : np.int32 =3600  
    t0 : np.int32  = 0      
    t1 : np.int32  = 28*86400       
    nt : np.int32  = 28*86400//60     
    dt : np.int32  = 60      
    
    @eqx.filter_jit
    def __call__(self):
        K = jnp.exp(self.pk)
        U = jnp.zeros( self.nt, dtype='complex')
        
        def __one_step(X0, it):
            K, U = X0  
            TA = self.TAx + 1j*self.TAy
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
    
class classic_slab1D_difx_only(eqx.Module):
    pk : jnp.array = jnp.array([-8.,-13.]) 
    TAx : jnp.array = jnp.array(0.2)      
    TAy : jnp.array = jnp.array(0.0)   
    fc : jnp.array = jnp.array(1e-4)      
    dt_forcing : np.int32 = 3600  
    t0 : np.int32  = 0      
    t1 : np.int32  = 28*86400       
    nt : np.int32  = 28*86400//60     
    dt : np.int32  = 60      
            
    @eqx.filter_jit
    def __call__(self):
        K = jnp.exp(self.pk)
        
        def vector_field(t, C, args):
            U,V = C
            fc, K, TAx, TAy = args
            # physic
            d_U = fc*V + K[0]*TAx - K[1]*U
            d_V = -fc*U + K[0]*TAy - K[1]*V
            d_y = d_U, d_V
            return d_y
        
        sol = diffeqsolve(terms=ODETerm(vector_field), 
                        solver=Euler(), 
                        t0=self.t0, 
                        t1=self.t1, 
                        y0=(0.0, 0.0), #, 
                        args=(self.fc, K, self.TAx, self.TAy), 
                        dt0=None, #dt, 
                        stepsize_controller=diffrax.StepTo(jnp.arange(self.t0, self.t1+self.dt, self.dt)),
                        saveat=diffrax.SaveAt(steps=True),
                        adjoint=diffrax.ForwardMode(),
                        max_steps=self.nt).ys
            
        return sol[0],sol[1]

def benchmark(func, N=10):
    L = np.zeros(N)
    _ = func() # run once for compilation
    for k in range(N):
        time1=clock.time()
        _ = func()
        L[k] = clock.time()-time1
    return L.mean(), L.std()

eqxmodel_only = classic_slab1D_eqx_only()
eqx_dfx_model_only = classic_slab1D_difx_only()

# new models with new control parameter pk
print('Forward model:')
print('     eqx_only:       mean, std (s)', benchmark(eqxmodel_only)) 
print('     eqx_dfx:        mean, std (s)', benchmark(eqx_dfx_model_only))



# end of issue
raise Exception
