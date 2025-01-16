import numpy as np
import xarray as xr
import warnings

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit,grad,lax


cpu_device = jax.devices('cpu')[0]
jax.config.update('jax_platform_name', 'cpu')
print(jax.devices())

class jUnstek1D:
    """
    JAX version, Unsteady Ekman Model 1D, with N layers 
    
    See : https://doi.org/10.3390/rs15184526
    """
    def __init__(self, Nl, forcing, observations):
        # from dataset
        self.nt = len(forcing.time)
        self.dt = forcing.time[1] - forcing.time[0]
        # forcing
        self.TA = jnp.asarray(forcing.TAx) + 1j*jnp.asarray(forcing.TAy) # wind = windU + j*windV
        self.fc = jnp.asarray(forcing.fc)  
        # obs
        Co = observations.get_obs()
        self.Uo, self.Vo = jnp.asarray(Co[0]), jnp.array(Co[1])
        self.Ri = self.Uo*0.+1
        # for reconstructed 
        self.nl = Nl
        self.isJax = True
        
        # JIT compiled functions
        self.do_forward_jit = jit(self.do_forward)
        self.cost_jit = jit(self.cost)
        self.grad_cost_jit = jit(self.grad_cost)
        
    def K_transform(self, pk):
        return jnp.exp(pk)
    
    def one_step(self, it, arg0):
        """
        1 time step advance
        
        INPUT:
            - pk : K vector
            - it: current time step
            - U: velocity at current time step 
        OUTPUT:
            - U: updated velocity at next time step 
        """
        pk ,U = arg0
        K = self.K_transform(pk)
        
        for ik in range(self.nl):
            if ((ik==0)&(ik==self.nl-1)): 
                #U[ik][it+1] = U[ik][it] + self.dt*( -1j*self.fc*U[ik][it] +K[2*ik]*self.TA[it] - K[2*ik+1]*(U[ik][it]) )
                U = U.at[ik,it+1].set( U[ik][it] + self.dt*( -1j*self.fc*U[ik][it] +K[2*ik]*self.TA[it] - K[2*ik+1]*(U[ik][it]) ) )
            else:
                if ik==0: 
                    #U[ik][it+1] = U[ik][it] + self.dt*( -1j*self.fc*U[ik][it] +K[2*ik]*self.TA[it] - K[2*ik+1]*(U[ik][it]-U[ik+1][it]) )
                    U = U.at[ik, it+1].set( U[ik][it] + self.dt*( -1j*self.fc*U[ik][it] +K[2*ik]*self.TA[it] - K[2*ik+1]*(U[ik][it]-U[ik+1][it]) ) )
                elif ik==self.nl-1: 
                    #U[ik][it+1] = U[ik][it] + self.dt*( -1j*self.fc*U[ik][it] -K[2*ik]*(U[ik][it]-U[ik-1][it]) - K[2*ik+1]*U[ik][it] )
                    U = U.at[ik,it+1].set( U[ik][it] + self.dt*( -1j*self.fc*U[ik][it] -K[2*ik]*(U[ik][it]-U[ik-1][it]) - K[2*ik+1]*U[ik][it] ) )
                else: 
                    #U[ik][it+1] = U[ik][it] + self.dt*( -1j*self.fc*U[ik][it] -K[2*ik]*(U[ik][it]-U[ik-1][it]) - K[2*ik+1]*(U[ik][it]-U[ik+1][it]) )
                    U = U.at[ik,it+1].set( U[ik][it] + self.dt*( -1j*self.fc*U[ik][it] -K[2*ik]*(U[ik][it]-U[ik-1][it]) - K[2*ik+1]*(U[ik][it]-U[ik+1][it]) ) )
        return pk, U
    
    def do_forward(self, pk, return_traj=False):
        """
        Unsteady Ekman model forward model
        
        for 1 layer: dU/dt = -j*fc*U + K0*Tau - K1*U

        INPUT:
            - pk     : list of boundaries of layer(s)
            - return_traj : if True, return current as complex number
        OUTPUT:
            - array of surface current
            
        Note: this function works with numpy arrays
        """ 
        if len(pk)//2!=self.nl:
            raise Exception('Your model is {} layers, but you want to run it with {} layers (k={})'.format(self.nl, len(pk)//2,pk))
        else:
            K = self.K_transform(pk)
            
            
        U0 = jnp.zeros((self.nl,self.nt), dtype='complex')
        arg0 = pk, U0
        with warnings.catch_warnings(action="ignore"): # dont show overflow results
            _, U0 = lax.fori_loop(0,self.nt-1,self.one_step,arg0)
            
            # for it in range(self.nt-1):
            #     U0 = self.one_step(pk, it,U0)
        if return_traj: 
            self.Ur_traj = U0
            return U0
        else: 
            self.Ua,self.Va = jnp.real(U0[0,:]), jnp.imag(U0[0,:])
            return self.Ua,self.Va
        
    def cost(self, pk):
        """
        Computes the cost function of reconstructed current vs observations

        INPUT:
            - pk     : K vector
            - Uo    : U current observation
            - Vo    : V current observation
            - Ri    : error on the observation
        OUTPUT:
            - scalar cost
            
        Note: this function works with numpy arrays
        """
        U, V = self.do_forward(pk)
        with warnings.catch_warnings(action="ignore"): # dont show overflow results
            J = 0.5 * jnp.nansum( ((self.Uo - U)*self.Ri)**2 + ((self.Vo - V)*self.Ri)**2 )
        # here use lax.cond 
        
        # def cond(pred, true_fun, false_fun, operand):
        #     if pred:
        #         return true_fun(operand)
        #     else:
        #         return false_fun(operand)
        
        # if jnp.sum( jnp.isnan(U) ) + jnp.sum(jnp.isnan(V))>0:
        #     # some nan have been detected, the model has crashed with 'pk'
        #     # so J is nan.
        #     J = jnp.nan
            
        cond = jnp.sum( jnp.isnan(U) ) + jnp.sum(jnp.isnan(V))>0    
        J = jax.lax.select( cond, jnp.nan, J)
        return J
    
    def grad_cost(self, pk):
        return grad(self.cost)(pk)