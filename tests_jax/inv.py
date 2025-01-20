import numpy as np
import warnings

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit,grad,lax
from functools import partial

class Variational:
    """
    """
    def __init__(self, model, observations):
        self.inJax = model.isJax
        self.Ri = model.Ri
        self.observations = observations
        self.obs_period = observations.obs_period
        self.model_dt = model.dt
        self.model = model
        self.param = []
        self.J = []
        self.G = []

        # JAX
        self.jax_grad_cost_jit = jit(self.jax_grad_cost)
        self.jax_cost_jit = jit(self.jax_cost)
  
    
  
    def nojax_cost(self, pk):
        """
        Computes the cost function of reconstructed current vs observations

        INPUT:
            - pk     : K vector
        OUTPUT:
            - scalar cost
            
        Note: this function works with numpy arrays
        """
        U_0 = np.zeros((self.model.nl), dtype='complex')
        _, C = self.model.do_forward(pk,U_0)
        U, V = np.real(C),np.imag(C)
        
        with warnings.catch_warnings(action="ignore"): # dont show overflow results
            A = U[::self.obs_period//self.model_dt]
            B = V[::self.obs_period//self.model_dt]
            J = 0.5 * np.nansum( ((self.observations.Uo - A)*self.Ri)**2 + ((self.observations.Vo - B)*self.Ri)**2 )
        if np.sum( np.isnan(U) ) + np.sum(np.isnan(V))>0:
            # some nan have been detected, the model has crashed with 'pk'
            # so J is nan.
            J = np.nan
        return J
 
    def nojax_grad_cost(self, pk):
        """
        Computes the gradient of the cost function

        INPUT:
            - pk    : K vector
        OUTPUT:
            - gradient of the cost function
            
        Note: this function works with numpy arrays
        """
        U_0 = np.zeros((self.model.nl), dtype='complex')
        _, C = self.model.do_forward(pk,U_0)
        U, V = np.real(C),np.imag(C)

        # distance to observations (innovation)
        # this is used in the adjoint to add a forcing where obs is available
        d_U, d_V = np.zeros(len(U)), np.zeros(len(V))
        d_U[::self.obs_period//self.model_dt] = (self.observations.Uo - U[::self.obs_period//self.model_dt])*self.Ri
        d_V[::self.obs_period//self.model_dt] = (self.observations.Vo - V[::self.obs_period//self.model_dt])*self.Ri
        # computing the gradient of cost function with TGL
        dJ_pk = self.adjoint(pk, [d_U,d_V])

        return -dJ_pk
    
      
    def jax_cost(self, pk):
        """
        Computes the cost function of reconstructed current vs observations

        INPUT:
            - pk     : K vector
        OUTPUT:
            - scalar cost
            
        Note: this function works with numpy arrays
        """
        
        _, C = self.model.do_forward(pk) # 
        U, V = jnp.real(C)[0], jnp.imag(C)[0]
        
        #with warnings.catch_warnings(action="ignore"): # dont show overflow results
        A = U[::self.obs_period//self.model_dt]
        B = V[::self.obs_period//self.model_dt]
        J = 0.5 * jnp.sum( ((self.observations.Uo - A)*self.Ri)**2 + ((self.observations.Vo - B)*self.Ri)**2 )
        # TO DO: 
        # here use lax.cond 
        # to detect if nan like 'nojax_grad_cost'
        return J
 
    def jax_grad_cost(self, pk):        
        """
        Using grad from JAX
        """
        return grad(self.jax_cost_jit)(pk)
        #return jnp.real( grad(self.jax_cost_jit)(pk) )        
            
    

    def cost(self, pk, save_iter=False):
        if self.inJax:
            J = self.jax_cost_jit(pk) #
        else:
            J = self.nojax_cost(pk)
        if save_iter:
            self.J.append(J)
        return float(J)   
    
    def grad_cost(self, pk, save_iter=False):
        if self.inJax:
            G = self.jax_grad_cost_jit(pk)
            print(jax.make_jaxpr(self.jax_grad_cost)(pk))
        else:
            self.adjoint = self.model.adjoint
            self.tgl = self.model.tgl
            G =  self.nojax_grad_cost(pk)
        if save_iter:
            self.G.append(G)
        return np.array(G)