import numpy as np
import warnings

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit,grad,lax


class Variational:
    """
    """
    def __init__(self, model, observations):
        self.inJax = model.isJax
        self.Ri = model.Ri
        self.observations = observations
        self.model = model

        
  
    def nojax_cost(self, pk):
        """
        Computes the cost function of reconstructed current vs observations

        INPUT:
            - pk     : K vector
        OUTPUT:
            - scalar cost
            
        Note: this function works with numpy arrays
        """
        U, V = self.model.do_forward(pk)
        with warnings.catch_warnings(action="ignore"): # dont show overflow results
            J = 0.5 * np.nansum( ((self.observations.Uo - U)*self.Ri)**2 + ((self.observations.Vo - V)*self.Ri)**2 )
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
        U, V = self.model.do_forward(pk)

        # distance to observations (innovation)
        # this is used in the adjoint to add a forcing where obs is available
        d_U = (self.observations.Uo - U)*self.Ri
        d_V = (self.observations.Vo - V)*self.Ri
        #   = 0 where no observation available
        d_U[np.isnan(d_U)]=0.
        d_V[np.isnan(d_V)]=0.
        # computing the gradient of cost function with TGL
        dJ_pk = self.adjoint(pk, [d_U,d_V])

        return -dJ_pk
    
    def jax_cost(self, pk):
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
        U, V = self.model.do_forward(pk)
        with warnings.catch_warnings(action="ignore"): # dont show overflow results
            J = 0.5 * jnp.nansum( ((self.observations.Uo - U)*self.Ri)**2 + ((self.observations.Vo - V)*self.Ri)**2 )
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
 
    def jax_grad_cost(self, pk):
        """
        """
        return grad(self.jax_cost)(pk)
    
    def cost(self, pk):
        if self.inJax:
            return self.jax_cost(pk)
        else:
            return self.nojax_cost(pk)

    def grad_cost(self, pk):
        if self.inJax:
            return self.jax_grad_cost(pk)
        else:
            self.adjoint = self.model.adjoint
            self.tgl = self.model.tgl
            return  self.nojax_grad_cost(pk)