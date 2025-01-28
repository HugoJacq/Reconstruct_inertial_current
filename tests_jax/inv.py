import numpy as np
import warnings

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit,grad,lax
from functools import partial
import optax
import optax.tree_utils as otu
from typing import NamedTuple
import chex
import time as clock
import jaxopt



def value_and_grad_jvp(f):   
    return lambda x: ( f(x), jax.jacfwd(f)(x) )
    #return lambda x:  jnp.diagonal( jax.jvp(f, (x,), (jnp.ones(( len(x),len(x) )), ) ) )
def grad_jvp(f):
    return lambda x: value_and_grad_jvp(f)(x)[1]

def value_and_grad_jvp_jit(f):   
    return jit(lambda x: ( f(x), jax.jacfwd(f)(x) ))

class InfoState(NamedTuple):
    iter_num: chex.Numeric



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
        
        #self.run_opt_jit = jit(self.run_opt)   
        
        #self.my_opt_jit = jit(self.my_opt)

  
    def nojax_cost(self, pk):
        """
        Computes the cost function of reconstructed current vs observations

        INPUT:
            - pk     : K vector
        OUTPUT:
            - scalar cost
            
        Note: this function works with numpy arrays
        """
        _, C = self.model.do_forward(pk)
        U, V = np.real(C)[0],np.imag(C)[0]
        A = U[::self.obs_period//self.model_dt]
        B = V[::self.obs_period//self.model_dt]
        J = 0.5 * np.sum( ((self.observations.Uo - A)*self.Ri)**2 + ((self.observations.Vo - B)*self.Ri)**2 )
        return J
 
    def nojax_grad_cost(self, pk):
        """
        Computes the gradient of the cost function

        INPUT:
            - pk    : K vector
        OUTPUT:
            - gradient of the cost function with respect to pk
            
        Note: this function works with numpy arrays
        """
        _, C = self.model.do_forward(pk)
        U, V = np.real(C)[0],np.imag(C)[0]

        # distance to observations (innovation)
        # this is used in the adjoint to add a forcing where obs is available
        d_U, d_V = np.zeros(len(U)), np.zeros(len(V))
        d_U[::self.obs_period//self.model_dt] = (self.observations.Uo - U[::self.obs_period//self.model_dt])*self.Ri
        d_V[::self.obs_period//self.model_dt] = (self.observations.Vo - V[::self.obs_period//self.model_dt])*self.Ri
        # computing the gradient of cost function with TGL
        dJ_pk = self.adjoint(pk, [d_U,d_V])

        return - dJ_pk
    
      
    def jax_cost(self, pk):
        """
        Computes the cost function of reconstructed current vs observations

        INPUT:
            - pk     : K vector
        OUTPUT:
            - scalar cost
            
        Note: this function works with numpy arrays
        """
        
        _, C = self.model.do_forward(pk) # _, C = self.model.do_forward_jit(pk)
        U, V = jnp.real(C)[0], jnp.imag(C)[0]
        
        
        A = jnp.zeros( len(self.observations.time_obs), dtype='float64')
        B = jnp.zeros( len(self.observations.time_obs), dtype='float64')
        #with warnings.catch_warnings(action="ignore"): # dont show overflow results
        A = A.at[:].set(U[::self.obs_period//self.model_dt])
        B = B.at[:].set(V[::self.obs_period//self.model_dt])
        J = 0.5 * jnp.sum( ((self.observations.Uo - A)*self.Ri)**2 + ((self.observations.Vo - B)*self.Ri)**2 )
        # TO DO: 
        # here use lax.cond 
        # to detect if nan like 'nojax_grad_cost'
        #jax.debug.print('inside minimize J = {}', J)
        return J
 
    def jax_grad_cost(self, pk):        
        """
        Using grad from JAX
        """
        #return grad(self.jax_cost_jit)(pk) # this is much slower than jax.jacfwd
        return jax.jacfwd(self.jax_cost_jit)(pk) # this is much faster than jax.grad
        #return jnp.real( grad(self.jax_cost_jit)(pk) )        
            
    

    def cost(self, pk, save_iter=False):
        if self.inJax:
            J = self.jax_cost_jit(pk)
        else:
            J = self.nojax_cost(pk)
        if save_iter:
            self.J.append(J)
        return J.astype(float)   
    
    def grad_cost(self, pk, save_iter=False):
        if self.inJax:
            G = self.jax_grad_cost_jit(pk)
            #print(jax.make_jaxpr(self.jax_grad_cost)(pk))
        else:
            self.adjoint = self.model.adjoint
            self.tgl = self.model.tgl
            G =  self.nojax_grad_cost(pk)
        if save_iter:
            self.G.append(G)
        return G.astype(float)   
    
    
    
    # minimization with jax
    def print_info(self):
        def init_fn(params):
            del params
            return InfoState(iter_num=0)

        def update_fn(updates, state, params, *, value, grad, **extra_args):
            del params, extra_args

            jax.debug.print(
                'Iteration: {i}, Value: {v}, Gradient norm: {e}',
                i=state.iter_num,
                v=value,
                e=otu.tree_l2_norm(grad),
            )
            return updates, InfoState(iter_num=state.iter_num + 1)

        return optax.GradientTransformationExtraArgs(init_fn, update_fn)
    

        
    def my_opt(self, X0, opt, max_iter):
        # trying to do my own minimizer, not working for now
        def step_my_opt(self, carry):
            """
            """
            X0, solver_state = carry
            value = self.jax_cost(X0)
            grad = self.jax_grad_cost(X0)
            #value, grad = value_and_grad_fun(X0) # , state=opt_state
            updates, opt_state = opt.update(grad, opt_state, X0, value=value, grad=grad, value_fn=self.jax_cost)
            X0 = optax.apply_updates(X0, updates)
            return X0, opt_state
        
        opt_state = opt.init(X0)
        #value_and_grad_fun = lambda params: (fn(params),(grad_fn(params)))
        carry = X0, opt_state
        X0, _ = lax.fori_loop(0, max_iter, step_my_opt, carry)
        
        # for _ in range(max_iter):
        #     value = fn(X0)
        #     grad = grad_fn(X0)
        #     #value, grad = value_and_grad_fun(X0) # , state=opt_state
        #     updates, opt_state = solver.update(grad, opt_state, X0, value=value, grad=grad, value_fn=fn)
        #     X0 = optax.apply_updates(X0, updates)
        return X0
    
       
    def run_opt(fun, init_params, opt, max_iter, tol):
        # comes from optax example

        value_and_grad_fun = value_and_grad_jvp_jit(fun)
        def step(carry):
            params, state = carry
            value, grad = value_and_grad_fun(params) # , state=state
            jax.debug.print('value, grad, {}, {}',value,grad)
            updates, state = opt.update(
                grad, state, params, value=value, grad=grad, value_fn=fun
            )
            params = optax.apply_updates(params, updates)
            return params, state

        def continuing_criterion(carry):
            _, state = carry
            iter_num = otu.tree_get(state, 'count')
            grad = otu.tree_get(state, 'grad')
            err = otu.tree_l2_norm(grad)
            return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

        init_carry = (init_params, opt.init(init_params))
        final_params, final_state = jax.lax.while_loop(
            continuing_criterion, step, init_carry
        )
        return final_params, final_state

  
    def jaxopt_opt(self, fun, init_params, max_iter, tol):
        # this is not faster than scipy optimize.
        # I need to better tune this
        
        value_and_grad_fun = value_and_grad_jvp_jit(fun)
       
        opt = jaxopt.LBFGS(value_and_grad_fun, value_and_grad=True, maxiter=max_iter, jit=True,
                           verbose=False, linesearch='zoom',tol=1e-8, stop_if_linesearch_fails=True,linesearch_init='current') # ,tol=tol, jit=False, linesearch_init='current'
        
        res,_ = opt.run(init_params)
        
        return res
