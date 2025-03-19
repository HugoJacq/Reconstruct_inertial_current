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
import scipy

import equinox as eqx
import jax.tree_util as jtu
import optax

from src import constants

# for jax
cpu_device = jax.devices('cpu')[0]
try:
    gpu_device = jax.devices('gpu')[0]
except:
    gpu_device = cpu_device
if constants.FORCE_CPU:
    jax.config.update('jax_platform_name', 'cpu')
    
    

# def value_and_grad_jvp(f):   
#     return lambda x: ( f(x), jax.jacfwd(f)(x) )
#     #return lambda x:  jnp.diagonal( jax.jvp(f, (x,), (jnp.ones(( len(x),len(x) )), ) ) )
# def grad_jvp(f):
#     return lambda x: value_and_grad_jvp(f)(x)[1]

# def value_and_grad_jvp_jit(f):   
#     return jit(lambda x: ( f(x), jax.jacfwd(f)(x) ))


def value_and_grad_jacfwd(cost, pk):
    def cost_for_jacfwd(cost):
        return cost,cost
    grad, value = jax.jacfwd(cost, has_aux=True)(pk)
    return value, grad

class InfoState(NamedTuple):
    iter_num: chex.Numeric


# print infos if scipy.minimize
def print_info(cost,res):
    if np.isnan(cost(res['x'])): # , Uo, Vo, Ri
        print('The model has crashed.')
    else:
        print(' vector K solution ('+str(res.nit)+' iterations)',res['x'])
        print(' cost function value with K solution:',cost(res['x']))
            
class Variational:
    """
    """
    def __init__(self, model, observations):
        self.inJax = model.isJax
        Co = observations.get_obs()
        self.Uo = Co[0]
        self.Vo = Co[1]
        self.Ri = self.Uo*0.+1
        self.observations = observations
        self.obs_period = observations.obs_period
        self.model_dt = model.dt
        self.dt_forcing = model.dt_forcing
        self.model = model
        self.param = []
        self.J = []
        self.G = []

        # JAX
        self.jax_grad_cost_jit = jit(self.jax_grad_cost, device=gpu_device) # faster on gpu
        self.jax_cost_jit = jit(self.jax_cost, device=gpu_device)
        #self.jax_cost_vect_jit = jit(self.jax_cost_vect, device=cpu_device) # faster on cpu
        self.__step_jax_cost_vect_jit_gpu = jit(self.__step_jax_cost_vect, device=gpu_device) 
        self.__step_jax_cost_vect_jit_cpu = jit(self.__step_jax_cost_vect, device=cpu_device) 
        
        #self.run_opt_jit = jit(self.run_opt)   
        #self.my_opt_jit = jit(self.my_opt)

 
    # NO JAX
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
    
    # JAX
    def jax_cost(self, pk):
        """
        Computes the cost function of reconstructed current vs observations

        INPUT:
            - pk     : K vector
        OUTPUT:
            - scalar cost
            
        Note: this function works with numpy arrays
        """
        
        _, C = self.model.do_forward_jit(pk) # _, C = self.model.do_forward_jit(pk)
        U, V = jnp.real(C)[0], jnp.imag(C)[0]
        
        
        A = jnp.zeros( self.Uo.shape, dtype='float64')
        B = jnp.zeros( self.Uo.shape, dtype='float64')
        A = A.at[:].set(U[::self.obs_period//self.dt_forcing])
        B = B.at[:].set(V[::self.obs_period//self.dt_forcing])
        #J = 0.5 * jnp.sum( ((self.observations.Uo - A)*self.Ri)**2 + ((self.observations.Vo - B)*self.Ri)**2 )
        J = jnp.mean( ((self.observations.Uo - A))**2 + ((self.observations.Vo - B))**2 )
        return J 
 
    def jax_grad_cost(self, pk):        
        """
        Using grad from JAX
        """
        #return grad(self.jax_cost_jit)(pk) # this is much slower than jax.jacfwd
        return jax.jacfwd(self.jax_cost)(pk) # this is much faster than jax.grad
        #return jnp.real( grad(self.jax_cost_jit)(pk) )        

    def __step_jax_cost_vect(self,ik,arg0):
            array_pk, indexes, J = arg0
            vector_k = array_pk[ik]
            i,j = indexes[ik][0],indexes[ik][1]
            #jax.debug.print('ik, vector: {}, {}',ik,vector_k)
            J = J.at[i,j].set( self.jax_cost(vector_k) )
            return array_pk, indexes, J  
        
    def jax_cost_vect(self, array_pk, indexes):
        """
        Vectorized version of 'jax_cost'

        INPUT:
            - array_pk : A 2D array with:
                    axis=0 the number of pk vector tested
                    axis=1 the individual pk vector, can be N dimension (tested with 2)
                    
                    example:
                    
                    array_pk = jnp.asarray( [ [k0,k1] for k0 in jtested_values for k1 in jtested_values] )
                    
                    if you want to test values for a bigger than len(vector_k)=2, you can only change 2 values 
                    to be plotable...
        OUTPUT:
            -        
        """
        N, Nl2 = jnp.shape(array_pk) # squared number of vector k, size of vector k
        Nchange = 2 # number of component of vector k that change
        Nl = Nl2//2
        
        sizeN = jnp.sqrt(array_pk.shape[0]).astype(int)
        Jshape = tuple([sizeN]*Nchange)
        J = jnp.zeros( Jshape )
        
        arg0 = array_pk, indexes, J
        if Nl==1:
            _, _, J = lax.fori_loop(0,N,self.__step_jax_cost_vect_jit_cpu,arg0)
        else:
            _, _, J = lax.fori_loop(0,N,self.__step_jax_cost_vect_jit_gpu,arg0)
        return J.transpose()
    
    # COMMON WRAPPER
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
    
    
    
    
    # minimization with jax, WIP
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

        #value_and_grad_fun = value_and_grad_jvp_jit(fun)
        def step(carry):
            params, state = carry
            value, grad = value_and_grad_jacfwd(params) # , state=state
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
        
        value_and_grad_fun =1# value_and_grad_jvp_jit(fun)
       
        opt = jaxopt.LBFGS(value_and_grad_fun, value_and_grad=True, maxiter=max_iter, jit=True,
                           verbose=False, linesearch='zoom',tol=1e-8, stop_if_linesearch_fails=True,linesearch_init='current') # ,tol=tol, jit=False, linesearch_init='current'
        
        res,_ = opt.run(init_params)
        
        return res
    
class Variational_diffrax:
    """
    """
    def __init__(self, model, observations, is_difx=True):
        self.observations = observations
        self.obs_period = observations.obs_period
        self.dt_forcing = model.dt_forcing
        self.model = model
        self.is_difx = is_difx

    def loss_fn(self, obs, sol):
        #print(sol[0].shape,obs[0].shape)
        return jnp.mean( (sol[0]-obs[0])**2 + (sol[1]-obs[1])**2 )
    
    def cost(self, dynamic_model, static_model, call_args):
        mymodel = eqx.combine(dynamic_model, static_model)
        dtime_obs = self.observations.obs_period
        obs = self.observations.get_obs()
        if self.is_difx:
            sol = mymodel(call_args, save_traj_at=dtime_obs).ys # use diffrax and equinox
        else:
            sol = mymodel(call_args, save_traj_at=dtime_obs) # only equinox
        return self.loss_fn(sol, obs)
        
   
        
    def my_partition(self, mymodel):
        filter_spec = jtu.tree_map(lambda arr: False, mymodel) # keep nothing
        filter_spec = eqx.tree_at( lambda tree: tree.pk, filter_spec, replace=True) # keep only pk
        return eqx.partition(mymodel, filter_spec)          
    
    @eqx.filter_jit
    def grad_cost(self, dynamic_model, static_model, call_args):
        def cost_for_grad(dynamic_model, static_model, call_args):
            y = self.cost(dynamic_model, static_model, call_args)
            if static_model.AD_mode=='F':
                return y,y # <- trick to have a similar behavior than value_and_grad (but here returns grad, value)
            else:
                return y
        
        if self.model.AD_mode=='F':
            val1, val2 =  eqx.filter_jacfwd(cost_for_grad, has_aux=True)(dynamic_model, static_model, call_args)
            return val2, val1
        else:
            val1, val2 = eqx.filter_value_and_grad(cost_for_grad)(dynamic_model, static_model, call_args)
            return val1, val2
    
    
    

    def my_minimizer(self, opti, mymodel, itmax, call_args, gtol=1e-5, verbose=False):
        """
        wrapper of optax minimizer, updating 'model' as the loop goes
        """
        
        
        if opti=='adam':
            solver = optax.adam(1e-2)
            opt_state = solver.init(mymodel)
            
            @eqx.filter_jit
            def step_minimize(model, opt, opt_state, call_args):
                dynamic_model, static_model = self.my_partition(mymodel)
                value, grad = self.grad_cost(dynamic_model, static_model, call_args)
                value_grad = grad.pk 
                updates, opt_state = opt.update(grad, opt_state)
                model = eqx.apply_updates(model, updates)
                return value, value_grad, model, opt_state
            
        elif opti=='lbfgs':
            if mymodel.AD_mode=='F':
                raise Exception('Error: LBFGS in optax uses linesearch, that itself uses value_and_grad. You have chosen a forward automatic differentiation, exiting ...')
            else:
                """
                The rest of this minimizer, in reverse mode, is still WIP
                """
                
                
                solver = optax.scale_by_lbfgs() # lbfgs()
                                    # max_history_size=10,           # Number of previous gradients to store (adjust as needed)
                                    # min_step_size=1e-6,             # Minimum step size
                                    # max_step_size=1.0,              # Maximum step size
                                    # line_search_factor=0.5,         # Factor for backtracking line search
                                    # max_line_search_iterations=20)  # Maximum number of line search iterations
                        
                #opt_state = opt.init(value_and_grad_fn, mymodel.pk)
                opt_state = solver.init(mymodel.pk)
                
                
                
                def value_and_grad_fn(params): # L-BFGS expects a function that returns both value and gradient
                        dynamic_model, static_model = self.my_partition(mymodel)
                        new_dynamic_model = eqx.tree_at(lambda m: m.pk, dynamic_model, params) # replace new pk
                        print(static_model)
                        value, grad = self.grad_cost(new_dynamic_model, static_model, call_args)
                        return value, grad.pk
                def cost_fn(params):
                    value, _ = value_and_grad_fn(params)
                    return value
                
                # optax.scale_by_zoom_linesearch() IS USING VALUE_AND_GRAD (SO REVERSE AD) !!!
                # l.1583 in optax/src/linesearch.py
                @eqx.filter_jit
                def step_minimize(carry):
                    model, opt_state = carry
                    params = model.pk
                    
                    #dynamic_model, static_model = self.my_partition(model)
                    value, grad = value_and_grad_fn(params)
                    
                    updates, opt_state = solver.update(grad, opt_state, params) 
                    
                    params = optax.apply_updates(params, updates)
                    
                    print(opt_state)
                    
                    if verbose:
                        # Extract the current iteration number
                        iter_num = otu.tree_get(opt_state, 'count')
                        err = otu.tree_l2_norm(grad)
                        print(f"it: {iter_num}, J: {value}, err: {err}, pk: {params}")
                
                    # Apply updates to model
                    model = eqx.tree_at(lambda m: m.pk, model, updates)
                    # Compute value and gradient at the new point for stopping criterion
                    # value, grad = self.grad_cost(eqx.tree_at(lambda m: m.pk, 
                    #                                          dynamic_model, updates),
                    #                             static_model, 
                    #                             call_args)
                    return value, grad, model, opt_state

                def stop_criterion(prev_cost, next_cost, tol):
                    return jnp.abs(next_cost-prev_cost) >= tol
                
                def gstop_criterion(grad, gtol=1e-5):
                    return jnp.amax(jnp.abs(grad))>=gtol
                    
                

                # initialisation
                
                prev_cost, next_cost = 0,1
                it = 0
                grad = jnp.ones(len(mymodel.pk))
                
                # loop
                #while it<itmax and stop_criterion(prev_cost, next_cost, tol=tol): # #for it in range(itmax):
                while it<itmax and gstop_criterion(grad, gtol): # #for it in range(itmax):
                    carry = mymodel, opt_state
                    value, grad, mymodel, opt_state = step_minimize(carry) #mymodel, solver, opt_state, call_args)
                    #prev_cost = next_cost
                    #next_cost = value
                    if verbose:
                        print("it, J, K :",it, value, mymodel.pk) # value, mymodel
                    it += 1
                    
                print('Final pk is:',mymodel.pk)
                return mymodel
    
    def scipy_lbfgs_wrapper(self, mymodel, maxiter, call_args, verbose=False):
        
        def value_and_grad_for_scipy(params): # L-BFGS expects a function that returns both value and gradient
                    dynamic_model, static_model = self.my_partition(mymodel)
                    new_dynamic_model = eqx.tree_at(lambda m: m.pk, dynamic_model, params) # replace new pk
                    value, grad = self.grad_cost(new_dynamic_model, static_model, call_args)
                    return value, grad.pk
            

        vector_k = mymodel.pk
        print('intial pk',vector_k)
        
        
        res = scipy.optimize.minimize(value_and_grad_for_scipy, 
                                            vector_k,
                                            options={'maxiter':maxiter},
                                            method='L-BFGS-B',
                                            jac=True)
        
        new_k = jnp.asarray(res['x'])
        mymodel = eqx.tree_at( lambda tree:tree.pk, mymodel, new_k)

        if verbose:
            dynamic_model, static_model = self.my_partition(mymodel)
            value, gradtree = self.grad_cost(dynamic_model, static_model, call_args)
            grad = gradtree.pk
            print('final cost, grad')
            print(value, grad)
            print(' vector K solution ('+str(res.nit)+' iterations)',res['x'])
            print(' cost function value with K solution:',value)
        
        return mymodel
        
        
        
        