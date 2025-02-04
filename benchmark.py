import numpy as np
import time as clock
import os
import jax.numpy as jnp

from inv import *

def benchmark_func(pk, func, Nexec):
    Ltimes = []
    for _ in range(Nexec):
        t1 = clock.time()
        _ = func(pk)
        Ltimes.append(clock.time()-t1)
    return np.array(Ltimes)
    
def benchmark_all(pk, Lmodel, observations, Nexec):
    """
    """
    NB_dec = 8
    SAVE_PREVIOUS = False
    name_bench = 'benchmark_'+str(Lmodel[0].nl)+'_layers'
    print(name_bench)
    
    Ltimes_forward = np.zeros((len(Lmodel),Nexec))
    Ltimes_cost = np.zeros((len(Lmodel),Nexec))
    Ltimes_grad = np.zeros((len(Lmodel),Nexec))
    Nb_param = np.zeros(len(Lmodel))
    for k in range(len(Lmodel)):
        model = Lmodel[k]
        print('     running:',type(model).__name__)
        var = Variational(model, observations)
        
        if type(model).__name__ == 'jUnstek1D_Kt':
            jpk = model.kt_2D_to_1D(model.kt_ini(pk))
        
        if model.isJax:
            jpk = jnp.asarray(pk)
            if type(model).__name__ == 'jUnstek1D_Kt':
                jpk = model.kt_2D_to_1D(model.kt_ini(jpk))
            Ltimes_forward[k] = benchmark_func(jpk, model.do_forward_jit, Nexec)
        else:
            jpk = np.asarray(pk)
            Ltimes_forward[k] = benchmark_func(jpk, model.do_forward, Nexec)
        Nb_param[k] = jpk.size            
        Ltimes_cost[k] = benchmark_func(jpk, var.cost, Nexec)
        Ltimes_grad[k] = benchmark_func(jpk, var.grad_cost, Nexec)
    
    # writing in file
    if SAVE_PREVIOUS:
        os.system('mv -f '+name_bench+'.txt '+name_bench+'_previous.txt')
    with open(name_bench+".txt", "w") as f:
        f.write("*=============================================\n")
        f.write('* BENCHMARK: '+str(Nexec)+ ' runs of each func\n')
        f.write('* N layer = '+str(model.nl)+'\n')
        f.write("* C0: MODEL, isJax, Nb param\n")
        f.write("* C1: Mean Execution time (s)\n")
        f.write("* C2: Std Execution time (s)\n")
        f.write("* C3: Compilation time (s)\n")
        
        for k in range(len(Lmodel)):
            name = type(Lmodel[k]).__name__
            JAXED = Lmodel[k].isJax
            if JAXED:
                txt_f = str(np.round( Ltimes_forward[k,0]-Ltimes_forward[k,1],NB_dec ))
                txt_c = str(np.round(Ltimes_cost[k,0]-Ltimes_cost[k,1],NB_dec ))
                txt_g = str(np.round(Ltimes_grad[k,0]-Ltimes_grad[k,1],NB_dec ))
                ind0 = 1
            else:
                ind0 = 0
                txt_f,txt_c,txt_g = '-','-','-'
                
            f.write(name + ', ' + str(JAXED)+', '+str(Nb_param[k])+'\n')
            f.write('   - forward, '+str(np.round( np.mean(Ltimes_forward[ind0:]),NB_dec ))+', '+str(np.round( np.std(Ltimes_forward[ind0:]),NB_dec ))+', '+txt_f+'\n')
            f.write('   - cost   , '+str(np.round(np.mean(Ltimes_cost[ind0:]),    NB_dec ))+', '+str(np.round( np.std(Ltimes_cost[ind0:]),NB_dec ))+   ', '+txt_f+'\n')
            f.write('   - grad   , '+str(np.round( np.mean(Ltimes_grad[ind0:]),   NB_dec ))+', '+str(np.round( np.std(Ltimes_grad[ind0:]),NB_dec ))+   ', '+txt_f+'\n')
        
    f = open(name_bench+'.txt')
    print('Results:')
    for line in f:
        print(line[:-1]),
    f.close()