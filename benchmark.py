import numpy as np
import time as clock
import os
import jax.numpy as jnp

from inv import *

def print_and_save_info(Ltimes, JAXED, class_name, name):
    """
    """
    NB_dec = 5
    Ltimes = np.asarray(Ltimes)
    if JAXED:
        time_compile = Ltimes[0]-Ltimes[1]
        txt_add = str(np.round(time_compile,NB_dec))
        Ltimes = Ltimes[1:]    
    mean_time = np.mean(Ltimes)
    std_time = np.std(Ltimes)
    Ntimes = len(Ltimes)
    Name_file = 'BENCH_'+class_name+'_'+name
    
    # print
    print('*==================================')
    print('* BENCHMARK (s) '+name)
    print('* ')
    print('* IN JAX : '+str(JAXED))
    if JAXED:
        print('*    compile time = '+txt_add)
    print('* ')
    print('* execution time:')
    print('*    mean = '+str(np.round(mean_time,NB_dec)))
    print('*    std  = '+str(np.round(std_time,NB_dec)))
    print('* number of run = '+str(Ntimes))
    print('*==================================')

    os.system('mv '+Name_file+'.txt '+Name_file+'_previous.txt')
    with open(Name_file+".txt", "w") as f:
        f.write("*==================================\n")
        f.write('* BENCHMARK (s) '+name+'\n')
        f.write('* '+'\n')
        f.write('* IN JAX : '+str(JAXED)+'\n')
        if JAXED:
            f.write('*    compile time = '+txt_add+'\n')
        f.write('* '+'\n')
        f.write('* execution time:'+'\n')
        f.write('*    mean = '+str(np.round(mean_time,NB_dec))+'\n')
        f.write('*    std  = '+str(np.round(std_time,NB_dec))+'\n')
        f.write('* number of run = '+str(Ntimes)+'\n')
        f.write('*==================================')



def benchmark_forward_model(pk, model, Nexec):
    Ltimes = []
    for k in range(Nexec):
        t1 = clock.time()
        _,_ = model.do_forward(pk)
        Ltimes.append(clock.time()-t1)
    return np.array(Ltimes)
    
    
def benchmark_cost_function(pk, var, Nexec):
    Ltimes = []
    for _ in range(Nexec):
        t1 = clock.time()
        _ = var.cost(pk)
        Ltimes.append(clock.time()-t1)
    return np.array(Ltimes)
    
def benchmark_grad_cost_function(pk, var, Nexec):
    Ltimes = []
    for _ in range(Nexec):
        t1 = clock.time()
        _ = var.grad_cost(pk)
        Ltimes.append(clock.time()-t1)
    return np.array(Ltimes)
    
def benchmark_all(pk, Lmodel, observations, Nexec):
    """
    """
    NB_dec = 5
    name_bench = 'benchmark_'+Lmodel[k].nl+'_layers'
    
    Ltimes_forward = np.zeros((len(Lmodel),Nexec))
    Ltimes_cost = np.zeros((len(Lmodel),Nexec))
    Ltimes_grad = np.zeros((len(Lmodel),Nexec))
    Nb_param = np.zeros(len(Lmodel))
    for k in range(len(Lmodel)):
        model = Lmodel[k]
        print(type(model).__name__)
        var = Variational(model, observations)
        if model.isJax:
            jpk = jnp.asarray(pk)
        else:
            jpk = np.asarray(pk)
        Nb_param[k] = jpk.size
        Ltimes_forward[k] = benchmark_forward_model(jpk, model, Nexec)
        Ltimes_cost[k] = benchmark_cost_function(jpk, var, Nexec)
        Ltimes_grad[k] = benchmark_grad_cost_function(jpk, var, Nexec)
        
    
    # writing in file
    os.system('mv -f '+name_bench+'.txt '+name_bench+'_previous.txt')
    with open(name_bench+".txt", "w") as f:
        f.write("*=============================================\n")
        f.write('* BENCHMARK: '+str(Nexec)+ ' runs of each func\n')
        f.write('* N layer = '+str(model.nl)+'\n')
        f.write('* Nb param = '+str(Nb_param[k])+'\n')
        f.write("* C0: MODEL\n")
        f.write("* C1: JAX\n")
        f.write("* C2: Mean Execution time (s)\n")
        f.write("* C3: Std Execution time (s)\n")
        f.write("* C4: Compilation time (s)\n")
        
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
                
            f.write(name + ', ' + str(JAXED)+'\n')
            f.write('   - forward, '+str(np.round( np.mean(Ltimes_forward[ind0:]),NB_dec ))+', '+str(np.round( np.std(Ltimes_forward[ind0:]),NB_dec ))+', '+txt_f+'\n')
            f.write('   - cost   , '+str(np.round(np.mean(Ltimes_cost[ind0:]),    NB_dec ))+', '+str(np.round( np.std(Ltimes_cost[ind0:]),NB_dec ))+   ', '+txt_f+'\n')
            f.write('   - grad   , '+str(np.round( np.mean(Ltimes_grad[ind0:]),   NB_dec ))+', '+str(np.round( np.std(Ltimes_grad[ind0:]),NB_dec ))+   ', '+txt_f+'\n')
        
    f = open(name_bench+'.txt')
    for line in f:
        print(line[:-1]),
    f.close()