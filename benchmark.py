import numpy as np
import time as clock
import os

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
    """
    """
    JAXED = model.isJax
    class_name = type(model).__name__
    Ltimes = []
    if JAXED:
        Nexec += 1
    for k in range(Nexec):
        t1 = clock.time()
        _,_ = model.do_forward(pk)
        Ltimes.append(clock.time()-t1)
    print_and_save_info(Ltimes, JAXED, class_name, 'benchmark_forward_model')
    
    
def benchmark_cost_function(pk, model, var, Nexec):
    """
    """
    
def benchmark_grad_cost_function(pk, model, var, Nexec):
    """
    """