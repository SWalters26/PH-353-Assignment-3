import numpy as np
import random as rand
import matplotlib.pyplot as plt
import statistics as stat

#instead of delta H, now want S_e, k*T = hbar = 1 for this new stuff 
#S_e = int^{t_f}_{t_0}(dt * L_E)
#L_E = lagrangian for specific case, harmonic oscillator in this case 
#Z = int(D_x e^(-S_e/hbar)) <- partition function 
#for the harmonic oscillator, S_e = a*sum^{N}_{i=1}(1/2 m ((x_{i+1}-x_i)/a)^2 + 1/2 mu^2 (x_i)^2)
#generate path, give it a kick(some random amount a kick or like 1 or 2...), accept/reject, 
#when making a kick, do the accept reject step on the action of new config - action of old config 

def x_initial_func(N):
    x = np.ones(N)
    return x

def x_candidate_func(N, func_to_kick):
    x_init = func_to_kick
    i = rand.randint(0,N-1)
    x_init[i] = x_init[i] * rand.uniform(-2,2)
    x_candidate = x_init
    return x_candidate

def action_calc(N, func):
    a = 1
    mu = 1
    m = 1
    S_e = 0
    x = func
    for i in range(1,N-1):
        S_e += (1/2*m*((x[i+1]-x[i])/a)**2)+1/2*mu**2 * x[i]**2
    return S_e


def metro(N, func_init, func_cand):
    S_e_init = action_calc(N, func_init)
    S_e_cand = action_calc(N, func_cand)
    delta_S_e = S_e_init - S_e_cand 
    if np.exp(-delta_S_e) >= 1:
        return(func_cand) #accept - to be made still
    else:
        if np.exp(-delta_S_e) >= rand.uniform(0,1):
            return(func_cand)#accept - to be made still
        else:
            return(func_init)#reject - to be made still


def metro_repeats(N,runs,func_init):
    func_array = np.zeros((runs,N))
    func_array[0] = func_init
    for i in range(1, runs, 1):
        func_array[i] = metro(N,func_array[i-1],x_candidate_func(N,func_array[i-1]))
    return func_array
