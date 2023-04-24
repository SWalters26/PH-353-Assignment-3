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
'''
def metro(x,N):

    count = 0
    x_array = np.zeros(N+1)
    x_array[count] = x
    x_candidate = x
    while count < N:
        x_candidate = x*rand.uniform(0.1,1.9)
        delta_H = (x_array[count])**2-x_candidate**2
        if np.exp(-delta_H) >= 1:
            count += 1 
            x_array[count] = x_candidate
        else:
            if x_candidate >= rand.uniform(0,1): #accept case
                count += 1
                x_array[count] = x_candidate
            else: #reject case
                x_candidate = x_array[count]
                count += 1
                x_array[count] = x_candidate
    average_metro = np.mean(x_array)
    uncertainty = np.sqrt((1/(N-1)))*stat.stdev(x_array)
    return x_array,uncertainty


'''

def x_initial_func(N):
    x = np.ones(N)
    return x

def x_candidate_function(N):
    x_init = x_initial_func(N)
    i = rand(1,N-1)
    x_init[i] += rand(-10,10)
    x_candidate = x_init
    return x_candidate

def action_calc(N, func):
    a = 1
    mu = 10
    m = 2
    S_e = 0
    x = func(N)
    for i in range(1,N-1):
        S_e += (1/2*m*((x[i+1]-x[i])/a)**2)+1/2*mu**2 * x[i]**2
    return S_e


def metro(N):
    S_e_init = action_calc(N, x_initial_func(N))
    S_e_cand = action_calc(N, x_candidate_function(N))
    delta_S_e = S_e_init - S_e_cand 
    if np.exp(-delta_S_e) >= 1:
        print(True) #accept
    else:
        if np.exp(-delta_S_e) >= rand.uniform(0,1):
            print(True)
        else:
            print(False)
    
    
