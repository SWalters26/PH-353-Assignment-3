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

def x_initial_func(N): #N is size of initial function
    x = np.ones(N)
    return x

def x_candidate_func(N, func_to_kick):
    x_init = func_to_kick
    i = rand.randint(0,N-1)
    r = rand.randint(0,1)
    d = rand.randint(0,1)
    if r == 0:
        sign = -1
    else:
        sign = 1
        
    if d == 0:
        b = rand.uniform(-1.1,-0.9)
    else:
        b = rand.uniform(0.9,1.1)
    x_init[i] = x_init[i] * b * sign
    x_candidate = x_init
    return x_candidate

def action_calc(N, func):
    a = 1
    mu = 1
    m = 1
    x = func
    S_e = 0
    for i in range(0,N-1):
        S_e += (1/2*m*((x[i+1]-x[i])/a)**2)+1/2*mu**2 * x[i]**2
    return S_e


def metro(N, func_init, func_cand):
    S_e_init = action_calc(N, func_init)
    S_e_cand = action_calc(N, func_cand)
    delta_S_e = S_e_init - S_e_cand 
    if np.exp(-delta_S_e) >= 1:
        return(func_cand) #accept
    else:
        if np.exp(-delta_S_e) >= rand.uniform(0,1):
            return(func_cand) #accept
        else:
            return(func_init) #reject


def metro_repeats(N,runs,func_init):
    func_array = np.zeros((runs,N))
    uncertainty=np.zeros(runs)
    func_array[0,:] = func_init
    for i in range(1, runs, 1):
        func_array[i,:] = metro(N,func_array[i-1,:],x_candidate_func(N,func_array[i-1,:]))
    for y in range(0,runs,1):
        uncertainty[y] = stat.stdev(func_array[y])
    
    #plt.errorbar(range(func_array[:,0]), func_array[0,:], yerr=uncertainty)
    return func_array, uncertainty

def ground_state(N,runs,func_init):
    mu = 1
    test = np.zeros(runs)
    metro_output = metro_repeats(N,runs,func_init)
    metro = metro_output[0]
    uncer = metro_output[1]
    for i in range(0,runs):
        test[i] = np.mean((metro[i,N-1])**2)
    E_0 = np.mean(test)*mu**2
    uncertainty = np.mean(uncer)
    E_0_uncertainty = np.sqrt((2*uncertainty/(E_0))**2)*E_0
    #E_0 = (np.mean((metro_repeats(N,runs,func_init[:,N-1]))**2))*mu**2
    return E_0, E_0_uncertainty,Theory_ground_state(1,1,1)#, metro 

def ground_state_prob_dis(N,runs,g_s_runs,func_init):
    test = np.zeros(g_s_runs)
    uncertainty = np.zeros(g_s_runs)
    for i in range(g_s_runs):
        ground = ground_state(N, runs, func_init)
        ground_st = ground[0]
        uncertainty[i] = ground[1]
        test[i] = ground_st
    test_mean = np.mean(test)
    uncertainty_mean = np.mean(uncertainty)
    plt.errorbar(test,range(len(test)),xerr=uncertainty)
    return test,uncertainty, test_mean, uncertainty_mean #histogram of test matrix shows distribution more accurately than the average

'''
def Z(N, func):
    Z = 0
    for i in range(0,N):
        Z += np.exp(-action_calc(N,func))
    return Z
'''

def excited_state(N,runs,g_s_runs,func_init):
    E_0 = ground_state_prob_dis(N, runs, g_s_runs, func_init)
    delta_S_e = action_calc(N, E_0[0]) - action_calc(N, x_initial_func(N))
    delta_E = -np.log(np.exp(delta_S_e))
    E_1 = E_0[3] + delta_E
    uncertainty = 1/(N*(N-1))*sum(E_0[1])
    return E_1, uncertainty, n_excited_state(1)

#theoretical calculation of the ground state
def Theory_ground_state(mu, m, a):
    return 1/2 * (mu/np.sqrt(m) * (1 - (mu**2 * a**2)/(8*m)))

#theoretical calculation of excited states
def n_excited_state(n): #run time increases exponentially with n, don't run for more than n = 10 (this takes about 20s)
    mu = 1
    m = 1
    a = 1
    E_n = 0
    w = (mu/np.sqrt(m) * (1 - (mu**2 * a**2)/(24*m)))
    if n == 0:
        E_n = Theory_ground_state(mu, m, a)
        return E_n
    else:
        for i in range(0,n):
            E_n += n_excited_state(n-1) + w
        return E_n
    

def anharmonic_action_calc(N, func):
    a = 1
    mu = 1
    m = 1
    lmbda = 1
    x = func
    S_e = 0
    for i in range(0,N-1):
        S_e += (1/2*m*((x[i+1]-x[i])/a)**2)+1/2*mu**2 * x[i]**2 + lmbda*x[i]**2
    return S_e

def anharmonic_metro(N, func_init, func_cand):
    S_e_init = anharmonic_action_calc(N, func_init)
    S_e_cand = anharmonic_action_calc(N, func_cand)
    delta_S_e = S_e_init - S_e_cand 
    if np.exp(-delta_S_e) >= 1:
        return(func_cand) #accept
    else:
        if np.exp(-delta_S_e) >= rand.uniform(0,1):
            return(func_cand) #accept
        else:
            return(func_init) #reject

def anharmonic_metro_repeats(N,runs,func_init):
    func_array = np.zeros((runs,N))
    uncertainty=np.zeros(runs)
    func_array[0,:] = func_init
    for i in range(1, runs, 1):
        func_array[i,:] = anharmonic_metro(N,func_array[i-1,:],x_candidate_func(N,func_array[i-1,:]))
    for y in range(0,runs,1):
        uncertainty[y] = stat.stdev(func_array[y])
    
    #plt.errorbar(range(func_array[:,0]), func_array[0,:], yerr=uncertainty)
    return func_array, uncertainty

def anharmonic_ground_state(N,runs,func_init):
    mu = 1
    lmbda = 1
    test = np.zeros(runs)
    test2 = np.zeros(runs)
    metro_output = anharmonic_metro_repeats(N,runs,func_init)
    metro = metro_output[0]
    uncer = metro_output[1]
    for i in range(0,runs):
        test[i] = np.mean((metro[i,N-1])**2)
    for y in range(0,runs):
        test2[y] = np.mean((metro[y,N-1])**4)
    E_0 = np.mean(test)*mu**2 + 3*lmbda*np.mean(test2)
    uncertainty = np.mean(uncer)
    E_0_uncertainty = np.sqrt((4*uncertainty/(E_0))**2)*E_0 
    #E_0 = (np.mean((metro_repeats(N,runs,func_init[:,N-1]))**2))*mu**2
    return E_0, E_0_uncertainty#,Theory_ground_state(1,1,1), metro 

def anharmonic_ground_state_prob_dis(N,runs,g_s_runs,func_init):
    test = np.zeros(g_s_runs)
    uncertainty = np.zeros(g_s_runs)
    for i in range(g_s_runs):
        ground = anharmonic_ground_state(N, runs, func_init)
        uncertainty[i] = ground[1]
        test[i] = ground[0]
    test_mean = np.mean(test)
    uncertainty_mean = np.mean(uncertainty)
    plt.errorbar(test,range(len(test)),xerr=uncertainty)
    return test,uncertainty, test_mean, uncertainty_mean #histogram of test matrix shows distribution more accurately than the average


def anharmonic_excited_state(N,runs,g_s_runs,func_init):
    E_0 = ground_state_prob_dis(N, runs, g_s_runs, func_init)
    delta_S_e = action_calc(N, E_0[0]) - action_calc(N, x_initial_func(N))
    delta_E = -np.log(np.exp(delta_S_e))
    E_1 = E_0[3] + delta_E
    uncertainty = 1/(N*(N-1))*sum(E_0[1])
    return E_1, uncertainty, n_excited_state(1)