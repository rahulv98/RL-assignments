import numpy as np
from tqdm import tqdm

def softmax(temperature, arms, time_steps, runs, trueQ):

     #Finding the optimal arms in all runs
    opt_arm = np.argmax(trueQ, axis = 1)

    #Initialising estimates to zero for all runs and time steps
    estQ = np.zeros((runs, arms))
    picks_count = np.zeros((runs, arms))  #keeps track of no of picks of each arm

    R_avg = np.zeros(time_steps)
    opt_count = np.zeros(time_steps)
    
    #sampling all the rewards from standard normal and later adding true arm mean
    rewards = np.random.normal(size=(runs, time_steps))

    #Looping over all time steps
    for t in tqdm(range(time_steps)):
        
        numerator = np.exp(estQ / temperature)

        #finding the gibbs distrubtion for all arms
        probab = np.divide(numerator, np.sum(numerator, axis = 1, keepdims=True))
        
        #sampling from gibbs distribution
        arms_picked = get_samples(probab)

        #counting the no of optimal actions for current time step
        opt_count[t] = (opt_arm == arms_picked).sum()

        for run_id in range(runs):
            
            arm = arms_picked[run_id]  
            picks_count[run_id][arm] += 1

            reward = rewards[run_id][t] + trueQ[run_id][arm]

            #Updating R_avg and estQ with incremental method
            R_avg[t] += (reward - R_avg[t])/(run_id + 1)

            estQ[run_id][arm] += (reward - estQ[run_id][arm])/(picks_count[run_id][arm])

    
    return R_avg, ((opt_count * 100) / runs)


def get_samples(prob_matrix):
    #calculating cdf
    cdf = np.cumsum(prob_matrix, axis=1)
    #generating random number from uniform distribution, one per row i.e run
    rnd_num = np.tile(np.random.rand(prob_matrix.shape[0]), (prob_matrix.shape[1], 1)).T

    #finding arm resulting to the random number
    sampled_arms = (cdf < rnd_num).sum(axis=1)
    return sampled_arms
