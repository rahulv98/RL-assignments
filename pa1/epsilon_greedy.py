import numpy as np
from tqdm import tqdm

def epsilon_greedy(epsilon, arms, time_steps, runs, trueQ):
    
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
        
        trails = np.random.uniform(size=runs)

        for run_id, trail in enumerate(trails):

            if trail < epsilon:
                arm = np.random.randint(arms)
                picks_count[run_id][arm] += 1   

            else: #Greedy action
                arm = np.argmax(estQ[run_id])
                picks_count[run_id][arm] += 1
            
            if opt_arm[run_id] == arm:
                opt_count[t] += 1

            reward = rewards[run_id][t] + trueQ[run_id][arm]

            #Updating R_avg and estQ with incremental method
            R_avg[t] += (reward - R_avg[t])/(run_id + 1)

            estQ[run_id][arm] += (reward - estQ[run_id][arm])/(picks_count[run_id][arm]) 

    return R_avg, ((opt_count * 100) / runs)
