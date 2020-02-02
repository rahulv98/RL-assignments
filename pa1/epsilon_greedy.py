import numpy as np
from matplotlib import pyplot as plt

def epsilon_greedy(epsilon, arms, time_steps, runs):

    trueQ = np.random.normal(0, 1, (runs, arms))
    opt_arm = np.argmax(trueQ, axis = 1)

    estQ = np.zeros((runs, arms))
    picks_count = np.zeros((runs, arms))

    R_avg = np.zeros(time_steps)
    opt_count = np.zeros(time_steps)

    for t in range(time_steps):
        trails = np.random.uniform(size=runs)

        for run_id, trail in enumerate(trails):

            if trail < epsilon:
                arm = np.random.randint(arms)
                picks_count[run_id][arm] += 1   

            else:
                arm = np.argmax(estQ[run_id])
                picks_count[run_id][arm] += 1
            
            if opt_arm[run_id] == arm:
                opt_count[t] += 1

            reward = np.random.normal(trueQ[run_id][arm], 1)

            R_avg[t] += (reward - R_avg[t])/(run_id + 1)

            estQ[run_id][arm] += (reward - estQ[run_id][arm])/(picks_count[run_id][arm]) 

    return R_avg, opt_count
