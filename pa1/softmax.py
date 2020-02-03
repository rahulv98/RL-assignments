import numpy as np

def softmax(temperature, arms, time_steps, runs, trueQ):

    
    opt_arm = np.argmax(trueQ, axis = 1)

    estQ = np.zeros((runs, arms))
    picks_count = np.zeros((runs, arms))

    R_avg = np.zeros(time_steps)
    opt_count = np.zeros(time_steps)

    for t in range(time_steps):
        
        numerator = np.exp(estQ / temperature)

        for run_id in range(runs):

            probab = numerator[run_id] / np.sum(numerator[run_id])
            
            arm = np.random.choice(arms, p=probab)
            picks_count[run_id][arm] += 1

            if opt_arm[run_id] == arm:
                opt_count[t] += 1

            reward = np.random.normal(trueQ[run_id][arm], 1)

            R_avg[t] += (reward - R_avg[t])/(run_id + 1)

            estQ[run_id][arm] += (reward - estQ[run_id][arm])/(picks_count[run_id][arm])

    
    return R_avg, ((opt_count * 100) / runs)

