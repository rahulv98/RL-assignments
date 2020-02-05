import numpy as np
from tqdm import tqdm

def median_elimination(e, delta, arms_init, runs, trueQ):
    
    R_avg = []
    opt_count = []
    
    for run_id in range(runs):
        
        t = 0
        el = e/4
        dl = delta/2
        arms = arms_init

        estQ = np.zeros(arms)
        picks_count = np.zeros(arms)

        Q = trueQ[run_id]
        opt_val = np.max(Q)

        while(np.size(estQ) > 1):
            
            l = (np.log(3 / dl)) / ((el / 2) ** 2)
            
            for i in tqdm(range(int(l))):
                for arm in range(arms):
            
                    picks_count[arm] += 1

                    if opt_val == Q[arm]:
                        if len(opt_count) > t:
                            opt_count[t] += 1
                        else:
                            opt_count.append(1)
                    
                    elif len(opt_count) <= t:
                        opt_count.append(0)

                    reward = np.random.normal(Q[arm], 1)
                    
                    if len(R_avg) > t:
                        R_avg[t] += (reward - R_avg[t])/(run_id + 1)
                    else:
                        R_avg.append(reward)

                    estQ[arm] += (reward - estQ[arm])/(picks_count[arm])
                    t += 1
        
            ml = np.median(estQ)
            arms_toremove = []

            for arm, qest_arm in enumerate(estQ):

                if qest_arm < ml:
                    arms_toremove.append(arm)

            estQ = np.delete(estQ, arms_toremove)
            Q = np.delete(Q, arms_toremove)
            picks_count = np.delete(picks_count, arms_toremove)
            arms -= np.size(arms_toremove)
                    
            
            el *= 0.75
            dl *= 0.5

    return R_avg, (np.asarray(opt_count) * 100 / runs)

