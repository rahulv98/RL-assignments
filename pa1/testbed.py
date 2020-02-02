import numpy as np
from matplotlib import pyplot as plt

num_plays = 1000
num_runs = 2000
num_arms = 10
epsilon = 0.1


Ravg = np.zeros(num_plays)

for i in range(1, num_runs + 1):
    qest = np.zeros(num_arms)
    play_count = np.zeros(num_arms)
    trueq = np.random.normal(0, 1, num_arms)


    for t in range(0, num_plays):

        if(np.random.uniform() < epsilon):
            is_greedy = False
        else:
            is_greedy = True

        if is_greedy:
            opt_idx = np.argmax(qest)
            play_count[opt_idx] += 1
        else:
            opt_idx = np.random.randint(num_arms)
            play_count[opt_idx] += 1

        r = np.random.normal(trueq[opt_idx], 1)

        Ravg[t] += (r - Ravg[t])/i

        qest[opt_idx] += (r - qest[opt_idx])/play_count[opt_idx]


plt.plot(np.arange(num_plays), Ravg)
plt.show()