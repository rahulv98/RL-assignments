import numpy as np

arms = 10
runs = 2000
time_steps = 1000
trueQ = np.random.normal(0, 1, (runs, arms))

arms_large = 1000
time_steps_large = 10000
trueQ_large = np.random.normal(0, 1, (runs, arms_large))