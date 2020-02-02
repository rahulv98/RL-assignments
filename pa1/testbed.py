import numpy as np

arms = 10
runs = 2000
time_steps = 1000

trueQ = np.random.normal(0, 1, (runs, arms))