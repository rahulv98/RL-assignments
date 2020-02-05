from matplotlib import pyplot as plt 
from scipy.signal import savgol_filter

from testbed import *
from median_elimination import median_elimination


def plot_median_elimination(epsilons, deltas, arms, runs, trueQ):

    fig1 = plt.figure(1).add_subplot(ylabel="Average Reward", xlabel="Time Steps")
    fig2 = plt.figure(2).add_subplot(ylabel="%Optimal action", xlabel="Time Steps")
    colors = ["green", "red", "blue"]

    for i, epsilon in enumerate(epsilons):
        reward, opt_percent = median_elimination(epsilon, deltas[i], arms, runs, trueQ)

        fig1.plot(range(len(reward)), savgol_filter(reward, 1001, 3), colors[i], label = "epsilon = " + str(epsilon) + "delta = " +str(deltas[i]))
        fig2.plot(range(len(opt_percent)), savgol_filter(opt_percent, 1001, 3), colors[i], label = "epsilon = " + str(epsilon) + "delta = " +str(deltas[i]))


    fig1.legend()
    fig2.legend()
    plt.show()

epsilons = [0.1, 0.2, 0.5]
deltas = [0.1, 0.2, 0.5]

plot_median_elimination(epsilons, deltas, arms, 1, trueQ)