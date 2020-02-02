from matplotlib import pyplot as plt

from testbed import *
from epsilon_greedy import epsilon_greedy

def plot_epsilon_greedy(epsilons, arms, time_steps, runs, trueQ):

    fig1 = plt.figure(1).add_subplot(ylabel="Average Reward", xlabel="Time Steps")
    fig2 = plt.figure(2).add_subplot(ylabel="%Optimal action", xlabel="Time Steps")
    colors = ["green", "red", "blue"]

    for i, epsilon in enumerate(epsilons):
        reward, opt_percent = epsilon_greedy(epsilon, arms, time_steps, runs, trueQ)

        fig1.plot(range(time_steps), reward, colors[i], label = "epsilon = " + str(epsilon))
        fig2.plot(range(time_steps), opt_percent, colors[i], label = "epsilon = " + str(epsilon))


    fig1.legend(loc="lower right")
    fig2.legend(loc="lower right")
    plt.show()

epsilons = [0, 0.01, 0.1]

plot_epsilon_greedy(epsilons, arms, time_steps, runs, trueQ)