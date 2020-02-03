from matplotlib import pyplot as plt

from ucb1 import ucb1
from softmax import softmax
from epsilon_greedy import epsilon_greedy
from testbed import *

epsilon = 0.1
temperature  = 0.1

def plot_ucb1(epsilon, temperature, arms, time_steps, runs, trueQ):
    
    fig1 = plt.figure(1).add_subplot(ylabel="Average Reward", xlabel="Time Steps")
    fig2 = plt.figure(2).add_subplot(ylabel="%Optimal action", xlabel="Time Steps")
    
    colors = ["green", "red", "blue"]

    reward_egreedy, opt_percent_egreedy = epsilon_greedy(epsilon, arms, time_steps, runs, trueQ)

    fig1.plot(range(time_steps), reward_egreedy, colors[0], label = "epsilon greedy")
    fig2.plot(range(time_steps), opt_percent_egreedy, colors[0], label = "epsilon greedy")


    reward_softmax, opt_percent_softmax = softmax(temperature, arms, time_steps, runs, trueQ)

    fig1.plot(range(time_steps), reward_softmax, colors[1], label = "softmax")
    fig2.plot(range(time_steps), opt_percent_softmax, colors[1], label = "softmax")

    reward_ucb1, opt_percent_ucb1 = ucb1(arms, time_steps, runs, trueQ)

    fig1.plot(range(time_steps), reward_ucb1, colors[2], label = "ucb1")
    fig2.plot(range(time_steps), opt_percent_ucb1, colors[2], label = "ucb1")


    fig1.legend(loc="lower right")
    fig2.legend(loc="lower right")
    plt.show()

plot_ucb1(epsilon, temperature, arms, time_steps, runs, trueQ)