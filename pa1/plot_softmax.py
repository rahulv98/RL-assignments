from matplotlib import pyplot as plt

from testbed import *
from softmax import softmax 

def plot_softmax(temperatures, arms, time_steps, runs, trueQ):

    fig1 = plt.figure(1).add_subplot(ylabel="Average Reward", xlabel="Time Steps")
    fig2 = plt.figure(2).add_subplot(ylabel="%Optimal action", xlabel="Time Steps")
    colors = ["green", "red", "blue", "black"]

    for i, temperature in enumerate(temperatures):
        reward, opt_percent = softmax(temperature, arms, time_steps, runs, trueQ)

        fig1.plot(range(time_steps), reward, colors[i], label = "temperature = " + str(temperature))
        fig2.plot(range(time_steps), opt_percent, colors[i], label = "temperature = " + str(temperature))


    fig1.legend()
    fig2.legend()
    plt.show()

temperatures = [0.1]

plot_softmax(temperatures, arms_large, time_steps_large, runs, trueQ_large)