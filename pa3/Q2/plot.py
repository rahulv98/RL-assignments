import numpy as np
import matplotlib.pyplot as plt

import os
import experiments

def plot_learning_curve(name, runs):
    avg_rew = np.zeros(500)
    plt.clf()
    for i in range(runs):
        rews = np.load("results/learning_curve_"+ name +"/run_" + str(i) + ".npy")

        mov_avg_rew = [np.mean(rews[:i][-100:]) for i in range(len(rews))]
        
        plt.plot(mov_avg_rew, linewidth=0.75, linestyle="-.", label="run-" + str(i), alpha=0.7)
        avg_rew += mov_avg_rew

    plt.plot(avg_rew/runs, c="black", linewidth=1.5, label="average over runs")
    
    plt.plot(195 * np.ones(500), c='black', linewidth=1, linestyle='--', label="195 mark")
    plt.legend(loc=0)
    plt.title(name + " Learning Curve for CartPole-v0 over \n"+ str(runs) +" independent runs")
    plt.xlabel("Episodes")
    plt.ylabel("Reward(Episode Length)")
    plt.savefig("plots/learning_curve_"+ name + ".png")

def plot_test_curve(runs):
    avg_rew = np.zeros(20)
    plt.clf()
    for i in range(runs):
        rews = np.load("results/learning_curve_best/test_run_" + str(i) + ".npy")

        plt.plot(rews, linewidth=1, label="run-" + str(i), alpha=0.75)
        avg_rew += rews

    plt.plot(avg_rew/runs, linewidth=2.5, label="average over runs")
    
    plt.legend(loc=4)
    plt.title("Test Performace for CartPole-v0 over " + str(runs) + " independent runs")
    plt.xlabel("Episodes")
    plt.ylabel("Reward(Episode Length)")
    plt.savefig("plots/test_plot.png")



def plot_comparision(exp_name, exps):
    plt.clf()
    for i, exp in enumerate(exps):
        rews = np.load("results/exp_" + exp_name + "/run_" + str(i)+ ".npy")

        mov_avg_rew = [np.mean(rews[:i][-100:]) for i in range(len(rews))]
        
        plt.plot(mov_avg_rew, label= exp_name +  "-" + str(exp))
    plt.legend(loc=0)
    plt.title("Learning Curve for CartPole-v0 with various " + exp_name + "s")
    plt.xlabel("Episodes")
    plt.ylabel("Reward(Episode Length)")
    plt.savefig("plots/"+ exp_name + ".png")

if __name__ == "__main__":
    plot_learning_curve("best", 4)
    plot_learning_curve("without_replay", 4)
    plot_learning_curve("without_target", 4)
    plot_learning_curve("without_replay_and_target", 4)
    plot_test_curve(4)
    plot_comparision("mini_batch_size", experiments.MINI_BATCH_SIZES)
    plot_comparision("learning_rate", experiments.LEARNING_RATES)
    plot_comparision("start_epsilion", experiments.EPSILON_STARTS)
    plot_comparision("replay_memory_size", experiments.REPLAY_MEMORY_SIZES)