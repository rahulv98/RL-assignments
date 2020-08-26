import numpy as np
import matplotlib.pyplot as plt

import os
import gym
from env.four_rooms import FourRooms

option_names = ["Up", "Right", "Down", "Left", "Clockwise", "Anti-Clockwise"]

class Visualizer:

    def __init__(self, goal="G1", fixed_start=False, intra_option=False):
        self.env = gym.make('fourrooms-v0')
        self.goal = goal
        self.env.set_goal(goal)
        self.fixed_start = fixed_start
        self.intra_option = intra_option
        self.img = np.loadtxt(os.path.abspath(os.getcwd()) + '/env/room.txt', delimiter=',')		
        self.fig = plt.figure()
        self.img[self.env.goal_global[0], self.env.goal_global[1]] = 0.3
        
        if self.fixed_start:
            self.res_file_ext = "_fixed_start.npy"
            self.plot_name_ext = "fixed_start_"
            self.plot_title = " Fixed start state in room-4 "
        else:
            self.res_file_ext = ".npy"
            self.plot_name_ext = ""
            self.plot_title = " Random start state in room-1 "

        if self.intra_option:
            self.algo = "intra_option"
        else:
            self.algo = "SMDPQ"
        

    def value_function(self):
        plt.clf()
        plt.pcolor(self.img[::-1], cmap='terrain', edgecolor='k')
        fig = plt.gcf()
        ax = fig.gca()

        self.V = np.max(np.load("results/" +self.algo + "_" + self.goal + self.res_file_ext), axis = 1)

        for i in range(len(self.V)-1):
            value = self.V[i]
            centre = self.env.get_global_coord(i)
            circle = plt.Circle((centre[1] + 0.5, 12 - centre[0] + 0.5), value/2, color='k')
            ax.add_artist(circle)
        plt.title(self.algo + " Learning : Value Function for \n Goal " + self.goal + self.plot_title)
        plt.savefig("plots/" + self.algo + "_" + self.plot_name_ext + self.goal + ".png")

    def Q_values(self, option):
        plt.clf()
        plt.pcolor(self.img[::-1], cmap='terrain', edgecolor='k')
        fig = plt.gcf()
        ax = fig.gca()
    
        self.Q = np.load("results/" +self.algo + "_" + self.goal + self.res_file_ext)[:, option]
        for i in range(len(self.Q)-1):
            value = self.Q[i]
            centre = self.env.get_global_coord(i)
            circle = plt.Circle((centre[1] + 0.5, 12 - centre[0] + 0.5), value/2, color='k')
            ax.add_artist(circle)
        plt.title(self.algo + " Learning : Q-values for option : " + option_names[option] + "\n for Goal " + self.goal + self.plot_title)
        plt.savefig("plots/" +self.algo + "_" + self.plot_name_ext + self.goal + "_option_" + option_names[option] + ".png")

    def Learnt_policy(self):
        plt.clf()
        plt.pcolor(self.img[::-1], cmap='terrain', edgecolor='k')
        fig = plt.gcf()
        ax = fig.gca()

        self.Q = np.load("results/" +self.algo + "_" + self.goal + self.res_file_ext)
        self.policy = np.argmax(self.Q, axis=1)

        for i in range(len(self.policy)-1):
            action = self.policy[i]
            if action == 0:
                if np.all(self.Q[i] == 0):
                    action = "x"
            centre = self.env.get_global_coord(i)
            plt.text(centre[1] + 0.4, 12 - centre[0] + 0.4, str(action))
        plt.title(self.algo + " Learning : Learnt Policy for \n Goal " + self.goal + self.plot_title)
        plt.savefig("plots/" + self.algo + "_policy_" + self.plot_name_ext + self.goal + ".png")

def plot_steps_all(goal="G1"):
    steps = np.load("results/SMDPQ_steps_" + goal + ".npy")
    steps_io = np.load("results/intra_option_steps_" + goal + ".npy")
    steps_fixed = np.load("results/SMDPQ_steps_" + goal + "_fixed_start.npy")
    steps_io_fixed = np.load("results/intra_option_steps_" + goal + "_fixed_start.npy")

    plt.clf()
    plt.plot(steps, label="SMDPQ room-1 random start", alpha=0.7)
    plt.plot(steps_io, label="Intra Option room-1 random start", alpha=0.7)
    plt.plot(steps_fixed, label="SMDPQ room-4 fixed start", alpha=0.7)
    plt.plot(steps_io_fixed, label="Intra Option room-4 fixed start", alpha=0.7)
    plt.xscale("log")
    plt.xlabel("Episodes in log-10 scale")
    plt.ylabel("Average no. of steps")
    plt.title("Average steps plot for goal " + goal)
    plt.legend()
    plt.savefig("plots/steps_" + goal + ".png")
    
if __name__ == "__main__":
    for goal in ["G1", "G2"]:
        
        plot_steps_all(goal)

        for fixed_start in [False, True]:
            for intra_option in [False, True]:
                visualizer = Visualizer(goal=goal, fixed_start=fixed_start, intra_option=intra_option)
                visualizer.value_function()
                visualizer.Learnt_policy()
                if not intra_option:
                    visualizer.Q_values(0)
                    visualizer.Q_values(1)
                    visualizer.Q_values(2)
                    visualizer.Q_values(3)
                    visualizer.Q_values(4)
                    visualizer.Q_values(5)
                
    
    
    
