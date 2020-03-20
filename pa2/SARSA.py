import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from penv.puddleworld import puddleEnv

from plot_policy import plot_policy

GAMMA = 0.9
NO_OF_RUNS = 50
NO_OF_EPISODES = 1000
EPSILON = 0.1
ALPHA = 0.1
MODES = ['A', 'B', 'C']
max_steps = 1000
env = gym.make('puddleworld-v0')

#Initialising average variables
for id, mode in enumerate(MODES):
    #setting mode
    env.set_mode(mode)

    avg_reward = np.zeros(NO_OF_EPISODES)
    avg_steps = np.zeros(NO_OF_EPISODES)
    q_table_avg = np.zeros((env.observation_space[0].n, env.observation_space[1].n , env.action_space.n))

    for run in tqdm(range(NO_OF_RUNS)):
        
        q_table = np.random.rand(env.observation_space[0].n, env.observation_space[1].n , env.action_space.n)

        for episode in tqdm(range(NO_OF_EPISODES)):
            state = env.reset()

            #Initialising episode variables
            total_reward = 0
            steps = 0
            is_terminal = False

            #action selection using e-greedy algorithm
            if np.random.uniform(0, 1) < EPSILON:
                action = env.action_space.sample()
            else:
                # action = np.random.choice(np.flatnonzero(q_table[state] == q_table[state].max()))
                action = np.argmax(q_table[state])

            while (not is_terminal) and steps < max_steps:
                #take action
                next_state, reward, is_terminal, _ = env.step(action)

                #next action with e-greedy
                if np.random.uniform(0, 1) < EPSILON:
                    next_action = env.action_space.sample()
                else:
                    # next_action = np.random.choice(np.flatnonzero(q_table[state] == q_table[state].max()))
                    next_action  = np.argmax(q_table[next_state])

                #Update Q using TD(0) 
                td_error = (reward + GAMMA * q_table[next_state[0], next_state[1], next_action]) - q_table[state[0], state[1], action]
                q_table[state[0], state[1], action] += (td_error * ALPHA) 

                #moving to next state and next action
                state = next_state
                action = next_action
                steps += 1
                total_reward += reward

            avg_steps[episode] += steps 
            avg_reward[episode] += total_reward
        
        q_table_avg += q_table  

    q_table_avg /= NO_OF_RUNS

    opt_policy = np.argmax(q_table_avg, axis = 2)

    plot_policy(opt_policy, mode, "SARSA ")
        
    plt.plot(avg_steps/NO_OF_RUNS, 'r-')
    plt.xlabel("Episodes", fontsize = 18)
    plt.ylabel("Average no of steps", fontsize = 18)
    plt.title("SARSA - Average no of steps at each episode - experiment " + mode, fontsize = 24)
    plt.show()
    plt.plot(avg_reward/NO_OF_RUNS, 'g-')
    plt.xlabel("Episodes", fontsize = 18)
    plt.ylabel("Average reward", fontsize = 18)
    plt.title("SARSA - Average reward at each episode - experiment " + mode, fontsize = 24)
    plt.show()