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
lamdas = [0, 0.3, 0.5, 0.9, 0.99, 1]

max_steps = 10000
def sarsa_lamda(GAMMA, NO_OF_RUNS, NO_OF_EPISODES, EPSILON, ALPHA, MODE, lamda, max_steps):
    env = gym.make('puddleworld-v0')


    #setting mode
    env.set_mode(mode)

    #Initialising average variables
    avg_reward = np.zeros((NO_OF_EPISODES, len(lamdas)))
    avg_steps = np.zeros((NO_OF_EPISODES, len(lamdas)))
    q_table_avg = np.zeros((env.observation_space[0].n, env.observation_space[1].n , env.action_space.n))

    for run in tqdm(range(NO_OF_RUNS)):
        
        q_table = np.random.rand(env.observation_space[0].n, env.observation_space[1].n , env.action_space.n)

        for episode in tqdm(range(NO_OF_EPISODES)):
            state = env.reset()

            #Initialising episode variables
            total_reward = 0
            steps = 0
            is_terminal = False
            e_k = np.zeros((env.observation_space[0].n, env.observation_space[1].n , env.action_space.n))

            #action selection using e-greedy algorithm
            if np.random.uniform(0, 1) < EPSILON:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])


            while (not is_terminal) and steps < max_steps:
                steps += 1
                
                #take action
                next_state, reward, is_terminal, _ = env.step(action)

                total_reward += reward

                #updating eligibility
                e_k = e_k * lamda * GAMMA
                e_k[state[0], state[1], action] += 1

                #next action with e-greedy
                if np.random.uniform(0, 1) < EPSILON:
                    next_action = env.action_space.sample()
                else:
                    next_action = np.argmax(q_table[next_state])

                #Update Q using TD(0) 
                td_error = reward + GAMMA * q_table[next_state[0], next_state[1], next_action] - q_table[state[0], state[1], action]
                q_table[state[0], state[1], action] += (td_error) * ALPHA * e_k[state[0], state[1], action]

                #moving to next state
                state = next_state
                action = next_action

            avg_steps[episode] += steps 
            avg_reward[episode] += total_reward

        q_table_avg += q_table
            

        q_table_avg /= NO_OF_RUNS

    return avg_steps, avg_reward, q_table          
        
        
cols = ['r', 'k', 'b', 'y', 'g', 'm']

fig1=plt.figure().add_subplot(111)
fig2=plt.figure().add_subplot(111)

mode = "A"

for i, lamda in enumerate(lamdas):
    s, r, q = sarsa_lamda(GAMMA, NO_OF_RUNS, NO_OF_EPISODES, EPSILON, ALPHA, mode, lamda, max_steps)
    fig1.plot(s[:]/NO_OF_RUNS, c = cols[i], label = "$\lambda$ = " + str(lamda), alpha=0.2)
    fig2.plot(r[:]/NO_OF_RUNS, c = cols[i], label = "$\lambda$ = " + str(lamda), alpha=0.2)
fig1.set_xlabel("Episodes", fontsize = 18)
fig1.set_ylabel("Average no of steps", fontsize = 18)
fig1.set_title("SARSA($\lambda$) - Average no of steps at each episode - experiment " + mode)
fig2.set_xlabel("Episodes", fontsize = 18)
fig2.set_ylabel("Average reward", fontsize = 18)
fig2.set_title("SARSA($\lambda$) - Average reward at each episode - experiment " + mode)
plt.show()
        