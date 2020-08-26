import gym
import numpy as np
from tqdm import tqdm

from env.four_rooms import FourRooms

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class SMDPQ:

    DISCOUNT_FACTOR = 0.9
    LEARNING_RATE = 0.25
    EPSILON = 0.1
    MAX_STEPS = 100000
    NO_OF_EPISODES = 1000
    NO_OF_RUNS = 50

    def __init__(self, env, intra_option=False, goal='G1', fixed_start=False):
        self.env = gym.make(env)
        self.goal = goal
        self.env.set_goal(self.goal)
        self.Q_avg = np.zeros((self.env.observation_space.n, self.env.action_space.n + 2)) #2 possible doorway options at each state
        self.total_steps = np.zeros(self.NO_OF_EPISODES)
        self.intra_option = intra_option
        self.fixed_start = fixed_start

    def train(self):

        for run in tqdm(range(self.NO_OF_RUNS)):
            self.q = np.zeros((self.env.observation_space.n, self.env.action_space.n + 2)) #2 possible doorway options at each state
            
            for episode in range(self.NO_OF_EPISODES):
                state = self.env.reset(self.fixed_start)
            
                #Initialising episode variables
                is_teriminal = False
                episode_steps = 0

                while not is_teriminal and episode_steps < self.MAX_STEPS:
                    #action selection using e-greedy algorithm
                    if np.random.uniform(0, 1) < self.EPSILON:
                        option = np.random.randint(6)
                    else:
                        # option = np.argmax(self.q[state])
                        option = np.random.choice(np.flatnonzero(self.q[state] == self.q[state].max()))
                    # print("state : ", state, "option: ", option)

                    next_state, reward, is_teriminal, steps = self.perform_option(state, option)

                    episode_steps += steps

                    if not self.intra_option:
                        td_error = reward + (self.DISCOUNT_FACTOR ** steps) * np.max(self.q[next_state]) - self.q[state, option]
                        self.q[state, option] += self.LEARNING_RATE * td_error

                    state = next_state
                
                self.total_steps[episode] += episode_steps

            self.Q_avg += self.q
        self.total_steps /= self.NO_OF_RUNS
        self.Q_avg /= self.NO_OF_RUNS

        if self.intra_option:
            if self.fixed_start:
                np.save("results/intra_option_" + self.goal + "_fixed_start.npy", self.Q_avg)
                np.save("results/intra_option_steps_" + self.goal + "_fixed_start.npy", self.total_steps)
            else:
                np.save("results/intra_option_" + self.goal + ".npy", self.Q_avg)
                np.save("results/intra_option_steps_" + self.goal + ".npy", self.total_steps)

        else:
            if self.fixed_start:
                np.save("results/SMDPQ_" + self.goal + "_fixed_start.npy", self.Q_avg)
                np.save("results/SMDPQ_steps_" + self.goal + "_fixed_start.npy", self.total_steps)
            else:
                np.save("results/SMDPQ_" + self.goal + ".npy", self.Q_avg)
                np.save("results/SMDPQ_steps_" + self.goal + ".npy", self.total_steps)

    def perform_option(self, state, option, render=False):
        option_reward = 0
        if option < 4:
            steps = 1
            next_state, reward, done, _ = self.env.step(option)
            option_reward += reward
            if self.intra_option:
                td_error = reward + (self.DISCOUNT_FACTOR * np.max(self.q[next_state])) - self.q[state, option]
                self.q[state, option] += self.LEARNING_RATE * td_error
            
            if render:
                self.env.render()

        else:
            exit_hallway = self.get_exit_hallway(state, option)
            rewards = []
            done = False
            curr_room, _ = self.env.decode(state)
            while (state != exit_hallway and 
                   (exit_hallway in self.env.room_connections[curr_room]
                    or self.env.in_hallway_index(state)) and 
                  not done):

                action = self.get_primitive_action(state, exit_hallway)

                next_state, reward, done, _ = self.env.step(action)
                rewards.append(reward)
                next_room, _ = self.env.decode(next_state)

                if render:
                    self.env.render()

                if self.intra_option:
                    #If option continues use same option Q value for update
                    if (next_state != exit_hallway and 
                        (exit_hallway in self.env.room_connections[next_room]
                            or self.env.in_hallway_index(next_state)) and 
                        not done):
                        td_error = reward + (self.DISCOUNT_FACTOR * self.q[next_state, option]) - self.q[state, option]
                        self.q[state, option] += self.LEARNING_RATE * td_error
                        
                    
                    #else use max Q value for update
                    else:
                        td_error = reward + (self.DISCOUNT_FACTOR * np.max(self.q[next_state])) - self.q[state, option]
                        self.q[state, option] += self.LEARNING_RATE * td_error

                state = next_state
                curr_room = next_room

            for rew in reversed(rewards):
                option_reward += rew
                option_reward *= self.DISCOUNT_FACTOR

            steps = len(rewards)

        return next_state, option_reward, done, steps

    def get_exit_hallway(self, state, option):
        
        if(self.env.in_hallway_index(state)):
            hallway_index = self.env.offsets.index(state + 1)
            if option == 4: #clockwise direction hallway
                exit_hallway_index = (hallway_index + 1) % 4
                return self.env.offsets[exit_hallway_index] - 1

            elif option == 5: #anticlock wise direction hallway
                exit_hallway_index = (hallway_index - 1)
                if exit_hallway_index == 0:
                    exit_hallway_index = 4
                return self.env.offsets[exit_hallway_index] - 1

        else:
            room, coord_in_room = self.env.decode(state)
            
            if option == 4: #clockwise direction hallway
                return self.env.offsets[room + 1] - 1
            
            elif option == 5: #anticlock wise direction hallway
                return (self.env.offsets[room] - 1) % (self.env.n_states - 1)


    def get_primitive_action(self, state, exit_hallway):
        #Option policy

        state_coords_global = self.env.get_global_coord(state)
        hallway_coords_global = self.env.get_global_coord(exit_hallway)

        if (self.env.in_hallway_index(state)):  
            #Currently at a hallway
            curr_hallway_index = self.env.offsets.index(state + 1)
            exit_hallway_index = self.env.offsets.index(exit_hallway + 1)

            if curr_hallway_index in [1, 3]:
                if exit_hallway_index == 2:
                    action = RIGHT
                else:
                    action = LEFT
            
            else:
                if exit_hallway_index == 1:
                    action = UP
                else:
                    action = DOWN

        else:
            row_s, col_s = state_coords_global
            row_h, col_h = hallway_coords_global 

            if abs(row_s - row_h) > abs(col_s - col_h):
                if row_s > row_h:
                    action = UP
                else:
                    action = DOWN
            
            elif abs(row_s - row_h) < abs(col_s - col_h):
                if col_s > col_h:
                    action = LEFT
                else:
                    action = RIGHT
            
            else:
                if np.random.uniform(0, 1) < 0.5:
                    if row_s > row_h:
                        action = UP
                    else:
                        action = DOWN
                else:
                    if col_s > col_h:
                        action = LEFT
                    else:
                        action = RIGHT
        
        return action

    def play_model(self):
        
        for run in range(20):
            state = self.env.reset(self.fixed_start)
        
            #Initialising episode variables
            is_teriminal = False
            total_steps = 0

            while not is_teriminal:
                option = np.argmax(self.Q_avg[state])

                next_state, reward, is_teriminal, steps = self.perform_option(state, option, render=True)
                total_steps += steps
                state = next_state
                
            print("EPISODE: ", run, "LENGTH : ", total_steps)

 
if __name__ == "__main__":
    for goal in ["G1", "G2"]:
        for fixed_start in [False, True]:
            for intra_option in [False, True]:
                smdpq = SMDPQ('fourrooms-v0', intra_option=intra_option, goal=goal, fixed_start=fixed_start)
                smdpq.train()
                # smdpq.play_model()