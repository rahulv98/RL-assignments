import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


GRID_HEIGHT = 12
GRID_WIDTH = 12


class puddleEnv(gym.Env):

    def __init__(self):
        self.height = GRID_HEIGHT
        self.width = GRID_WIDTH
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((spaces.Discrete(self.height), spaces.Discrete(self.width)))

        self.seed()


        self.moves = {
            0 : (-1, 0), #up
            1 : (0, 1),  #right
            2 : (1, 0),  #down
            3 : (0, -1)  #left
        }

        self.end_pos = {
            'A' : (0, 11),
            'B' : (2, 9),
            'C' : (6, 7)
        }
        #default mode
        self.mode = 'A'
        self.wind = True


        # begin in start state
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_mode(self, mode):
        if mode == 'C':
            self.wind = False
        else:
            self.wind = True
        self.mode = mode

    def step(self, action):
        
        if np.random.uniform() < (0.13334):  # (0.1 + 0.1/3) pick any action with equal probability 
            
            action = np.random.randint(self.action_space.n)
        
        x, y = self.moves[action]
        self.S = self.S[0] + x, self.S[1] + y

        #westerly wind
        if np.random.uniform() < 0.5 and self.wind:
            self.S = self.S[0], self.S[1] + 1

        #bounding to grid
        self.S = max(0, self.S[0]), max(0, self.S[1])
        self.S = (min(self.S[0], self.height - 1),
                  min(self.S[1], self.width - 1))

        reward = 0

        if self.S in [(4, 5), (4, 6), (5, 5), (6, 5)]:
            reward = -3
        
        elif self.S in [(3, 4), (3, 5), (3, 6), (3, 7), 
                        (4,4), (4, 7), (5, 4), (5, 6), 
                        (5, 7), (6, 4), (6, 6), (7, 4), (7, 5), (7, 6)]:
            reward = -2
        
        elif self.S in [(2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), 
                        (3, 3), (3, 8), (4, 3), (4, 8), (5, 3), (5, 8), 
                        (6, 3), (6, 7), (6, 8), (7, 3), (7, 7), (8, 3), 
                        (8, 4), (8, 5), (8, 6), (8, 7)]:
            reward = -1
        

        if self.S == self.end_pos[self.mode]:
            return self.S, 10, True, {}
        else:
            return self.S, reward, False, {}

    def reset(self):
        start_pos = [(5, 0), (6, 0), (10, 0), (11, 0)]
        self.S = start_pos[np.random.randint(len(start_pos))]
        return self.S
