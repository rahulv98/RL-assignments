 
from gym.envs.registration import register

register(
    id='fourrooms-v0',
    entry_point='env.four_rooms:FourRooms',
    # max_episode_steps = 100000,
)