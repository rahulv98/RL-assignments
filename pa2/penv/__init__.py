from gym.envs.registration import register

register(
    id='puddleworld-v0',
    entry_point='penv.puddleworld:puddleEnv',
    # max_episode_steps = 5000,
)