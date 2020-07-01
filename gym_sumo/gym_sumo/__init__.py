from gym.envs.registration import register

register(
    id='gym_sumo-v0',
    entry_point='gym_sumo.envs:SumoEnv',
)
