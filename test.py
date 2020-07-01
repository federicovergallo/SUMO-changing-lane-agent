import gym
import gym_sumo
from a2c import A2CAgent

env = gym.make('gym_sumo-v0')
fn = ''
agent = A2CAgent(fn=None)

rewards_history = agent.train(env)
