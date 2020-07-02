import gym
import gym_sumo
from a2c import A2CAgent
from dqn import DQNAgent

env = gym.make('gym_sumo-v0')
fn = ''
#agent = A2CAgent(fn=None)
agent = DQNAgent(fn=None)

agent.train(env)
