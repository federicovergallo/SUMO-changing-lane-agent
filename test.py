import gym
import gym_sumo
from a2c import A2CAgent
from dqn import DQNAgent

env = gym.make('gym_sumo-v0')
agent = A2CAgent()
#agent = DQNAgent()

#agent.train(env)
agent.test(env)