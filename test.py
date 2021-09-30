import random
from pettingzoo.mpe import simple_adversary_v2

env = simple_adversary_v2.env()

env.reset()
for i, agent in enumerate(env.agent_iter()):
	observation, reward, done, info = env.last()
	action = 0 if not done else None
	env.step(action)
