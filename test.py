from pettingzoo.utils import average_total_reward
from pettingzoo.mpe import simple_adversary_v2
import random
from tqdm import tqdm
import statistics

env = simple_adversary_v2.env(N=2, max_cycles=100, continuous_actions=False)
env.reset()

agent_idx = {}
for i, agent in enumerate(env.agents):
    agent_idx[agent] = i
    
rewards = []
episodes = 1000
for i in tqdm(range(episodes)):
    env.reset()
    r = 0
    for agent in env.agent_iter():
        observation, reward, done, info = env.last()
        action = random.randint(0,4) if not done else None
        env.step(action)
        
        idx = agent_idx[agent]
        if idx != 0:
            r += reward
    rewards.append(r/2)

print(f"Average reward of agents over {episodes} episodes: {sum(rewards)/episodes:.2f} +- {statistics.stdev(rewards):.2f}")
