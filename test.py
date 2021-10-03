from pettingzoo.utils import average_total_reward
from pettingzoo.mpe import simple_adversary_v2
import random
from tqdm import tqdm

env = simple_adversary_v2.env(N=2, max_cycles=1000, continuous_actions=False)
env.reset()

agent_idx = {}
for i, agent in enumerate(env.agents):
    agent_idx[agent] = i
    
cum_reward = 0
episodes = 100
for i in tqdm(range(episodes)):
    env.reset()
    for agent in env.agent_iter():
        observation, reward, done, info = env.last()
        action = random.randint(0,4) if not done else None
        env.step(action)
        
        idx = agent_idx[agent]
        if idx != 0:
            cum_reward += reward

print(f"Average reward of agents over {episodes} episodes: {cum_reward/episodes/2:.2f}")
