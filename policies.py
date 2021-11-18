from tqdm import tqdm
import numpy as np
import random
import statistics
from pettingzoo.mpe import simple_adversary_v2
import time

"""
Moveset:
0 - Nothing
1 - Left
2 - Right
3 - Up
4 - Down

Agent observation space:
- Goal relative x
- Goal relative y
- Landmark 1 relative x
- Landmark 1 relative y
- Landmark 2 relative x
- Landmark 2 relative y
- Adversary relative x
- Adversary relative y
- Other agent relative x
- Other agent relative y

Adversary observation space:
- Landmark 1 relative x
- Landmark 1 relative y
- Landmark 2 relative x
- Landmark 2 relative y
- Agent 1 relative x
- Agent 1 relative y
- Agent 2 relative x
- Agent 2 relative y
"""

def move_towards(x, y):
    x_move = 1 if x < 0 else 2
    y_move = 3 if y < 0 else 4
    return x_move if abs(x) > abs(y) else y_move

def euclidean(x1, y1, x2, y2):
    return (x1-x2)**2 + (y1-y2)**2

def distance(obs, obj1, obj2):
    return euclidean(obs[obj1*2], obs[obj1*2+1], obs[obj2*2], obs[obj2*2+1])
    
def static_policy(obs=None):
    return 0
    
def random_policy(obs=None):
    return random.randint(0, 4)
    
def follow(obs, obj=3, eps=0.5):
    # obj refers to the object in the observation space
    if random.random() < eps:
        return random_policy()
    return move_towards(obs[obj*2], obs[obj*2+1])

def follow_non_goal_landmark_policy(obs, eps=0.2):
    if random.random() < eps:
        return random_policy()
    d1 = distance(obs, obj1=0, obj2=1)
    d2 = distance(obs, obj1=0, obj2=2)
    assert d1 == 0 or d2 == 0
    if d1 == 0:
        return follow(obs, obj=2, eps=0)
    elif d2 == 0:
        return follow(obs, obj=1, eps=0)
    raise Exception("Didn't work as expected")

def follow_goal_landmark_policy(obs, eps=0.2):
    if random.random() < eps:
        return random_policy()
    d1 = distance(obs, obj1=0, obj2=1)
    d2 = distance(obs, obj1=0, obj2=2)
    assert d1 == 0 or d2 == 0
    if d1 == 0:
        return follow(obs, obj=1, eps=0)
    elif d2 == 0:
        return follow(obs, obj=2, eps=0)
    raise Exception("Didn't work as expected")
    
def follow_agent_closest_to_landmark_policy(obs, eps=0.2):
    if random.random() < eps:
        return random_policy()
    dist_a1_l1 = distance(obs, obj1=2, obj2=0)
    dist_a2_l1 = distance(obs, obj1=3, obj2=0)
    dist_a1_l2 = distance(obs, obj1=2, obj2=1)
    dist_a2_l2 = distance(obs, obj1=3, obj2=1)
    if np.argmin([dist_a1_l1, dist_a2_l1, dist_a1_l2, dist_a2_l2]) < 2:
        return follow(obs, obj=0, eps=0)
    else:
        return follow(obs, obj=1, eps=0)

class spurious_policy:
    def __init__(self):
        self.previous = None
    def __call__(self, obs, eps=0.05):
        if self.previous is None:
            return random.randint(0, 4)
        else:
            if random.random() < eps:
                return random.randint(0, 4)
            action_space = [0, 1, 2, 3, 4]
            action_space.remove(self.previous)
            self.previous = None
            return random.choice(action_space)
        
def compute_reward(rewards):
        return sum(rewards) / episodes, statistics.stdev(rewards)


if __name__ == "__main__":
    env = simple_adversary_v2.env(N=2, max_cycles=100, continuous_actions=False)
    env.reset()

    agent_idx = {}
    for i, agent in enumerate(env.agents):
        agent_idx[agent] = i

    episodes = 1000

    agent1_rewards = []
    agent2_rewards = []
    adversary_rewards = []

    for i in tqdm(range(episodes)):
        env.reset()
        agent1_reward = 0
        agent2_reward = 0
        adversary_reward = 0
        
        for agent in env.agent_iter():
            idx = agent_idx[agent]
            
            obs, reward, done, info = env.last()

            if idx == 0:
                # action = static_policy() if not done else None
                # action = follow(obs, obj=3) if not done else None
                action = follow_agent_closest_to_landmark_policy(obs, eps=0) if not done else None
                
            elif idx == 1:
                #action = static_policy() if not done else None
                action = follow(obs, obj=0, eps=0.3) if not done else None
            else:
                #action = random_policy() if not done else None
                action = follow(obs, obj=2, eps=0.1) if not done else None
            
            env.step(action)
            
            if idx == 0:
                adversary_reward += reward
            elif idx == 1:
                agent1_reward += reward
            elif idx == 2:
                agent2_reward += reward
            
            # env.render()
        
        agent1_rewards.append(agent1_reward)
        agent2_rewards.append(agent2_reward)
        adversary_rewards.append(adversary_reward)
            

    a1_mean, a1_std = compute_reward(agent1_rewards)
    a2_mean, a2_std = compute_reward(agent2_rewards)
    ad_mean, ad_std = compute_reward(adversary_rewards)
    print(f"Agent 1: {a1_mean:.3f} +- {a1_std:.3f}")
    print(f"Agent 2: {a2_mean:.3f} +- {a2_std:.3f}")
    print(f"Adversary: {ad_mean:.3f} +- {ad_std:.3f}")
