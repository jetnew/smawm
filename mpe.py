from os.path import join, exists
import random
import numpy as np
from pettingzoo.mpe import simple_adversary_v2


def generate_data(episodes=1000, data_dir="datasets/mpe", agents=2, episode_length=100):
    assert exists(data_dir), f"{data_dir} does not exist. Create a new directory {data_dir}."
    env = simple_adversary_v2.env(N=agents, max_cycles=episode_length, continuous_actions=False)
    env.reset()
    
    agent_idx = {}
    for i, agent in enumerate(env.agents):
        agent_idx[agent] = i

    for i in range(episodes):
        env.reset()
        s_rollout = []
        r_rollout = []
        d_rollout = []
        a_rollout = []
        a_t = []
        
        for agent in env.agent_iter():
            observation, reward, done, info = env.last()
            action = random.randint(0,4) if not done else None
            env.step(action)
            
            a_t.append(action)
            idx = agent_idx[agent]
            if idx == env.max_num_agents - 1:
                s_rollout.append(observation)
                r_rollout.append(reward)
                d_rollout.append(done)
                a_rollout.append(a_t)
                a_t = []
            
        print(f"End of episode {i}.")
        np.savez(join(data_dir, f'episode_{i}'),
                observations=np.array(s_rollout)[:-1],
                rewards=np.array(r_rollout)[:-1],
                actions=np.array(a_rollout)[:-1],
                dones=np.array(d_rollout)[:-1])
    env.close()

# need to be length to be 1000 otherwise MDRNN will error
generate_data(episodes=1000, data_dir='datasets/mpe', agents=2, episode_length=1000)