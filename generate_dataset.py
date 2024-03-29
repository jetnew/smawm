"""
generate_dataset.py - Generates either the Random, Spurious or Expert dataset.
"""
from os import mkdir, getcwd
from os.path import join, exists
import h5py
import random
import numpy as np
from pettingzoo.mpe import simple_adversary_v2
from policies import random_policy, spurious_policy, follow_non_goal_landmark_policy, follow_goal_landmark_policy, follow_agent_closest_to_landmark_policy
from tqdm import tqdm


def save_list_dict_h5py(array_dict, fname):
    """Save list of dictionaries containing numpy arrays to h5py file."""
    with h5py.File(fname, 'w') as hf:
        for i in range(len(array_dict)):
            grp = hf.create_group(str(i))
            for key in array_dict[i].keys():
                grp.create_dataset(key, data=array_dict[i][key])


def generate_dataset(
        setting,
        data_dir="datasets",
        episodes=1000,
        episode_length=100,
        agents=2,
        generate_new=False,
        verbose=False):
    """
    Generates either the Random, Spurious or Expert dataset.
    Args:
        setting: str - Either "random", "spurious" or "expert"
        data_dir: str - Dataset save folder. Default: "datasets"
        episodes: int - Number of rounds in dataset. Default: 1000
        episode_length: int - Number of tuples per episode. Default: 1000
        agents: int - Number of cooperative agents. Default: 2
        seed: int - Random seed
    """
    data_dir = join(getcwd(), data_dir, setting)
    if exists(data_dir) and not generate_new:
        if verbose:
            print(f"Dataset {data_dir} exists.")
        return
    elif not exists(data_dir):
        mkdir(data_dir)
    if verbose:
        print(f"Generating dataset: {data_dir}")

    if setting == "random":
        adversary = random_policy
        agent1 = random_policy
        agent2 = random_policy
    elif setting == "spurious":
        adversary = random_policy
        agent1 = spurious_policy()
        agent2 = spurious_policy()
    elif setting == "expert":
        adversary = follow_agent_closest_to_landmark_policy
        agent1 = follow_non_goal_landmark_policy
        agent2 = follow_goal_landmark_policy
    else:
        raise Exception(f"Setting '{setting}' not available!")

    env = simple_adversary_v2.parallel_env(N=agents, max_cycles=episode_length, continuous_actions=False)

    for n in tqdm(range(episodes), disable=not verbose):
        obs = env.reset()
        s_rollout = []
        r_rollout = []
        d_rollout = []
        a_rollout = []

        for i in range(episode_length):
            adversary_action = adversary(obs['adversary_0'])
            agent1_action = agent1(obs['agent_0'])
            agent2_action = agent2(obs['agent_1'])
            actions = [adversary_action, agent1_action, agent2_action]

            obs, reward, done, _ = env.step({
                agent: action for agent, action in (('adversary_0', adversary_action),
                                                    ('agent_0', agent1_action),
                                                    ('agent_1', agent2_action))})

            s_rollout.append(obs['agent_1'])
            r_rollout.append(reward['agent_1'])
            d_rollout.append(done['agent_1'])
            a_rollout.append(actions)

        np.savez(join(data_dir, f'episode_{n}'),
                 observations=np.array(s_rollout),
                 rewards=np.array(r_rollout),
                 actions=np.array(a_rollout),
                 dones=np.array(d_rollout))

    replay_buffer = []
    for n in range(episodes):
        data = np.load(join(data_dir, f'episode_{n}.npz'), allow_pickle=True)
        dataset = {k: np.copy(v).tolist() for k, v in data.items()}
        replay_buffer.append({'obs': [], 'action': [], 'next_obs': []})
        for i in range(episode_length - 1):
            replay_buffer[n]['obs'].append(dataset['observations'][i])
            replay_buffer[n]['action'].append(dataset['actions'][i])
            replay_buffer[n]['next_obs'].append(dataset['observations'][i + 1])
    save_list_dict_h5py(replay_buffer, join(data_dir, "episodes.h5"))

    prediction_buffer = []
    for n in range(episodes):
        data = np.load(join(data_dir, f'episode_{n}.npz'), allow_pickle=True)
        dataset = {k: np.copy(v).tolist() for k, v in data.items()}
        prediction_buffer.append({**{f'obs_t{t}': [] for t in range(11)},
                                  **{f"action_t{t}": [] for t in range(10)}})
        for i in range(episode_length - 10):
            for t in range(10):
                prediction_buffer[n][f'obs_t{t}'].append(dataset['observations'][t])
                prediction_buffer[n][f'action_t{t}'].append(dataset['actions'][t])
            prediction_buffer[n]['obs_t10'].append(dataset['observations'][10])
    save_list_dict_h5py(prediction_buffer, join(data_dir, "predictions.h5"))

    env.close()

if __name__ == "__main__":
    generate_dataset(setting="random")
    generate_dataset(setting="spurious")
    generate_dataset(setting="expert")
