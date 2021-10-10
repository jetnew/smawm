import os
import h5py
import numpy as np
from tqdm import tqdm

def save_list_dict_h5py(array_dict, fname):
    """Save list of dictionaries containing numpy arrays to h5py file."""

    # Ensure directory exists
    directory = os.path.dirname(fname)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with h5py.File(fname, 'w') as hf:
        for i in range(len(array_dict)):
            grp = hf.create_group(str(i))
            for key in array_dict[i].keys():
                grp.create_dataset(key, data=array_dict[i][key])

episodes = 1000
episode_length = 10

replay_buffer = []
for i in tqdm(range(episodes)):
    data = np.load(f"datasets/mpe/episode_{i}.npz", allow_pickle=True)
    dataset = {k: np.copy(v).tolist() for k, v in data.items()}

    replay_buffer.append({
        'obs': [],
        'action': [],
        'next_obs': [],
    })
    for j in range(episode_length - 1):
        replay_buffer[i]['obs'].append(dataset['observations'][j])
        replay_buffer[i]['action'].append(dataset['actions'][j])
        replay_buffer[i]['next_obs'].append(dataset['observations'][j + 1])

save_list_dict_h5py(replay_buffer, "datasets/mpe/episodes.h5")