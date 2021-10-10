""" Some data loading utilities """
from bisect import bisect
from os import listdir
from os.path import join
from tqdm import tqdm
import torch
import torch.utils.data
import numpy as np


def to_one_hot(indices, max_index):
    """Get one-hot encoding of index tensors."""
    zeros = torch.zeros(
        indices.size()[0], max_index, dtype=torch.float32,
        device=indices.device)
    indices = indices.long() + torch.arange(0, max_index, action_dim, dtype=torch.long, device=indices.device)
    return zeros.scatter_(1, indices, 1)

class _RolloutDataset(torch.utils.data.Dataset):
    def __init__(self, root, buffer_size=200, train=True):
        self._files = [
            join(root, sd)
            for sd in listdir(root)]
        
        if train:
            self._files = self._files[:-600]
        else:
            self._files = self._files[-600:]

        self._cum_size = None
        self._buffer = None
        self._buffer_fnames = None
        self._buffer_index = 0
        self._buffer_size = buffer_size

    def load_next_buffer(self):
        """ Loads next buffer """
        self._buffer_fnames = self._files[self._buffer_index:self._buffer_index + self._buffer_size]
        self._buffer_index += self._buffer_size
        self._buffer_index = self._buffer_index % len(self._files)
        self._buffer = []
        self._cum_size = [0]

        # progress bar
        pbar = tqdm(total=len(self._buffer_fnames),
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
        pbar.set_description("Loading file buffer ...")

        for f in self._buffer_fnames:
            with np.load(f, allow_pickle=True) as data:
                self._buffer += [{k: np.copy(v) for k, v in data.items()}]
                self._cum_size += [self._cum_size[-1] +
                                   self._data_per_sequence(data['rewards'].shape[0])]
            pbar.update(1)
        pbar.close()

    def __len__(self):
        # to have a full sequence, you need self.seq_len + 1 elements, as
        # you must produce both an seq_len obs and seq_len next_obs sequences
        if not self._cum_size:
            self.load_next_buffer()
        return self._cum_size[-1]

    def __getitem__(self, i):
        # binary search through cum_size
        file_index = bisect(self._cum_size, i) - 1
        seq_index = i - self._cum_size[file_index]
        data = self._buffer[file_index]
        return self._get_data(data, seq_index)

    def _get_data(self, data, seq_index):
        pass

    def _data_per_sequence(self, data_length):
        pass


class RolloutSequenceDataset(_RolloutDataset):
    """ Encapsulates rollouts.
    Rollouts should be stored in subdirs of the root directory, in the form of npz files,
    each containing a dictionary with the keys:
        - observations: (rollout_len, *obs_shape)
        - actions: (rollout_len, action_size)
        - rewards: (rollout_len,)
        - terminals: (rollout_len,), boolean
     As the dataset is too big to be entirely stored in rams, only chunks of it
     are stored, consisting of a constant number of files (determined by the
     buffer_size parameter).  Once built, buffers must be loaded with the
     load_next_buffer method.
    Data are then provided in the form of tuples (obs, action, reward, done, next_obs):
    - obs: (seq_len, *obs_shape)
    - actions: (seq_len, action_size)
    - reward: (seq_len,)
    - done: (seq_len,) boolean
    - next_obs: (seq_len, *obs_shape)
    NOTE: seq_len < rollout_len in moste use cases
    :args root: root directory of data sequences
    :args seq_len: number of timesteps extracted from each rollout
    :args train: if True, train data, else test
    """
    def __init__(self, root, seq_len, buffer_size=200, train=True): # pylint: disable=too-many-arguments
        super().__init__(root, buffer_size, train)
        self._seq_len = seq_len

    def _get_data(self, data, seq_index):
        obs_data = data['observations'][seq_index:seq_index + self._seq_len + 1]
        obs_data = obs_data.astype(np.float32)
        obs, next_obs = obs_data[:-1], obs_data[1:]
        action = data['actions'][seq_index+1:seq_index + self._seq_len + 1]
        action = to_one_hot(action, 15)
        action = action.astype(np.float32)
        reward, done = [data[key][seq_index+1:
                                      seq_index + self._seq_len + 1].astype(np.float32)
                            for key in ('rewards', 'dones')]
        # data is given in the form
        # (obs, action, reward, done, next_obs)
        return obs, action, reward, done, next_obs

    def _data_per_sequence(self, data_length):
        return data_length - self._seq_len

class RolloutObservationDataset(_RolloutDataset): # pylint: disable=too-few-public-methods
    """ Encapsulates rollouts.
    Rollouts should be stored in subdirs of the root directory, in the form of npz files,
    each containing a dictionary with the keys:
        - observations: (rollout_len, *obs_shape)
        - actions: (rollout_len, action_size)
        - rewards: (rollout_len,)
        - terminals: (rollout_len,), boolean
     As the dataset is too big to be entirely stored in rams, only chunks of it
     are stored, consisting of a constant number of files (determined by the
     buffer_size parameter).  Once built, buffers must be loaded with the
     load_next_buffer method.
    Data are then provided in the form of images
    :args root: root directory of data sequences
    :args seq_len: number of timesteps extracted from each rollout
    :args train: if True, train data, else test
    """
    def _data_per_sequence(self, data_length):
        return data_length

    def _get_data(self, data, seq_index):
        return data['observations'][seq_index]