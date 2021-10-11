"""NOT USED"""
import os
import numpy as np
import h5py
import pickle
import torch
from torch.utils import data
from torch import nn
import torch.nn.functional as F


def to_one_hot(indices, max_index):
    """Get one-hot encoding of index tensors."""
    zeros = torch.zeros(
        indices.size()[0], max_index, dtype=torch.float32,
        device=indices.device)
    indices = indices.long() + torch.arange(0, max_index, action_dim, dtype=torch.long, device=indices.device)
    return zeros.scatter_(1, indices, 1)

def load_list_dict_h5py(fname):
    """Restore list of dictionaries containing numpy arrays from h5py file."""
    array_dict = list()
    with h5py.File(fname, 'r') as hf:
        for i, grp in enumerate(hf.keys()):
            array_dict.append(dict())
            for key in hf[grp].keys():
                array_dict[i][key] = hf[grp][key][:]
    return array_dict
    
def to_float(np_array):
    """Convert numpy array to float32."""
    return np.array(np_array, dtype=np.float32)

class StateTransitionsDataset(data.Dataset):
    """Create dataset of (o_t, a_t, o_{t+1}) transitions from replay buffer."""

    def __init__(self, hdf5_file):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience
                buffer
        """
        self.experience_buffer = load_list_dict_h5py(hdf5_file)

        # Build table for conversion between linear idx -> episode/step idx
        self.idx2episode = list()
        step = 0
        for ep in range(len(self.experience_buffer)):
            num_steps = len(self.experience_buffer[ep]['action'])
            idx_tuple = [(ep, idx) for idx in range(num_steps)]
            self.idx2episode.extend(idx_tuple)
            step += num_steps

        self.num_steps = step

    def __len__(self):
        return self.num_steps

    def __getitem__(self, idx):
        ep, step = self.idx2episode[idx]

        obs = to_float(self.experience_buffer[ep]['obs'][step])
        action = self.experience_buffer[ep]['action'][step]
        next_obs = to_float(self.experience_buffer[ep]['next_obs'][step])

        return obs, action, next_obs


def get_act_fn(act_fn):
    if act_fn == 'relu':
        return nn.ReLU()
    elif act_fn == 'leaky_relu':
        return nn.LeakyReLU()
    elif act_fn == 'elu':
        return nn.ELU()
    elif act_fn == 'sigmoid':
        return nn.Sigmoid()
    elif act_fn == 'softplus':
        return nn.Softplus()
    else:
        raise ValueError('Invalid argument for `act_fn`.')


class MAPolicy(nn.Module):
    def __init__(self, latent_dim, hidden_dim=32, action_dim=5, num_agents=2):
        super(MAPolicy, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, action_dim * num_agents)
        self.act2 = nn.Softmax()
        
    def forward(self, *inputs):
        cat_in = torch.cat(inputs, dim=1)
        h = self.act1(self.fc1(cat_in))
        return self.act2(self.fc2(h))
        
        
if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    
    np.random.seed(0)
    torch.manual_seed(0)
    if cuda:
        torch.cuda.manual_seed(0)
    
    save_folder = 'checkpoints/agents'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    model_file = os.path.join(save_folder, 'MAPolicy.pt')
    
    device = torch.device('cuda' if cuda else 'cpu')
    dataset = "datasets/mpe-episodes.h5"
        
    batch_size = 32
    dataset = StateTransitionsDataset(
        hdf5_file=dataset)
    train_loader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    RSIZE = 30
    LSIZE = 15
    ASIZE = 15
    
    # Load VAE and MDRNN
    vae_file, rnn_file = \
        [join(mdir, m, 'best.tar') for m in ['vae', 'mdrnn']]

    assert exists(vae_file) and exists(rnn_file),\
        "Either vae or mdrnn is untrained."

    vae_state, rnn_state = [
        torch.load(fname, map_location={'cuda:0': str(device)})
        for fname in (vae_file, rnn_file)]

    for m, s in (('VAE', vae_state), ('MDRNN', rnn_state)):
        print("Loading {} at epoch {} "
              "with test loss {}".format(
                  m, s['epoch'], s['precision']))

    vae = VAE(10, LSIZE).to(device)
    vae.load_state_dict(vae_state['state_dict'])

    mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(device)
    mdrnn.load_state_dict(
        {k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()})
    
    
    latent_dim = LSIZE + RSIZE
    hidden_dim = 32
    action_dim = 5
    num_agents = 2
    
    model = MAPolicy(latent_dim, hidden_dim, action_dim, num_agents)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print('Starting model training...')
    cur_best = None
    epochs = 5
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        
        hidden = [torch.zeros(1, RSIZE).to(device) for _ in range(2)]
        
        for batch_idx, data_batch in enumerate(train_loader):
            data_batch = [tensor.to(device) for tensor in data_batch]
            optimizer.zero_grad()
            
            obs, action, next_obs = data_batch
            
            _, latent_mu, _ = vae(obs)
            action_pred = model(latent_mu, hidden[0])
            
            loss = F.binary_cross_entropy(action_pred, action, reduction='sum') / action.size(0)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
            if batch_idx % 20 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data_batch)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')

        avg_loss = train_loss / len(train_loader.dataset)
        print('====> Epoch: {} Average loss: {:.6f}'.format(
            epoch, avg_loss))

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_file)
        
        
        
    