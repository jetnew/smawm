import os
from os.path import join, exists
from os import mkdir, getcwd, listdir
import h5py
import numpy as np
import pickle
import torch
from torch.utils import data
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm


def to_float(np_array):
    return np.array(np_array, dtype=np.float32)

def to_one_hot(indices, max_index, action_dim=5):
    zeros = torch.zeros(
        indices.size()[0], max_index, dtype=torch.float32,
        device=indices.device)
    indices = indices.long() + torch.arange(0, max_index, action_dim, dtype=torch.long, device=indices.device)
    return zeros.scatter_(1, indices, 1)

def to_one_hot2(indices, max_index):
    zeros = torch.zeros(
        indices.size()[0], max_index, dtype=torch.float32,
        device=indices.device)
    return zeros.scatter_(1, indices.unsqueeze(1), 1)


def load_dict_h5py(fname):
    array_dict = dict()
    with h5py.File(fname, 'r') as hf:
        for key in hf.keys():
            array_dict[key] = hf[key][:]
    return array_dict


def load_list_dict_h5py(fname):
    array_dict = list()
    with h5py.File(fname, 'r') as hf:
        for i, grp in enumerate(hf.keys()):
            array_dict.append(dict())
            for key in hf[grp].keys():
                array_dict[i][key] = hf[grp][key][:]
    return array_dict

def unsorted_segment_sum(tensor, segment_ids, num_segments):
    result_shape = (num_segments, tensor.size(1))
    result = tensor.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, tensor.size(1))
    result.scatter_add_(0, segment_ids, tensor)
    return result


class StateTransitionsDataset(data.Dataset):
    def __init__(self, hdf5_file):
        self.experience_buffer = load_list_dict_h5py(hdf5_file)
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


class ContrastiveSWM(nn.Module):
    def __init__(self, agent_latent_dim, n_hidden, n_layers):
        super(ContrastiveSWM, self).__init__()
        self.agent_latent_dim = agent_latent_dim
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.action_dim = 5
        self.num_agents = 3
        self.hinge = 1.
        self.sigma = 0.5
        self.ignore_action = False
        self.copy_action = False
        self.pos_loss = 0
        self.neg_loss = 0
        self.latent_dim = agent_latent_dim * self.num_agents
        self.obj_encoder = EncoderMLP(
            agent_latent_dim=agent_latent_dim,
            n_hidden=n_hidden)
        self.transition_model = TransitionGNN(
            agent_latent_dim=agent_latent_dim,
            n_hidden=n_hidden,
            n_layers=n_layers)
        self.decoder = DecoderMLP(
            agent_latent_dim=agent_latent_dim,
            n_hidden=n_hidden)
    def energy(self, state, action, next_state, no_trans=False):
        norm = 0.5 / (self.sigma ** 2)
        if no_trans:
            diff = state - next_state
        else:
            pred_trans = self.transition_model(state, action)
            diff = state + pred_trans - next_state
        return norm * diff.pow(2).sum(2).mean(1)
    def transition_loss(self, state, action, next_state):
        return self.energy(state, action, next_state).mean()
    def contrastive_loss(self, obs, action, next_obs):
        state = self.obj_encoder(obs)
        next_state = self.obj_encoder(next_obs)
        batch_size = state.size(0)
        perm = np.random.permutation(batch_size)
        neg_state = state[perm]
        self.pos_loss = self.energy(state, action, next_state)
        zeros = torch.zeros_like(self.pos_loss)
        self.pos_loss = self.pos_loss.mean()
        self.neg_loss = torch.max(
            zeros, self.hinge - self.energy(
                state, action, neg_state, no_trans=True)).mean()
        loss = self.pos_loss + self.neg_loss
        return loss
    def decoder_loss(self, obs, action, next_obs):
        state = self.obj_encoder(obs)
        rec = torch.sigmoid(self.decoder(state))
        loss = F.mse_loss(rec, obs, reduction='sum') / obs.size(0)
        next_state_pred = state + self.transition_model(state, action)
        next_rec = torch.sigmoid(self.decoder(next_state_pred))
        next_loss = F.mse_loss(next_rec, next_obs, reduction='sum') / obs.size(0)
        loss += next_loss
        return loss

    def forward(self, obs):
        return self.obj_encoder(obs)


class TransitionGNN(torch.nn.Module):
    def __init__(self, agent_latent_dim, n_hidden, n_layers):
        super(TransitionGNN, self).__init__()
        self.agent_latent_dim = agent_latent_dim
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.num_agents = 3
        self.ignore_action = False
        self.copy_action = False
        self.action_dim = 5
        edge_layers = [nn.Linear(agent_latent_dim * 2, n_hidden), nn.ReLU()]
        for _ in range(n_layers):
            edge_layers.append(nn.Linear(n_hidden, n_hidden))
            edge_layers.append(nn.LayerNorm(n_hidden))
            edge_layers.append(nn.ReLU())
        edge_layers.append(nn.Linear(n_hidden, n_hidden))
        self.edge_mlp = nn.Sequential(*edge_layers)
        node_input_dim = n_hidden + agent_latent_dim + self.action_dim
        node_layers = [nn.Linear(node_input_dim, n_hidden), nn.ReLU()]
        for _ in range(n_layers):
            node_layers.append(nn.Linear(n_hidden, n_hidden))
            node_layers.append(nn.LayerNorm(n_hidden))
            node_layers.append(nn.ReLU())
        node_layers.append(nn.Linear(n_hidden, agent_latent_dim))
        self.node_mlp = nn.Sequential(*node_layers)
        self.edge_list = None
        self.batch_size = 0
    def _edge_model(self, source, target, edge_attr):
        del edge_attr  # Unused.
        out = torch.cat([source, target], dim=1)
        return self.edge_mlp(out)
    def _node_model(self, node_attr, edge_index, edge_attr):
        if edge_attr is not None:
            row, col = edge_index
            agg = unsorted_segment_sum(
                edge_attr, row, num_segments=node_attr.size(0))
            out = torch.cat([node_attr, agg], dim=1)
        else:
            out = node_attr
        return self.node_mlp(out)
    def _get_edge_list_fully_connected(self, batch_size, num_agents, cuda):
        if self.edge_list is None or self.batch_size != batch_size:
            self.batch_size = batch_size
            adj_full = torch.ones(num_agents, num_agents)
            adj_full -= torch.eye(num_agents)
            self.edge_list = adj_full.nonzero()
            self.edge_list = self.edge_list.repeat(batch_size, 1)
            offset = torch.arange(
                0, batch_size * num_agents, num_agents).unsqueeze(-1)
            offset = offset.expand(batch_size, num_agents * (num_agents - 1))
            offset = offset.contiguous().view(-1)
            self.edge_list += offset.unsqueeze(-1)
            self.edge_list = self.edge_list.transpose(0, 1)
            if cuda:
                self.edge_list = self.edge_list.cuda()
        return self.edge_list
    def forward(self, states, action):
        cuda = states.is_cuda
        batch_size = states.size(0)
        num_nodes = states.size(1)
        node_attr = states.view(-1, self.agent_latent_dim)
        edge_attr = None
        edge_index = None
        if num_nodes > 1:
            edge_index = self._get_edge_list_fully_connected(
                batch_size, num_nodes, cuda)
            row, col = edge_index
            edge_attr = self._edge_model(
                node_attr[row], node_attr[col], edge_attr)
        if not self.ignore_action:
            if self.copy_action:
                action_vec = to_one_hot(
                    action, self.action_dim).repeat(1, self.num_agents)
                action_vec = action_vec.view(-1, self.action_dim)
            else:
                action_vec = to_one_hot(
                    action, self.action_dim * num_nodes)
                action_vec = action_vec.view(-1, self.action_dim)
            node_attr = torch.cat([node_attr, action_vec], dim=-1)
        node_attr = self._node_model(
            node_attr, edge_index, edge_attr)
        return node_attr.view(batch_size, num_nodes, -1)


class EncoderMLP(nn.Module):
    def __init__(self, agent_latent_dim, n_hidden):
        super(EncoderMLP, self).__init__()
        self.agent_latent_dim = agent_latent_dim
        self.num_agents = 3
        self.fc1 = nn.Linear(10, n_hidden)
        self.ln1 = nn.LayerNorm(n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.ln2 = nn.LayerNorm(n_hidden)
        self.fc3 = nn.Linear(n_hidden, self.agent_latent_dim * self.num_agents)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.act(self.ln1(self.fc1(x)))
        x = self.act(self.ln2(self.fc2(x)))
        return self.fc3(x).view(-1, self.num_agents, self.agent_latent_dim)

class DecoderMLP(nn.Module):
    def __init__(self, agent_latent_dim, n_hidden):
        super(DecoderMLP, self).__init__()
        self.agent_latent_dim = agent_latent_dim
        self.num_agents = 3
        self.fc1 = nn.Linear(agent_latent_dim * self.num_agents, n_hidden)
        self.ln1 = nn.LayerNorm(n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.ln2 = nn.LayerNorm(n_hidden)
        self.fc3 = nn.Linear(n_hidden, 10)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.act(self.ln1(self.fc1(x.view(-1, self.agent_latent_dim * self.num_agents))))
        x = self.act(self.ln2(self.fc2(x)))
        return self.fc3(x).view(-1, 10)


def train_swm(
        setting,
        agent_latent_dim,
        n_hidden,
        n_layers,
        epochs,
        data_dir="datasets",
        model_dir="models",
        verbose=False,
        count_params=False):
    cuda = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if cuda else "cpu")
    data_dir = join(data_dir, setting, "episodes.h5")
    model_dir = join(getcwd(), model_dir, setting)
    if not exists(model_dir):
        mkdir(model_dir)
    if verbose:
        print(f"Training SWM in {model_dir} on: {data_dir}")

    dataset = StateTransitionsDataset(hdf5_file=data_dir)
    train_loader = data.DataLoader(dataset, batch_size=32, shuffle=True)
    model = ContrastiveSWM(agent_latent_dim, n_hidden, n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count_params:
        print(f"SWM agent_latent_dim={agent_latent_dim} n_hidden={n_hidden} n_layers={n_layers} param_count={param_count}")
        print(model)
        return

    best = None
    for _ in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, data_batch in enumerate(train_loader):
            data_batch = [tensor.to(device) for tensor in data_batch]
            optimizer.zero_grad()
            loss = model.decoder_loss(*data_batch)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        avg_loss = train_loss / len(train_loader.dataset)
        if not best or avg_loss < best:
            best = avg_loss
            torch.save(model, join(model_dir, 'swm.tar'))
    if verbose:
        print(f"Trained SWM with loss: {best:.3g}")

    return {
        'swm_loss': best,
        'swm_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }


if __name__ == "__main__":
    # train_swm(setting="random", agent_latent_dim=5, n_hidden=5, n_layers=0, epochs=1)
    # train_swm(setting="spurious", agent_latent_dim=5, n_hidden=5, n_layers=0, epochs=1)
    # train_swm(setting="expert", agent_latent_dim=5, n_hidden=5, n_layers=0, epochs=1)

    # train_swm(setting="random", agent_latent_dim=5, n_hidden=10, n_layers=1, epochs=1, count_params=True)  # 1150
    # train_swm(setting="random", agent_latent_dim=5, n_hidden=15, n_layers=1, epochs=1, count_params=True)  # 2090
    # train_swm(setting="random", agent_latent_dim=10, n_hidden=10, n_layers=1, epochs=1, count_params=True)  # 1520

    train_swm(setting="random", agent_latent_dim=5, n_hidden=10, n_layers=1, epochs=1, count_params=True)  # 1590
    train_swm(setting="random", agent_latent_dim=10, n_hidden=10, n_layers=1, epochs=1, count_params=True)  # 2110
    train_swm(setting="random", agent_latent_dim=5, n_hidden=10, n_layers=2, epochs=1, count_params=True)  # 1850