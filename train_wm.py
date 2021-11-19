"""
train_wm.py - Train the VAE and MDRNN of the world model.
"""
from functools import partial
from os.path import join, exists
from os import mkdir, getcwd, listdir
from bisect import bisect
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.distributions.normal import Normal
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import tools


def to_one_hot(indices, max_index):
    indices = torch.from_numpy(indices.astype(np.float32))
    zeros = torch.zeros(indices.size()[0], max_index, dtype=torch.float32)
    indices = indices.long() + torch.arange(0, max_index, 5, dtype=torch.long)
    return zeros.scatter_(1, indices, 1).numpy()


class _RolloutDataset(torch.utils.data.Dataset):
    def __init__(self, root, buffer_size=200, train=True, episodes=1000):
        self._files = [
            join(root, sd)
            for sd in listdir(root)
            if sd != "episodes.h5"
        ]
        if train:
            self._files = self._files[:episodes]
        else:
            self._files = self._files[episodes//5*4:]
        self._cum_size = None
        self._buffer = None
        self._buffer_fnames = None
        self._buffer_index = 0
        self._buffer_size = buffer_size
    def load_next_buffer(self):
        self._buffer_fnames = self._files[self._buffer_index:self._buffer_index + self._buffer_size]
        self._buffer_index += self._buffer_size
        self._buffer_index = self._buffer_index % len(self._files)
        self._buffer = []
        self._cum_size = [0]
        for f in self._buffer_fnames:
            with np.load(f, allow_pickle=True) as data:
                self._buffer += [{k: np.copy(v) for k, v in data.items()}]
                self._cum_size += [self._cum_size[-1] +
                                   self._data_per_sequence(data['rewards'].shape[0])]
    def __len__(self):
        if not self._cum_size:
            self.load_next_buffer()
        return self._cum_size[-1]
    def __getitem__(self, i):
        file_index = bisect(self._cum_size, i) - 1
        seq_index = i - self._cum_size[file_index]
        data = self._buffer[file_index]
        return self._get_data(data, seq_index)
    def _get_data(self, data, seq_index):
        pass
    def _data_per_sequence(self, data_length):
        pass


class RolloutSequenceDataset(_RolloutDataset):
    def __init__(self, root, seq_len, buffer_size=200, train=True):
        super().__init__(root, buffer_size, train)
        self._seq_len = seq_len
    def _get_data(self, data, seq_index):
        obs_data = data['observations'][seq_index:seq_index + self._seq_len + 1]
        obs_data = obs_data.astype(np.float32)
        obs, next_obs = obs_data[:-1], obs_data[1:]
        action = data['actions'][seq_index + 1:seq_index + self._seq_len + 1]
        action = to_one_hot(action, 15)
        action = action.astype(np.float32)
        reward, done = [data[key][seq_index + 1:
                                  seq_index + self._seq_len + 1].astype(np.float32)
                        for key in ('rewards', 'dones')]
        return obs, action, reward, done, next_obs

    def _data_per_sequence(self, data_length):
        return data_length - self._seq_len


class RolloutObservationDataset(_RolloutDataset):
    def _data_per_sequence(self, data_length):
        return data_length
    def _get_data(self, data, seq_index):
        return data['observations'][seq_index]


class Encoder(nn.Module):
    def __init__(self, latent_dim, n_hidden):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.n_hidden = n_hidden
        self.fc1 = nn.Linear(10, n_hidden)
        self.ln1 = nn.Linear(n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.ln2 = nn.LayerNorm(n_hidden)
        self.mu = nn.Linear(n_hidden, latent_dim)
        self.logsigma = nn.Linear(n_hidden, latent_dim)
    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.mu(x), self.logsigma(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, n_hidden):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.n_hidden = n_hidden
        self.fc1 = nn.Linear(latent_dim, n_hidden)
        self.ln1 = nn.LayerNorm(n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.ln2 = nn.LayerNorm(n_hidden)
        self.fc3 = nn.Linear(n_hidden, 10)
    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)


class VAE(nn.Module):
    def __init__(self, latent_dim, n_hidden):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.n_hidden = n_hidden
        self.encoder = Encoder(latent_dim, n_hidden)
        self.decoder = Decoder(latent_dim, n_hidden)
    def forward(self, x):
        mu, logsigma = self.encoder(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)
        recon_x = self.decoder(z)
        return recon_x, mu, logsigma

def loss_function(recon_x, x, mu, logsigma):
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    return BCE + KLD


def train(model, dataset_train, train_loader, device, optimizer):
    model.train()
    dataset_train.load_next_buffer()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

def test(model, dataset_test, test_loader, device):
    model.eval()
    dataset_test.load_next_buffer()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
    test_loss /= len(test_loader.dataset)
    return test_loss


def gmm_loss(batch, mus, sigmas, logpi, reduce=True):
    batch = batch.unsqueeze(-2)
    normal_dist = Normal(mus, sigmas)
    g_log_probs = normal_dist.log_prob(batch)
    g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)
    max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
    g_log_probs = g_log_probs - max_log_probs
    g_probs = torch.exp(g_log_probs)
    probs = torch.sum(g_probs, dim=-1)
    log_prob = max_log_probs.squeeze() + torch.log(probs)
    if reduce:
        return - torch.mean(log_prob)
    return - log_prob

class _MDRNNBase(nn.Module):
    def __init__(self, spatial_latent_dim, temporal_latent_dim, n_gaussians, action_dim=5, n_agents=3):
        super().__init__()
        self.spatial_latent_dim = spatial_latent_dim
        self.latent_dim = temporal_latent_dim
        self.n_gaussians = n_gaussians
        self.action_dim = action_dim * n_agents
        # mu, sigma, pi, dones
        self.gmm_linear = nn.Linear(temporal_latent_dim, (2 * spatial_latent_dim + 1) * n_gaussians + 1)
    def forward(self, *inputs):
        pass

class MDRNN(_MDRNNBase):
    def __init__(self, spatial_latent_dim, temporal_latent_dim, n_gaussians, action_dim=5, n_agents=3):
        super().__init__(spatial_latent_dim, temporal_latent_dim, n_gaussians, action_dim, n_agents)
        self.rnn = nn.LSTM(spatial_latent_dim + action_dim * n_agents, temporal_latent_dim)
    def forward(self, actions, spatial_latents):
        seq_len, bs = actions.size(0), actions.size(1)
        ins = torch.cat([actions, spatial_latents], dim=-1)
        outs, _ = self.rnn(ins)
        gmm_outs = self.gmm_linear(outs)
        stride = self.n_gaussians * self.spatial_latent_dim
        mus = gmm_outs[:, :, :stride]
        mus = mus.view(seq_len, bs, self.n_gaussians, self.spatial_latent_dim)
        sigmas = gmm_outs[:, :, stride:2 * stride]
        sigmas = sigmas.view(seq_len, bs, self.n_gaussians, self.spatial_latent_dim)
        sigmas = torch.exp(sigmas)
        pi = gmm_outs[:, :, 2 * stride: 2 * stride + self.n_gaussians]
        pi = pi.view(seq_len, bs, self.n_gaussians)
        logpi = F.log_softmax(pi, dim=-1)
        ds = gmm_outs[:, :, -1]
        return mus, sigmas, logpi, ds


class MDRNNCell(_MDRNNBase):
    def __init__(self, spatial_latent_dim, temporal_latent_dim, n_gaussians, action_dim=5, n_agents=3):
        super().__init__(spatial_latent_dim, temporal_latent_dim, n_gaussians, action_dim, n_agents)
        self.rnn = nn.LSTMCell(spatial_latent_dim + action_dim * n_agents, temporal_latent_dim)
    def forward(self, action, spatial_latent, temporal_latent):
        in_al = torch.cat([action, spatial_latent], dim=1)
        next_hidden = self.rnn(in_al, temporal_latent)
        out_rnn = next_hidden[0]
        out_full = self.gmm_linear(out_rnn)
        stride = self.n_gaussians * self.spatial_latent_dim
        mus = out_full[:, :stride]
        mus = mus.view(-1, self.n_gaussians, self.spatial_latent_dim)
        sigmas = out_full[:, stride:2 * stride]
        sigmas = sigmas.view(-1, self.n_gaussians, self.spatial_latent_dim)
        sigmas = torch.exp(sigmas)
        pi = out_full[:, 2 * stride:2 * stride + self.n_gaussians]
        pi = pi.view(-1, self.n_gaussians)
        logpi = F.log_softmax(pi, dim=-1)
        d = out_full[:, -1]
        return mus, sigmas, logpi, d, next_hidden

def to_latent(vae, obs, next_obs, spatial_latent_dim):
    with torch.no_grad():
        (obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma) = [
            vae(x)[1:] for x in (obs, next_obs)]
        latent_obs, latent_next_obs = [
            (x_mu + x_logsigma.exp() * torch.randn_like(x_mu)).view(16, 32, spatial_latent_dim)
            for x_mu, x_logsigma in
            [(obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma)]]
    return latent_obs, latent_next_obs


def get_loss(mdrnn, latent_obs, action, reward, done, latent_next_obs, spatial_latent_dim):
    latent_obs, action, reward, done, latent_next_obs = \
        [arr.transpose(1, 0)
         for arr in [latent_obs, action, reward, done, latent_next_obs]]
    mus, sigmas, logpi, ds = mdrnn(action, latent_obs)
    gmm = gmm_loss(latent_next_obs, mus, sigmas, logpi)
    bce = F.binary_cross_entropy_with_logits(ds, done)
    scale = spatial_latent_dim + 1
    loss = (gmm + bce) / scale
    return dict(gmm=gmm, bce=bce, loss=loss)

def data_pass(vae, mdrnn, optimizer, loader, spatial_latent_dim, device, train):
    if train:
        mdrnn.train()
    else:
        mdrnn.eval()
    loader.dataset.load_next_buffer()
    cum_loss = 0
    cum_gmm = 0
    cum_bce = 0
    for i, data in enumerate(loader):
        obs, action, reward, done, next_obs = [arr.to(device) for arr in data]
        latent_obs, latent_next_obs = to_latent(vae, obs, next_obs, spatial_latent_dim)
        if train:
            losses = get_loss(mdrnn, latent_obs, action, reward, done, latent_next_obs, spatial_latent_dim)
            optimizer.zero_grad()
            losses['loss'].backward()
            optimizer.step()
        else:
            with torch.no_grad():
                losses = get_loss(mdrnn, latent_obs, action, reward, done, latent_next_obs, spatial_latent_dim)
        cum_loss += losses['loss'].item()
        cum_gmm += losses['gmm'].item()
        cum_bce += losses['bce'].item()
    return cum_loss * 16 / len(loader.dataset)


def train_vae(
        setting,
        latent_dim,
        n_hidden,
        epochs,
        data_dir="datasets",
        model_dir="models",
        verbose=False,
        count_params=False):
    cuda = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if cuda else "cpu")
    data_dir = join(data_dir, setting)
    model_dir = join(getcwd(), model_dir, setting)
    if not exists(model_dir):
        mkdir(model_dir)
    if verbose:
        print(f"Training VAE in {model_dir} on: {data_dir}")

    dataset_train = RolloutObservationDataset(data_dir, train=True)
    # dataset_test = RolloutObservationDataset(data_dir, train=False)
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=32, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(
    #     dataset_test, batch_size=32, shuffle=True)
    model = VAE(latent_dim, n_hidden).to(device)
    optimizer = optim.Adam(model.parameters())
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count_params:
        print(f"VAE latent_dim={latent_dim} n_hidden={n_hidden} param_count={param_count}")
        print(model)
        return

    best = None
    for _ in range(1, epochs + 1):
        loss = train(model, dataset_train, train_loader, device, optimizer)
        # test_loss = test(model, dataset_test, test_loader, device)
        if not best or loss < best:
            best = loss
            torch.save(model, join(model_dir, 'vae.tar'))
    if verbose:
        print(f"Trained VAE with loss: {best:.3f}")

    return {
        'vae_loss': best,
        'vae_params': param_count
    }


def train_mdrnn(
        setting,
        spatial_latent_dim,
        temporal_latent_dim,
        n_gaussians,
        epochs,
        data_dir="datasets",
        model_dir="models",
        verbose=False,
        count_params=False):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    data_dir = join(data_dir, setting)
    model_dir = join(getcwd(), model_dir, setting)
    if not exists(model_dir):
        mkdir(model_dir)
    if verbose:
        print(f"Training MDRNN in {model_dir} on: {data_dir}")
    vae_file = join(getcwd(), model_dir, "vae.tar")
    if not exists(vae_file):
        raise Exception(f"VAE not found: {vae_file}")
    vae = torch.load(vae_file)

    train_loader = DataLoader(
        RolloutSequenceDataset(data_dir, seq_len=32, buffer_size=30),
        batch_size=16, shuffle=True, drop_last=True)
    # test_loader = DataLoader(
    #     RolloutSequenceDataset(data_dir, seq_len=32, train=False, buffer_size=10),
    #     batch_size=16, drop_last=True)
    mdrnn = MDRNN(spatial_latent_dim, temporal_latent_dim, n_gaussians)
    mdrnn.to(device)
    mdrnn_cell = MDRNNCell(spatial_latent_dim, temporal_latent_dim, n_gaussians)
    mdrnn_cell.to(device)
    optimizer = torch.optim.RMSprop(mdrnn.parameters(), lr=1e-3, alpha=.9)
    train = partial(data_pass, vae, mdrnn, optimizer, train_loader, spatial_latent_dim, device, train=True)
    # test = partial(data_pass, vae, mdrnn, optimizer, test_loader, spatial_latent_dim, device, train=False)
    param_count = sum(p.numel() for p in mdrnn.parameters() if p.requires_grad)
    if count_params:
        print(f"MDRNN spatial_latent_dim={spatial_latent_dim} temporal_latent_dim={temporal_latent_dim} n_gaussians={n_gaussians} param_count={param_count}")
        print(mdrnn)
        return

    best = None
    for _ in range(epochs):
        loss = train()
        # test_loss = test()
        if not best or loss < best:
            best = loss
            mdrnn_cell.load_state_dict({k.strip('_l0'): v for k, v in mdrnn.state_dict().items()})
            torch.save(mdrnn_cell, join(model_dir, 'mdrnn.tar'))
    if verbose:
        print(f"Trained MDRNN with loss: {best:.3f}")

    return {
        'mdrnn_loss': best,
        'mdrnn_params': sum(p.numel() for p in mdrnn.parameters() if p.requires_grad)
    }


if __name__ == "__main__":
    # train_vae(setting="random", latent_dim=10, n_hidden=5, n_layers=1, epochs=1)
    # train_vae(setting="spurious", latent_dim=10, n_hidden=5, n_layers=1, epochs=1)
    # train_vae(setting="expert", latent_dim=10, n_hidden=5, n_layers=1, epochs=1)
    # train_mdrnn(setting="random", spatial_latent_dim=10, temporal_latent_dim=5, n_gaussians=3, epochs=1)
    # train_mdrnn(setting="spurious", spatial_latent_dim=10, temporal_latent_dim=5, n_gaussians=3, epochs=1)
    # train_mdrnn(setting="expert", spatial_latent_dim=10, temporal_latent_dim=5, n_gaussians=3, epochs=1)

    # Total params: 2434
    train_vae(setting="random", latent_dim=5, n_hidden=10, epochs=1, count_params=True)  # 780
    train_mdrnn(setting="random", spatial_latent_dim=5, temporal_latent_dim=10, n_gaussians=3, epochs=1, count_params=True)  # 1654

