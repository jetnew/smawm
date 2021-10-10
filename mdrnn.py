"""
Define MDRNN model, supposed to be used as a world model
on the latent space.
"""
from functools import partial
from os.path import join, exists
from os import mkdir, getcwd
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from misc import save_checkpoint
from loaders import RolloutSequenceDataset
from vae import VAE



def gmm_loss(batch, mus, sigmas, logpi, reduce=True):
    """ Computes the gmm loss.
    Compute minus the log probability of batch under the GMM model described
    by mus, sigmas, pi. Precisely, with bs1, bs2, ... the sizes of the batch
    dimensions (several batch dimension are useful when you have both a batch
    axis and a time step axis), gs the number of mixtures and fs the number of
    features.
    :args batch: (bs1, bs2, *, fs) torch tensor
    :args mus: (bs1, bs2, *, gs, fs) torch tensor
    :args sigmas: (bs1, bs2, *, gs, fs) torch tensor
    :args logpi: (bs1, bs2, *, gs) torch tensor
    :args reduce: if not reduce, the mean in the following formula is ommited
    :returns:
    loss(batch) = - mean_{i1=0..bs1, i2=0..bs2, ...} log(
        sum_{k=1..gs} pi[i1, i2, ..., k] * N(
            batch[i1, i2, ..., :] | mus[i1, i2, ..., k, :], sigmas[i1, i2, ..., k, :]))
    NOTE: The loss is not reduced along the feature dimension (i.e. it should scale ~linearily
    with fs).
    """
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
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__()
        self.latents = latents
        self.actions = actions
        self.hiddens = hiddens
        self.gaussians = gaussians

        self.gmm_linear = nn.Linear(
            hiddens, (2 * latents + 1) * gaussians + 2)

    def forward(self, *inputs):
        pass

class MDRNN(_MDRNNBase):
    """ MDRNN model for multi steps forward """
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__(latents, actions, hiddens, gaussians)
        self.rnn = nn.LSTM(latents + actions, hiddens)

    def forward(self, actions, latents): # pylint: disable=arguments-differ
        """ MULTI STEPS forward.
        :args actions: (SEQ_LEN, BSIZE, ASIZE) torch tensor
        :args latents: (SEQ_LEN, BSIZE, LSIZE) torch tensor
        :returns: mu_nlat, sig_nlat, pi_nlat, rs, ds, parameters of the GMM
        prediction for the next latent, gaussian prediction of the reward and
        logit prediction of terminality.
            - mu_nlat: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) torch tensor
            - sigma_nlat: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) torch tensor
            - logpi_nlat: (SEQ_LEN, BSIZE, N_GAUSS) torch tensor
            - rs: (SEQ_LEN, BSIZE) torch tensor
            - ds: (SEQ_LEN, BSIZE) torch tensor
        """
        seq_len, bs = actions.size(0), actions.size(1)

        ins = torch.cat([actions, latents], dim=-1)
        outs, _ = self.rnn(ins)
        gmm_outs = self.gmm_linear(outs)

        stride = self.gaussians * self.latents

        mus = gmm_outs[:, :, :stride]
        mus = mus.view(seq_len, bs, self.gaussians, self.latents)

        sigmas = gmm_outs[:, :, stride:2 * stride]
        sigmas = sigmas.view(seq_len, bs, self.gaussians, self.latents)
        sigmas = torch.exp(sigmas)

        pi = gmm_outs[:, :, 2 * stride: 2 * stride + self.gaussians]
        pi = pi.view(seq_len, bs, self.gaussians)
        logpi = f.log_softmax(pi, dim=-1)

        rs = gmm_outs[:, :, -2]

        ds = gmm_outs[:, :, -1]

        return mus, sigmas, logpi, rs, ds

class MDRNNCell(_MDRNNBase):
    """ MDRNN model for one step forward """
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__(latents, actions, hiddens, gaussians)
        self.rnn = nn.LSTMCell(latents + actions, hiddens)

    def forward(self, action, latent, hidden): # pylint: disable=arguments-differ
        """ ONE STEP forward.
        :args actions: (BSIZE, ASIZE) torch tensor
        :args latents: (BSIZE, LSIZE) torch tensor
        :args hidden: (BSIZE, RSIZE) torch tensor
        :returns: mu_nlat, sig_nlat, pi_nlat, r, d, next_hidden, parameters of
        the GMM prediction for the next latent, gaussian prediction of the
        reward, logit prediction of terminality and next hidden state.
            - mu_nlat: (BSIZE, N_GAUSS, LSIZE) torch tensor
            - sigma_nlat: (BSIZE, N_GAUSS, LSIZE) torch tensor
            - logpi_nlat: (BSIZE, N_GAUSS) torch tensor
            - rs: (BSIZE) torch tensor
            - ds: (BSIZE) torch tensor
        """
        in_al = torch.cat([action, latent], dim=1)

        next_hidden = self.rnn(in_al, hidden)
        out_rnn = next_hidden[0]

        out_full = self.gmm_linear(out_rnn)

        stride = self.gaussians * self.latents

        mus = out_full[:, :stride]
        mus = mus.view(-1, self.gaussians, self.latents)

        sigmas = out_full[:, stride:2 * stride]
        sigmas = sigmas.view(-1, self.gaussians, self.latents)
        sigmas = torch.exp(sigmas)

        pi = out_full[:, 2 * stride:2 * stride + self.gaussians]
        pi = pi.view(-1, self.gaussians)
        logpi = f.log_softmax(pi, dim=-1)

        r = out_full[:, -2]
        d = out_full[:, -1]

        return mus, sigmas, logpi, r, d, next_hidden
                

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading VAE
    logdir = 'exp_dir'
    vae_file = join(getcwd(), logdir, 'vae', 'best.tar')
    assert exists(vae_file), "No trained VAE in the logdir..."
    state = torch.load(vae_file)
    print(f"Loading VAE at epoch {state['epoch']} with test error {state['precision']}")
    vae = VAE(10, 15).to(device)
    vae.load_state_dict(state['state_dict'])


    LSIZE = 15
    ASIZE = 15  # 3 agents with 5D one-hot actions
    RSIZE = 30
    rnn_dir = join(getcwd(), logdir, 'mdrnn')
    rnn_file = join(rnn_dir, 'best.tar')
    if not exists(rnn_dir):
        mkdir(rnn_dir)
    mdrnn = MDRNN(latents=LSIZE, actions=ASIZE, hiddens=RSIZE, gaussians=5)
    mdrnn.to(device)
    optimizer = torch.optim.RMSprop(mdrnn.parameters(), lr=1e-3, alpha=.9)


    if exists(rnn_file):
        rnn_state = torch.load(rnn_file)
        print("Loading MDRNN at epoch {} "
              "with test error {}".format(
                  rnn_state["epoch"], rnn_state["precision"]))
        mdrnn.load_state_dict(rnn_state["state_dict"])
        optimizer.load_state_dict(rnn_state["optimizer"])


    # Data Loading
    BSIZE = 16
    SEQ_LEN = 32
    train_loader = DataLoader(
        RolloutSequenceDataset('datasets/mpe', SEQ_LEN, buffer_size=30),
        batch_size=BSIZE, shuffle=True)
    test_loader = DataLoader(
        RolloutSequenceDataset('datasets/mpe', SEQ_LEN, train=False, buffer_size=10),
        batch_size=BSIZE)
        

    def to_latent(obs, next_obs):
        """ Transform observations to latent space.
        :args obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)
        :args next_obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)
        :returns: (latent_obs, latent_next_obs)
            - latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
            - next_latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
        """
        with torch.no_grad():
            (obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma) = [
                vae(x)[1:] for x in (obs, next_obs)]

            latent_obs, latent_next_obs = [
                (x_mu + x_logsigma.exp() * torch.randn_like(x_mu)).view(BSIZE, SEQ_LEN, LSIZE)
                for x_mu, x_logsigma in
                [(obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma)]]
        return latent_obs, latent_next_obs

    def get_loss(latent_obs, action, reward, done,
                 latent_next_obs, include_reward: bool):
        """ Compute losses.
        The loss that is computed is:
        (GMMLoss(latent_next_obs, GMMPredicted) + MSE(reward, predicted_reward) +
             BCE(done, logit_terminal)) / (LSIZE + 2)
        The LSIZE + 2 factor is here to counteract the fact that the GMMLoss scales
        approximately linearily with LSIZE. All losses are averaged both on the
        batch and the sequence dimensions (the two first dimensions).
        :args latent_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor
        :args action: (BSIZE, SEQ_LEN, ASIZE) torch tensor
        :args reward: (BSIZE, SEQ_LEN) torch tensor
        :args latent_next_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor
        :returns: dictionary of losses, containing the gmm, the mse, the bce and
            the averaged loss.
        """
        latent_obs, action,\
            reward, done,\
            latent_next_obs = [arr.transpose(1, 0)
                               for arr in [latent_obs, action,
                                           reward, done,
                                           latent_next_obs]]
        mus, sigmas, logpi, rs, ds = mdrnn(action, latent_obs)
        gmm = gmm_loss(latent_next_obs, mus, sigmas, logpi)
        bce = f.binary_cross_entropy_with_logits(ds, done)
        if include_reward:
            mse = f.mse_loss(rs, reward)
            scale = LSIZE + 2
        else:
            mse = 0
            scale = LSIZE + 1
        loss = (gmm + bce + mse) / scale
        return dict(gmm=gmm, bce=bce, mse=mse, loss=loss)


    def data_pass(epoch, train, include_reward):
        """ One pass through the data """
        if train:
            mdrnn.train()
            loader = train_loader
        else:
            mdrnn.eval()
            loader = test_loader

        loader.dataset.load_next_buffer()

        cum_loss = 0
        cum_gmm = 0
        cum_bce = 0
        cum_mse = 0

        pbar = tqdm(total=len(loader.dataset), desc="Epoch {}".format(epoch))
        for i, data in enumerate(loader):
            obs, action, reward, done, next_obs = [arr.to(device) for arr in data]

            # transform obs
            latent_obs, latent_next_obs = to_latent(obs, next_obs)

            if train:
                losses = get_loss(latent_obs, action, reward,
                                  done, latent_next_obs, include_reward)

                optimizer.zero_grad()
                losses['loss'].backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    losses = get_loss(latent_obs, action, reward,
                                      done, latent_next_obs, include_reward)

            cum_loss += losses['loss'].item()
            cum_gmm += losses['gmm'].item()
            cum_bce += losses['bce'].item()
            cum_mse += losses['mse'].item() if hasattr(losses['mse'], 'item') else \
                losses['mse']

            pbar.set_postfix_str("loss={loss:10.6f} bce={bce:10.6f} "
                                 "gmm={gmm:10.6f} mse={mse:10.6f}".format(
                                     loss=cum_loss / (i + 1), bce=cum_bce / (i + 1),
                                     gmm=cum_gmm / LSIZE / (i + 1), mse=cum_mse / (i + 1)))
            pbar.update(BSIZE)
        pbar.close()
        return cum_loss * BSIZE / len(loader.dataset)
        
        
    train = partial(data_pass, train=True, include_reward=True)
    test = partial(data_pass, train=False, include_reward=True)

    cur_best = None
    epochs = 3
    for e in range(epochs):
        train(e)
        test_loss = test(e)

        is_best = not cur_best or test_loss < cur_best
        if is_best:
            cur_best = test_loss
        checkpoint_fname = join(rnn_dir, 'checkpoint.tar')
        save_checkpoint({
            "state_dict": mdrnn.state_dict(),
            "optimizer": optimizer.state_dict(),
            "precision": test_loss,
            "epoch": e}, is_best, checkpoint_fname,
                        rnn_file)