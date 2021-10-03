"""
Variational encoder model, used as a visual model
for our model of the world.
"""
from os.path import join, exists
from os import mkdir, getcwd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import optim
from loaders import RolloutObservationDataset
from misc import save_checkpoint


class Decoder(nn.Module):
    """ VAE decoder """
    def __init__(self, state_dim, latent_dim):
        super(Decoder, self).__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(latent_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, state_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Encoder(nn.Module):
    """ VAE encoder """
    def __init__(self, state_dim, latent_dim):
        super(Encoder, self).__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        
        self.fc1 = nn.Linear(state_dim, latent_dim)
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logsigma = nn.Linear(latent_dim, latent_dim)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)
        return mu, logsigma

class VAE(nn.Module):
    """ Variational Autoencoder """
    def __init__(self, state_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(state_dim, latent_dim)
        self.decoder = Decoder(state_dim, latent_dim)

    def forward(self, x):
        mu, logsigma = self.encoder(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)
        recon_x = self.decoder(z)
        return recon_x, mu, logsigma
        
        
def loss_function(recon_x, x, mu, logsigma):
    """ VAE loss function """
    BCE = F.mse_loss(recon_x, x, size_average=False)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    return BCE + KLD
    
    
def train(epoch):
    """ One training epoch """
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
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))
        
def test():
    """ One test epoch """
    model.eval()
    dataset_test.load_next_buffer()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss
        

if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if cuda else "cpu")

    dataset_train = RolloutObservationDataset('datasets/mpe', train=True)
    dataset_test = RolloutObservationDataset('datasets/mpe', train=False)


    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=32, shuffle=True)
        

    model = VAE(10, 15).to(device)
    optimizer = optim.Adam(model.parameters())


    logdir = 'exp_dir'
    vae_dir = join(getcwd(), logdir, 'vae')
    if not exists(vae_dir):
        mkdir(vae_dir)


    reload_file = join(vae_dir, 'best.tar')
    if exists(reload_file):
        state = torch.load(reload_file)
        print("Reloading model at epoch {}"
              ", with test error {}".format(
                  state['epoch'],
                  state['precision']))
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])


    cur_best = None
    epochs = 3
    for epoch in range(1, epochs + 1):
        train(epoch)
        test_loss = test()

        # checkpointing
        best_filename = join(vae_dir, 'best.tar')
        filename = join(vae_dir, 'checkpoint.tar')
        is_best = not cur_best or test_loss < cur_best
        if is_best:
            cur_best = test_loss

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'precision': test_loss,
            'optimizer': optimizer.state_dict()
        }, is_best, filename, best_filename)