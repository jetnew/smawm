from generate_dataset import *
from train_wm import *
from train_swm import *
from train_rl import *
from tools import AttrDict
import pandas as pd


def define_config():
    c = AttrDict()
    # Dataset config
    c.setting = "random"  # ['random', 'spurious', 'expert']
    # WM config
    c.wm_spatial_latent_dim = 5
    c.wm_temporal_latent_dim = 10
    c.wm_hidden = 10
    c.wm_layers = 1
    c.wm_gaussians = 3
    c.wm_epochs = 1
    # SWM config
    c.swm_agent_latent_dim = 5
    c.swm_hidden = 10
    c.swm_layers = 1
    c.swm_epochs = 1
    # Agent config
    c.agent = 'ppo'  # ['ppo', 'a2c']
    return c


class AnalysePredictionsDataset(data.Dataset):
    def __init__(self, hdf5_file):
        self.experience_buffer = load_list_dict_h5py(hdf5_file)
        self.idx2episode = list()
        step = 0
        for ep in range(len(self.experience_buffer)):
            num_steps = len(self.experience_buffer[ep]['action_t0'])
            idx_tuple = [(ep, idx) for idx in range(num_steps)]
            self.idx2episode.extend(idx_tuple)
            step += num_steps
        self.num_steps = step
    def __len__(self):
        return self.num_steps
    def __getitem__(self, idx):
        ep, step = self.idx2episode[idx]
        observations = [to_float(self.experience_buffer[ep][f'obs_t{t}'][step]) for t in range(11)]
        actions = [self.experience_buffer[ep][f'action_t{t}'][step] for t in range(10)]
        return observations, actions

def swm_prediction_loss(model, obs, actions):
    state = model.obj_encoder(obs[0])
    losses = []
    for i in range(10):
        state = state + model.transition_model(state, actions[i])
        obs_pred = model.decoder(state)
        loss = F.mse_loss(obs_pred, obs[i+1], reduction='sum') / obs[0].size(0)
        losses.append(loss.item())
    return np.array(losses)


def wm_prediction_loss(vae, mdrnn, obs, actions, device):
    temporal_latent = [torch.zeros(obs[0].size(0), mdrnn.latent_dim).to(device) for _ in range(2)]
    _, spatial_latent, _ = vae(obs[0])
    losses = []
    for i in range(10):
        action = to_one_hot(actions[i], 15)
        mu, sigma, logpi, _, temporal_latent = mdrnn(action, spatial_latent, temporal_latent)
        mix = Categorical(torch.exp(logpi))
        comp = Independent(Normal(mu, sigma), 1)
        gmm = MixtureSameFamily(mix, comp)
        spatial_latent = gmm.sample()
        obs_pred = vae.decoder(spatial_latent)
        loss = F.mse_loss(obs_pred, obs[i+1], reduction='sum') / obs[0].size(0)
        losses.append(loss.item())
    return np.array(losses)


def analyse_swm(
        setting,
        model_dir="models",
        data_dir="datasets",
        analysis_dir="analysis"
):
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    data_dir = join(data_dir, setting, "predictions.h5")
    csv_dir = join(analysis_dir, setting, "predictions-swm.csv")
    swm = torch.load(join(model_dir, setting, "swm.tar")).to(device)
    swm.eval()

    dataset = AnalysePredictionsDataset(hdf5_file=data_dir)
    data_loader = data.DataLoader(dataset, batch_size=32, shuffle=True)

    losses = np.zeros(10)
    for batch_idx, data_batch in enumerate(data_loader):
        obs, actions = data_batch
        obs = [tensor.to(device) for tensor in obs]
        actions = [tensor.to(device) for tensor in actions]
        loss = swm_prediction_loss(swm, obs, actions)
        losses += loss
    losses /= len(data_loader.dataset)
    print(f"SWM Setting: '{setting}'  1-Step: {losses[0]:.3f}  5-Step: {losses[5]:.3f}  10-Step: {losses[9]:.3f}")
    pd.DataFrame({"Loss": losses}).to_csv(csv_dir, index=False)
    return


def analyse_wm(
        setting,
        model_dir="models",
        data_dir="datasets",
        analysis_dir="analysis"
):
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    data_dir = join(data_dir, setting, "predictions.h5")
    csv_dir = join(analysis_dir, setting, "predictions-wm.csv")
    vae = torch.load(join(model_dir, setting, "vae.tar")).to(device)
    vae.eval()
    mdrnn = torch.load(join(model_dir, setting, "mdrnn.tar")).to(device)
    mdrnn.eval()

    dataset = AnalysePredictionsDataset(hdf5_file=data_dir)
    data_loader = data.DataLoader(dataset, batch_size=32, shuffle=True)

    losses = np.zeros(10)
    for batch_idx, data_batch in enumerate(data_loader):
        obs, actions = data_batch
        obs = [tensor.to(device) for tensor in obs]
        actions = [tensor.to(device) for tensor in actions]
        loss = wm_prediction_loss(vae, mdrnn, obs, actions, device)
        losses += loss
    losses /= len(data_loader.dataset)
    print(f"WM Setting: '{setting}'  1-Step: {losses[0]:.3f}  5-Step: {losses[5]:.3f}  10-Step: {losses[9]:.3f}")
    pd.DataFrame({"Loss": losses}).to_csv(csv_dir, index=False)
    return


if __name__ == "__main__":
    c = define_config()
    analyse_swm(setting="random")
    analyse_swm(setting="spurious")
    analyse_swm(setting="expert")
    analyse_wm(setting="random")
    analyse_wm(setting="spurious")
    analyse_wm(setting="expert")
