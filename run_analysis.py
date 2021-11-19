from generate_dataset import *
from train_wm import *
from train_swm import *
from train_rl import *


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

def prediction_loss(model, observations, actions):
    state = model.obj_encoder(observations[0])
    losses = []
    for i in range(10):
        next_state_pred = state + model.transition_model(state, actions[i])
        next_obs_pred = model.decoder(next_state_pred)
        loss = F.mse_loss(next_obs_pred, obs[i+1], reduction='sum') / obs.size(0)
        losses.append(loss.item())
    return losses


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


def analyse_swm(
        setting,
        model_dir="models",
        data_dir="datasets"
):
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    data_dir = join(data_dir, setting, "episodes.h5")
    swm = torch.load(join(model_dir, setting, "swm.tar")).to(device)
    swm.eval()

    dataset = AnalysePredictionsDataset(hdf5_file=data_dir)
    data_loader = data.DataLoader(dataset, batch_size=32, shuffle=True)

    loss_1_step = 0
    loss_5_step = 0
    loss_10_step = 0
    for batch_idx, data_batch in enumerate(data_loader):
        data_batch = [tensor.to(device) for tensor in data_batch]
        losses = prediction_loss(swm, *data_batch)
        loss_1_step += losses[0]
        loss_5_step += losses[4]
        loss_10_step += losses[9]
    loss_1_step = loss_1_step / len(train_loader.dataset)
    loss_5_step = loss_5_step / len(train_loader.dataset)
    loss_10_step = loss_10_step / len(train_loader.dataset)
    print(f"1-Step: {loss_1_step:.3f} 5-Step: {loss_1_step:.3f} 10-Step: {loss_1_step:.3f}")
    return loss_1_step, loss_5_step, loss_10_step


if __name__ == "__main__":
    c = define_config()
    train_vae(
        setting=c.setting,
        latent_dim=c.wm_spatial_latent_dim,
        n_hidden=c.wm_hidden,
        n_layers=c.wm_layers,
        epochs=c.wm_epochs)
    train_mdrnn(
        setting=c.setting,
        spatial_latent_dim=c.wm_spatial_latent_dim,
        temporal_latent_dim=c.wm_temporal_latent_dim,
        n_gaussians=c.wm_gaussians,
        epochs=c.wm_epochs)
    train_swm(
        setting=c.setting,
        agent_latent_dim=c.swm_agent_latent_dim,
        n_hidden=c.swm_hidden,
        n_layers=c.swm_layers,
        epochs=c.swm_epochs)

