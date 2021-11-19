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
        obs_t0 = to_float(self.experience_buffer[ep]['obs_t0'][step])
        action_t0 = self.experience_buffer[ep]['action_t0'][step]
        obs_t1 = to_float(self.experience_buffer[ep]['obs_t1'][step])
        action_t1 = self.experience_buffer[ep]['action_t1'][step]
        obs_t2 = to_float(self.experience_buffer[ep]['obs_t2'][step])
        action_t2 = self.experience_buffer[ep]['action_t2'][step]
        obs_t3 = to_float(self.experience_buffer[ep]['obs_t3'][step])
        action_t3 = self.experience_buffer[ep]['action_t3'][step]
        obs_t4 = to_float(self.experience_buffer[ep]['obs_t4'][step])
        action_t4 = self.experience_buffer[ep]['action_t4'][step]
        obs_t5 = to_float(self.experience_buffer[ep]['obs_t5'][step])
        action_t5 = self.experience_buffer[ep]['action_t5'][step]
        obs_t6 = to_float(self.experience_buffer[ep]['obs_t6'][step])
        action_t6 = self.experience_buffer[ep]['action_t6'][step]
        obs_t7 = to_float(self.experience_buffer[ep]['obs_t7'][step])
        action_t7 = self.experience_buffer[ep]['action_t7'][step]
        obs_t8 = to_float(self.experience_buffer[ep]['obs_t8'][step])
        action_t8 = self.experience_buffer[ep]['action_t8'][step]
        obs_t9 = to_float(self.experience_buffer[ep]['obs_t9'][step])
        action_t9 = self.experience_buffer[ep]['action_t9'][step]
        obs_t10 = to_float(self.experience_buffer[ep]['obs_t10'][step])
        return (obs_t0,
                action_t0,
                obs_t1,
                action_t1,
                obs_t2,
                action_t2,
                obs_t3,
                action_t3,
                obs_t4,
                action_t4,
                obs_t5,
                action_t5,
                obs_t6,
                action_t6,
                obs_t7,
                action_t7,
                obs_t8,
                action_t8,
                obs_t9,
                action_t9,
                obs_t10)


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

    dataset = StateTransitionsDataset(hdf5_file=data_dir)
    train_loader = data.DataLoader(dataset, batch_size=32, shuffle=True)
    model = ContrastiveSWM(agent_latent_dim, n_hidden, n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    train_loss = 0
    for batch_idx, data_batch in enumerate(train_loader):
        data_batch = [tensor.to(device) for tensor in data_batch]
    avg_loss = train_loss / len(train_loader.dataset)


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

