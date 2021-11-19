# Multi-step prediction error analysis cannot be performed yet because multi-step prediction requires predicting actions to be taken over multiple timesteps. Need to implement action prediction first.

# from generate_dataset import *
# from train_wm import *
# from train_swm import *
# from train_rl import *
#
#
# def define_config():
#     c = AttrDict()
#     # Dataset config
#     c.setting = "random"  # ['random', 'spurious', 'expert']
#     # WM config
#     c.wm_spatial_latent_dim = 5
#     c.wm_temporal_latent_dim = 10
#     c.wm_hidden = 10
#     c.wm_layers = 1
#     c.wm_gaussians = 3
#     c.wm_epochs = 1
#     # SWM config
#     c.swm_agent_latent_dim = 5
#     c.swm_hidden = 10
#     c.swm_layers = 1
#     c.swm_epochs = 1
#     # Agent config
#     c.agent = 'ppo'  # ['ppo', 'a2c']
#     return c
#
#
# def analyse_swm(
#         setting,
#         model_dir="models",
#         data_dir="datasets"
# ):
#     cuda = torch.cuda.is_available()
#     device = torch.device('cuda' if cuda else 'cpu')
#     data_dir = join(data_dir, setting, "episodes.h5")
#     swm = torch.load(join(model_dir, setting, "swm.tar")).to(device)
#     swm.eval()
#
#     dataset = StateTransitionsDataset(hdf5_file=data_dir)
#     train_loader = data.DataLoader(dataset, batch_size=32, shuffle=True)
#     model = ContrastiveSWM(agent_latent_dim, n_hidden, n_layers).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
#     param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
#
#     train_loss = 0
#     for batch_idx, data_batch in enumerate(train_loader):
#         data_batch = [tensor.to(device) for tensor in data_batch]
#     avg_loss = train_loss / len(train_loader.dataset)
#
#
# if __name__ == "__main__":
#     c = define_config()
#     train_vae(
#         setting=c.setting,
#         latent_dim=c.wm_spatial_latent_dim,
#         n_hidden=c.wm_hidden,
#         n_layers=c.wm_layers,
#         epochs=c.wm_epochs)
#     train_mdrnn(
#         setting=c.setting,
#         spatial_latent_dim=c.wm_spatial_latent_dim,
#         temporal_latent_dim=c.wm_temporal_latent_dim,
#         n_gaussians=c.wm_gaussians,
#         epochs=c.wm_epochs)
#     train_swm(
#         setting=c.setting,
#         agent_latent_dim=c.swm_agent_latent_dim,
#         n_hidden=c.swm_hidden,
#         n_layers=c.swm_layers,
#         epochs=c.swm_epochs)
#
