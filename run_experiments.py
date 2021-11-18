from generate_dataset import *
from train_wm import *
from train_swm import *
from train_rl import *
import pandas as pd
from tools import set_seed, AttrDict
from tqdm import tqdm


def define_config():
    c = AttrDict()
    # Dataset config
    c.setting = "random"  # ['random', 'spurious', 'expert']
    # WM config
    c.wm_spatial_latent_dim = 10  # [5,10,15,20,25,30]
    c.wm_temporal_latent_dim = 5  # [5,10,15,20,25,30]
    c.wm_hidden = 5  # [5,10,15,20]
    c.wm_layers = 1  # [1,2,3]
    c.wm_gaussians = 3  # [1,3,5]
    c.wm_epochs = 1  # [1,2,3,4,5]
    # SWM config
    c.swm_agent_latent_dim = 5  # [5,6,7,8,9,10]
    c.swm_hidden = 5  # [5,10,15,20]
    c.swm_layers = 1  # [1,2,3]
    c.wm_epochs = 1  # [1,2,3,4,5]
    # Agent config
    c.agent = 'ppo'  # ['ppo', 'a2c']
    return c

def run_experiment(c, run_wm=False, run_swm=False):
    generate_dataset(setting=c.setting)
    wm_log = {}
    swm_log = {}
    if run_wm:
        vae_log = train_vae(
            setting=c.setting,
            latent_dim=c.wm_spatial_latent_dim,
            n_hidden=c.wm_hidden,
            n_layers=c.wm_layers,
            epochs=c.wm_epochs)
        mdrnn_log = train_mdrnn(
            setting=c.setting,
            spatial_latent_dim=c.wm_spatial_latent_dim,
            temporal_latent_dim=c.wm_temporal_latent_dim,
            n_gaussians=c.wm_gaussians,
            epochs=c.wm_epochs)
        wmrl_log = train_rl(
            world_model="wm",
            setting=c.setting,
            agent=c.agent)
        wm_log = {**vae_log, **mdrnn_log, **wmrl_log}
    if run_swm:
        swm_log = train_swm(
            setting=c.setting,
            agent_latent_dim=c.swm_agent_latent_dim,
            n_hidden=c.swm_hidden,
            n_layers=c.swm_layers,
            epochs=c.swm_epochs)
        swmrl_log = train_rl(
            world_model="swm",
            setting=c.setting,
            agent=c.agent)
        swm_log = {**swm_log, **swmrl_log}
    return {**c, **wm_log, **swm_log}


if __name__ == "__main__":
    log = pd.DataFrame()
    seeds = 10
    for i in tqdm(range(seeds)):
        # Experiment: Tune WM
        for wm_spatial_latent_dim in [5,10,15]:
            c = define_config()
            c.wm_spatial_latent_dim = wm_spatial_latent_dim
            log = log.append(run_experiment(c, run_wm=True), ignore_index=True)
        # for wm_temporal_latent_dim in [5,10,15]:
        #     c = define_config()
        #     c.wm_temporal_latent_dim = wm_temporal_latent_dim
        #     log = log.append(run_experiment(c, run_wm=True), ignore_index=True)
        # for wm_hidden in [5,10,15]:
        #     c = define_config()
        #     c.wm_hidden = wm_hidden
        #     log = log.append(run_experiment(c, run_wm=True), ignore_index=True)
        # for wm_layers in [1,2]:
        #     c = define_config()
        #     c.wm_layers = wm_layers
        #     log = log.append(run_experiment(c, run_wm=True), ignore_index=True)
        # for wm_gaussians in [1,2,3]:
        #     c = define_config()
        #     c.wm_gaussians = wm_gaussians
        #     log = log.append(run_experiment(c, run_wm=True), ignore_index=True)
        # for wm_epochs in [1,2,3]:
        #     c = define_config()
        #     c.wm_epochs = wm_epochs
        #     log = log.append(run_experiment(c, run_wm=True), ignore_index=True)
        log.to_csv(f"experiments/wm_spatial_latent_dim.csv", index=False)