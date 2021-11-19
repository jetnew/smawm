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
    # # ===== WM (2264) vs SWM (1150) (Random Dataset) =====
    # log = pd.DataFrame()
    # for i in tqdm(range(10)):
    #     c = define_config()
    #     log = log.append(run_experiment(c, run_wm=True, run_swm=True), ignore_index=True)
    #     log.to_csv(f"experiments/wm2264-swm1150-random.csv", index=False)

    # ===== WM (2264) vs SWM (1150) (Spurious Dataset) =====
    # log = pd.DataFrame()
    # for i in tqdm(range(10)):
    #     c = define_config()
    #     c.setting = "spurious"
    #     log = log.append(run_experiment(c, run_wm=True, run_swm=True), ignore_index=True)
    #     log.to_csv(f"experiments/wm2264-swm1150-spurious.csv", index=False)

    # # ===== WM (2264) vs SWM (1150) (Expert Dataset) =====
    # log = pd.DataFrame()
    # for i in tqdm(range(10)):
    #     c = define_config()
    #     c.setting = "expert"
    #     log = log.append(run_experiment(c, run_wm=True, run_swm=True), ignore_index=True)
    #     log.to_csv(f"experiments/wm2264-swm1150-expert.csv", index=False)
    #
    # # ===== SWM (2090) (Random Dataset) =====
    # log = pd.DataFrame()
    # for i in tqdm(range(10)):
    #     c = define_config()
    #     c.swm_hidden = 15
    #     c.setting = "random"
    #     log = log.append(run_experiment(c, run_swm=True), ignore_index=True)
    #     log.to_csv(f"experiments/swm2090-random.csv", index=False)
    #
    # # ===== SWM (2090) (Spurious Dataset) =====
    # log = pd.DataFrame()
    # for i in tqdm(range(10)):
    #     c = define_config()
    #     c.swm_hidden = 15
    #     c.setting = "spurious"
    #     log = log.append(run_experiment(c, run_swm=True), ignore_index=True)
    #     log.to_csv(f"experiments/swm2090-spurious.csv", index=False)

    # ===== SWM (2090) (Expert Dataset) =====
    log = pd.DataFrame()
    for i in tqdm(range(10)):
        c = define_config()
        c.swm_hidden = 15
        c.setting = "expert"
        log = log.append(run_experiment(c, run_swm=True), ignore_index=True)
        log.to_csv(f"experiments/swm2090-expert.csv", index=False)