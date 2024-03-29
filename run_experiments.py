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
    # ===== WM (2434) vs SWM (1590) (Random Dataset) =====
    log = pd.DataFrame()
    for i in tqdm(range(10)):
        c = define_config()
        log = log.append(run_experiment(c, run_wm=True, run_swm=True), ignore_index=True)
        log.to_csv(f"experiments/wm2434-swm1590-random.csv", index=False)

    # ===== WM (2434) vs SWM (1590) (Spurious Dataset) =====
    log = pd.DataFrame()
    for i in tqdm(range(10)):
        c = define_config()
        c.setting = "spurious"
        log = log.append(run_experiment(c, run_wm=True, run_swm=True), ignore_index=True)
        log.to_csv(f"experiments/wm2434-swm1590-spurious.csv", index=False)

    # ===== WM (2434) vs SWM (1590) (Expert Dataset) =====
    log = pd.DataFrame()
    for i in tqdm(range(10)):
        c = define_config()
        c.setting = "expert"
        log = log.append(run_experiment(c, run_wm=True, run_swm=True), ignore_index=True)
        log.to_csv(f"experiments/wm2434-swm1590-expert.csv", index=False)

    # ===== SWM (2110) (Random Dataset) =====
    log = pd.DataFrame()
    for i in tqdm(range(10)):
        c = define_config()
        c.swm_agent_latent_dim = 10
        log = log.append(run_experiment(c, run_wm=False, run_swm=True), ignore_index=True)
        log.to_csv(f"experiments/swm2110-random.csv", index=False)

    # ===== SWM (2110) (Spurious Dataset) =====
    log = pd.DataFrame()
    for i in tqdm(range(10)):
        c = define_config()
        c.setting = "spurious"
        c.swm_agent_latent_dim = 10
        log = log.append(run_experiment(c, run_wm=False, run_swm=True), ignore_index=True)
        log.to_csv(f"experiments/swm2110-spurious.csv", index=False)

    # ===== SWM (2110) (Expert Dataset) =====
    log = pd.DataFrame()
    for i in tqdm(range(10)):
        c = define_config()
        c.setting = "expert"
        c.swm_agent_latent_dim = 10
        log = log.append(run_experiment(c, run_wm=False, run_swm=True), ignore_index=True)
        log.to_csv(f"experiments/swm2110-expert.csv", index=False)

    # ===== SWM (1850) (Random Dataset) =====
    log = pd.DataFrame()
    for i in tqdm(range(10)):
        c = define_config()
        c.swm_layers = 2
        log = log.append(run_experiment(c, run_wm=False, run_swm=True), ignore_index=True)
        log.to_csv(f"experiments/swm1850-random.csv", index=False)

    # ===== SWM (1850) (Spurious Dataset) =====
    log = pd.DataFrame()
    for i in tqdm(range(10)):
        c = define_config()
        c.setting = "spurious"
        c.swm_layers = 2
        log = log.append(run_experiment(c, run_wm=False, run_swm=True), ignore_index=True)
        log.to_csv(f"experiments/swm1850-spurious.csv", index=False)

    # ===== SWM (1850) (Expert Dataset) =====
    log = pd.DataFrame()
    for i in tqdm(range(10)):
        c = define_config()
        c.setting = "expert"
        c.swm_layers = 2
        log = log.append(run_experiment(c, run_wm=False, run_swm=True), ignore_index=True)
        log.to_csv(f"experiments/swm1850-expert.csv", index=False)