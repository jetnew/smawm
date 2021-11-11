# RUN WITH python smawm_experiments.py > experiment_results.txt
import warnings
warnings.filterwarnings("ignore")

from vae import vae_experiment
from mdrnn import mdrnn_experiment
from env import evaluate_experiment
from swm import swm_experiment
import tools
from model_summary import model_summary

def print_config(config):
    s = ""
    for p, v in config.items():
        s += f"{p}={v}  "
    print(s)

def define_config():
    config = tools.AttrDict()
    
    # Experiment
    config.ddir = 'datasets'
    config.data_dir = config.ddir + '/mpe'
    config.exp_dir = 'exp_dir'
    config.input_dim = 10
    config.n_agents = 3
    config.action_dim = 5
    config.input_dim = 10
    
    # WM
    config.vae_dim = 15
    config.vae_epochs = 10
    config.vae_dim = 15
    config.vae_epochs = 10
    config.mdrnn_dim = 30
    config.mdrnn_epochs = 10
    
    # SWM    
    config.swm_latent_dim = 15
    config.swm_hidden_dim = 32
    config.swm_epochs = 3
    
    # Evaluation
    config.exp_dir = 'exp_dir'
    config.policy = 'ppo'  # 'ppo' or 'a2c'
    config.adversary_eps = 0.5
    config.agent_eps = 0.5
    config.train_timesteps = 20_000
    config.eval_episodes = 100
    return config
    
def run_experiment(params):
    # params = {'param': [param1, param2, ...]}
    config = define_config()
    for p in params:
        config[p] = params[p]
    print_config(config)
    #vae_experiment(config)
    #mdrnn_experiment(config)
    #swm_experiment(config)
    model_summary(config)
    evaluate_experiment(config)

run_experiment({'vae_epochs':1, 'swm_epochs': 1, 'mdrnn_epochs': 1, 'train_timesteps': 2000})
#run_experiment({'data_dir': ['a', 'b', 'c']})
