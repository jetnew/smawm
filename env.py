from tqdm import tqdm
import gym
import numpy as np
from gym import spaces
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from pettingzoo.mpe import simple_adversary_v2
from policies import static_policy, random_policy, follow, follow_non_goal_landmark_policy, follow_agent_closest_to_landmark_policy, compute_reward

import torch
import torch.nn as nn
import sys
from os.path import join, exists
from os import mkdir, unlink, listdir, getpid, getcwd
import torch
import numpy as np
from misc import load_parameters, flatten_parameters, save_checkpoint
from vae import VAE
from mdrnn import MDRNNCell
import swm
import os


class SimpleAdversaryEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, adversary_policy, agent_policy, world_model=None, adversary_eps=0.5, agent_eps=0.5, time_limit=100):
        super(SimpleAdversaryEnv, self).__init__()
        self.adversary_policy = adversary_policy
        self.agent_policy = agent_policy
        self.world_model = world_model
        self.adversary_eps = adversary_eps
        self.agent_eps = agent_eps
        self.action_space = spaces.Discrete(5)
        shape = 10 if world_model is None else 45
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(shape,), dtype=np.float16)
        self._env = simple_adversary_v2.parallel_env(N=2, max_cycles=time_limit, continuous_actions=False)
        self.observations = self._env.reset()
        self.actions_taken = None
    def step(self, action):
        if isinstance(self.world_model, WM):
            self.world_model.update(self.actions_taken)
        actions = {}
        for agent in self._env.agents:
            if agent == 'adversary_0':
                actions[agent] = self.adversary_policy(self.observations[agent], eps=self.adversary_eps)
            elif self.agent_policy is not None and agent == 'agent_0':
                actions[agent] = self.agent_policy(self.observations[agent], eps=self.agent_eps)
            else:
                actions[agent] = action
        self.observations, rewards, dones, infos = self._env.step(actions)
        self.actions_taken = [actions[agent] for agent in ['adversary_0', 'agent_0', 'agent_1']]
        if self.world_model:
            features = self.world_model.forward(self.observations['agent_1'])
        else:
            features = self.observations['agent_1']
        return features, rewards['agent_1'], dones['agent_1'], infos['agent_1']
    def reset(self):
        self.actions_taken = None
        self.observations = self._env.reset()
        if self.world_model:
            features = self.world_model.forward(self.observations['agent_1'])
        else:
            features = self.observations['agent_1']
        return features
    def render(self, mode='human'):
        return self._env.render(mode)
    def close(self):
        return self._env.close()
        
        
class WM:
    def __init__(self, mdir='exp_dir', INPUT_DIM=10, LATENT_DIM=15, RSIZE=30, GAUSSIANS=5, n_agents=3, action_dim=5):
        self.RSIZE = RSIZE
        self.device = torch.device('cpu')
        vae_file, rnn_file = \
            [join(mdir, m, 'best.tar') for m in ['vae', 'mdrnn']]
            
        vae_state, rnn_state = [
            torch.load(fname, map_location={'cuda:0': str(self.device)})
            for fname in (vae_file, rnn_file)]
                    
                  
        self.vae = VAE(INPUT_DIM, LATENT_DIM).to(self.device)
        self.vae.load_state_dict(vae_state['state_dict'])

        self.mdrnn = MDRNNCell(LATENT_DIM, n_agents*action_dim, RSIZE, GAUSSIANS).to(self.device)  # 15 refers to 3 objects * action_dim=5
        self.mdrnn.load_state_dict(
            {k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()})
            
        self.hidden = [torch.zeros(1, RSIZE).to(self.device) for _ in range(2)]
    def forward(self, obs):
        obs = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        _, self.latent_mu, _ = self.vae(obs)
        return torch.cat((self.latent_mu, self.hidden[0]), dim=1).detach().numpy()
    def update(self, actions):
        if actions is None:
            self.hidden = [torch.zeros(1, self.RSIZE).to(self.device) for _ in range(2)]
        else:
            actions = [torch.eye(5, device=self.device)[a] for a in actions]
            actions = torch.cat(actions, 0).unsqueeze(0)
            _, _, _, _, _, self.hidden = self.mdrnn(actions, self.latent_mu, self.hidden)
            
            
class SWM:
    def __init__(self, mdir='exp_dir', embedding_dim=15, hidden_dim=32, action_dim=5, input_shape=10, num_objects=3):
        self.device = torch.device('cpu')
        sigma = 0.5
        hinge = 1.0
        ignore_action = False
        copy_action = False
        use_encoder = 'small'
        self.model = swm.ContrastiveSWM(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            input_dims=input_shape,
            num_objects=num_objects,
            sigma=sigma,
            hinge=hinge,
            ignore_action=ignore_action,
            copy_action=copy_action,
            encoder=use_encoder).to(self.device)
            
        save_folder = os.path.join(mdir, 'swm')
        model_file = os.path.join(save_folder, 'model.pt')
        self.model.load_state_dict(torch.load(model_file, map_location={'cuda:0': 'cpu'}))
        self.model.eval()
    def forward(self, obs):
        obs = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        obs = self.model.forward(obs)
        return torch.flatten(obs, start_dim=1).detach().numpy()
        
        
def evaluate_experiment(config):    
    if config.use_wm:
        world_model = WM(
            mdir=config.exp_dir,
            INPUT_DIM=config.input_dim,
            LATENT_DIM=config.vae_dim,
            RSIZE=config.mdrnn_dim,
            GAUSSIANS=5,
            n_agents=config.n_agents,
            action_dim=config.action_dim
        )

        env = SimpleAdversaryEnv(
            adversary_policy=follow_agent_closest_to_landmark_policy,
            agent_policy=follow_non_goal_landmark_policy,
            world_model=world_model,
            adversary_eps=config.adversary_eps,
            agent_eps=config.agent_eps)

        if config.policy == 'ppo':
            model = PPO("MlpPolicy", env, verbose=0)
        elif config.policy == 'a2c':
            model = A2C("MlpPolicy", env, verbose=0)
        
        model.learn(total_timesteps=config.train_timesteps)
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=config.eval_episodes)
        print(f"WM: {mean_reward:.3f} +- {std_reward:.3f}")
    
    if config.use_swm:
        world_model = SWM(
            mdir=config.exp_dir,
            embedding_dim=config.swm_latent_dim,
            hidden_dim=config.swm_hidden_dim,
            action_dim=config.action_dim,
            input_shape=config.input_dim,
            num_objects=config.n_agents
        )
        
        env = SimpleAdversaryEnv(
            adversary_policy=follow_agent_closest_to_landmark_policy,
            agent_policy=follow_non_goal_landmark_policy,
            world_model=world_model,
            adversary_eps=config.adversary_eps,
            agent_eps=config.agent_eps)

        if config.policy == 'ppo':
            model = PPO("MlpPolicy", env, verbose=0)
        elif config.policy == 'a2c':
            model = A2C("MlpPolicy", env, verbose=0)
        
        model.learn(total_timesteps=config.train_timesteps)
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=config.eval_episodes)
        print(f"SWM: {mean_reward:.3f} +- {std_reward:.3f}")


if __name__ == "__main__":
    #world_model = WM()
    world_model = SWM()

    env = SimpleAdversaryEnv(
        adversary_policy=follow_agent_closest_to_landmark_policy,
        agent_policy=follow_non_goal_landmark_policy,
        world_model=world_model,
        adversary_eps=0.5,
        agent_eps=0.5)

    model = PPO("MlpPolicy", env, verbose=1)
    #model = A2C("MlpPolicy", env, verbose=0)
    
    model.learn(total_timesteps=20_000)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100)
    print(f"PPO Reward: {mean_reward:.3f} +- {std_reward:.3f}")