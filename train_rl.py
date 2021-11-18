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
from train_wm import VAE, MDRNNCell, Encoder, Decoder, MDRNN
from train_swm import ContrastiveSWM, EncoderMLP, TransitionGNN
import os

import tools


class SimpleAdversaryEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, world_model):
        super(SimpleAdversaryEnv, self).__init__()
        self.adversary_policy = follow_agent_closest_to_landmark_policy
        self.agent_policy = follow_non_goal_landmark_policy
        self.world_model = world_model
        self.adversary_eps = 0.2
        self.agent_eps = 0.2
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(self.world_model.latent_dim,), dtype=np.float16)
        self._env = simple_adversary_v2.parallel_env(N=2, max_cycles=100, continuous_actions=False)
        self.observations = self._env.reset()
        self.actions_taken = None
    def step(self, action):
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
        features = self.world_model.forward(self.observations['agent_1'])
        return features, rewards['agent_1'], dones['agent_1'], infos['agent_1']
    def reset(self):
        self.actions_taken = None
        self.observations = self._env.reset()
        return self.world_model.forward(self.observations['agent_1'])
    def render(self, mode='human'):
        return self._env.render(mode)
    def close(self):
        return self._env.close()

class WorldModel:
    def __init__(self, wm, setting, model_dir="models"):
        self.wm = wm
        self.device = torch.device('cpu')
        if wm == "wm":
            self.vae = torch.load(join(model_dir, setting, "vae.tar")).to(self.device)
            self.vae.eval()
            self.mdrnn = torch.load(join(model_dir, setting, "mdrnn.tar")).to(self.device)
            self.mdrnn.eval()
            self.latent_dim = self.vae.latent_dim + self.mdrnn.latent_dim
            self.temporal_latent = [torch.zeros(1, self.mdrnn.latent_dim).to(self.device) for _ in range(2)]
        elif wm == "swm":
            self.swm = torch.load(join(model_dir, setting, "swm.tar")).to(self.device)
            self.swm.eval()
            self.latent_dim = self.swm.latent_dim
    def forward(self, obs):
        obs = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        if self.wm == "wm":
            _, self.spatial_latent, _ = self.vae(obs)
            return torch.cat((self.spatial_latent, self.temporal_latent[0]), dim=1).detach().numpy()
        elif self.wm == "swm":
            obs = self.swm.forward(obs)
            return torch.flatten(obs, start_dim=1).detach().numpy()
    def update(self, actions):
        if self.wm == "wm":
            if actions is None:
                self.temporal_latent = [torch.zeros(1, self.mdrnn.latent_dim).to(self.device)
                                        for _ in range(2)]
            else:
                actions = [torch.eye(5, device=self.device)[a] for a in actions]
                actions = torch.cat(actions, 0).unsqueeze(0)
                _, _, _, _, self.temporal_latent = self.mdrnn(actions, self.spatial_latent, self.temporal_latent)


def train_rl(
        world_model,
        setting,
        agent,
        train_timesteps=50_000,
        eval_episodes=100,
        model_dir="models",
        verbose=False):
    env = SimpleAdversaryEnv(WorldModel(world_model, setting))
    if agent == "ppo":
        model = PPO("MlpPolicy", env, verbose=0)
    elif agent == "a2c":
        model = A2C("MlpPolicy", env, verbose=0)
    else:
        raise Exception(f"Model {agent} not available.")
    model_dir = join(model_dir, setting)
    if verbose:
        print(f"Training {world_model}-{agent} at: {model_dir}")
    model.learn(total_timesteps=train_timesteps)
    mean, std = evaluate_policy(model, model.get_env(), n_eval_episodes=eval_episodes)
    if verbose:
        print(f"Saved {world_model}-{agent} at {model_dir}: {mean:.4f}")
    model.save(join(model_dir, f"{world_model}-{agent}"))

    return {
        f"{world_model}_agent_reward": mean
    }


if __name__ == "__main__":
    train_rl(world_model="wm", setting="random", agent="ppo")
    train_rl(world_model="swm", setting="random", agent="ppo")
    train_rl(world_model="wm", setting="spurious", agent="ppo")
    train_rl(world_model="swm", setting="spurious", agent="ppo")
    train_rl(world_model="wm", setting="expert", agent="ppo")
    train_rl(world_model="swm", setting="expert", agent="ppo")