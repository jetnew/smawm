from tqdm import tqdm
import gym
import numpy as np
from gym import spaces
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from pettingzoo.mpe import simple_adversary_v2
from policies import static_policy, random_policy, follow, follow_non_goal_landmark_policy, follow_agent_closest_to_landmark_policy, compute_reward


class SimpleAdversaryEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, adversary_policy, agent_policy=None, adversary_eps=0.5, agent_eps=0.5, time_limit=100):
        super(SimpleAdversaryEnv, self).__init__()
        self.adversary_policy = adversary_policy
        self.agent_policy = agent_policy
        self.adversary_eps = adversary_eps
        self.agent_eps = agent_eps
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(10,), dtype=np.float16)
        self._env = simple_adversary_v2.parallel_env(N=2, max_cycles=time_limit, continuous_actions=False)
        self.observations = self._env.reset()
    def step(self, action):
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
        return self.observations['agent_1'], rewards['agent_1'], dones['agent_1'], infos['agent_1']
    def reset(self):
        self.observations = self._env.reset()
        return self.observations['agent_1']
    def render(self, mode='human'):
        return self._env.render(mode)
    def close(self):
        return self._env.close()


if __name__ == "__main__":
    for ad, ag in [(0,0), (0.5,0), (0,0.5), (0.5,0.5)]:
        env = SimpleAdversaryEnv(
            adversary_policy=follow_agent_closest_to_landmark_policy,
            agent_policy=follow_non_goal_landmark_policy,
            adversary_eps=ad,
            agent_eps=ag)

        model = PPO("MlpPolicy", env, verbose=0)
        #model = A2C("MlpPolicy", env, verbose=0)
        #model = DQN("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=20_000)
        model.save(f"baselines/ppo-ad{ad}-ag{ag}")
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1000)
        print(f"PPO Reward for ad{ad}-ag{ag}: {mean_reward:.3f} +- {std_reward:.3f}")
        
    for ad, ag in [(0,0), (0.5,0), (0,0.5), (0.5,0.5)]:
        env = SimpleAdversaryEnv(
            adversary_policy=follow_agent_closest_to_landmark_policy,
            agent_policy=follow_non_goal_landmark_policy,
            adversary_eps=ad,
            agent_eps=ag)

        model = A2C("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=20_000)
        model.save(f"baselines/a2c-ad{ad}-ag{ag}")
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1000)
        print(f"A2C Reward for ad{ad}-ag{ag}: {mean_reward:.3f} +- {std_reward:.3f}")
        
    for ad, ag in [(0,0), (0.5,0), (0,0.5), (0.5,0.5)]:
        env = SimpleAdversaryEnv(
            adversary_policy=follow_agent_closest_to_landmark_policy,
            agent_policy=follow_non_goal_landmark_policy,
            adversary_eps=ad,
            agent_eps=ag)

        model = DQN("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=20_000)
        model.save(f"baselines/dqn-ad{ad}-ag{ag}")
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1000)
        print(f"DQN Reward for ad{ad}-ag{ag}: {mean_reward:.3f} +- {std_reward:.3f}")