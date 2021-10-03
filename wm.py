""" Test controller """
from os import getcwd
from os.path import join, exists
from controller import RolloutGenerator
import torch
from tqdm import tqdm
import statistics

logdir = 'exp_dir'

ctrl_file = join(getcwd(), logdir, 'ctrl', 'best.tar')

assert exists(ctrl_file),\
    "Controller was not trained..."

device = torch.device('cpu')

generator = RolloutGenerator(logdir, device, 1000)

with torch.no_grad():
    scores = []
    trials = 1000
    for _ in tqdm(range(trials)):
        score = -generator.rollout(None, render=False)
        scores.append(score)

print(f"Score: {sum(scores)/trials:.2f} +- {statistics.stdev(scores):.2f}")