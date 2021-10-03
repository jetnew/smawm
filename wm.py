""" Test controller """
from os import getcwd
from os.path import join, exists
from controller import RolloutGenerator
import torch

logdir = 'exp_dir'

ctrl_file = join(getcwd(), logdir, 'ctrl', 'best.tar')

assert exists(ctrl_file),\
    "Controller was not trained..."

device = torch.device('cpu')

generator = RolloutGenerator(logdir, device, 1000)

with torch.no_grad():
    generator.rollout(None, render=True)