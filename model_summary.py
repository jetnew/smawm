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
from controller import Controller

mdir = 'exp_dir'
device = torch.device('cpu')




vae_file, rnn_file, ctrl_file = \
    [join(mdir, m, 'best.tar') for m in ['vae', 'mdrnn', 'ctrl']]
    
vae_state, rnn_state = [
    torch.load(fname, map_location={'cuda:0': str(device)})
    for fname in (vae_file, rnn_file)]
            
for m, s in (('VAE', vae_state), ('MDRNN', rnn_state)):
    print("Loading {} at epoch {} "
          "with test loss {}".format(m, s['epoch'], s['precision']))
              
              
vae = VAE(10, 15).to(device)
vae.load_state_dict(vae_state['state_dict'])

mdrnn = MDRNNCell(15, 15, 30, 5).to(device)
mdrnn.load_state_dict(
    {k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()})

controller = Controller(15, 30, 4).to(device)

if exists(ctrl_file):
    ctrl_state = torch.load(ctrl_file, map_location={'cuda:0': str(device)})
    print("Loading Controller with reward {}".format(
        ctrl_state['reward']))
    controller.load_state_dict(ctrl_state['state_dict'])
    
print("VAE:", vae)
print("MDRNN:", mdrnn)
print("Controller:", controller)