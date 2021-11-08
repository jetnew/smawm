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
import swm
import os


print("===== WM =====")
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
    
vae_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
mdrnn_params = sum(p.numel() for p in mdrnn.parameters() if p.requires_grad)
controller_params = sum(p.numel() for p in controller.parameters() if p.requires_grad)

print("VAE trainable params:", vae_params, vae)
print("MDRNN trainable params:", mdrnn_params, mdrnn)
print("Controller trainable params:", controller_params, controller)
print("Total trainable params:", vae_params + mdrnn_params + controller_params)


# ====== SWM =====
print("===== SWM =====")
from swm_controller import Controller
mdir = 'exp_dir_swm'
ctrl_file = join(mdir, 'ctrl', 'best.tar')

embedding_dim = 15
hidden_dim = 10
action_dim = 5
input_shape = 10
num_objects = 3
sigma = 0.5
hinge = 1.0
ignore_action = False
copy_action = False
use_encoder = 'small'
model = swm.ContrastiveSWM(
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    action_dim=action_dim,
    input_dims=input_shape,
    num_objects=num_objects,
    sigma=sigma,
    hinge=hinge,
    ignore_action=ignore_action,
    copy_action=copy_action,
    encoder=use_encoder).to(device)
    
#save_folder = "checkpoints"
#model_file = os.path.join(save_folder, 'model.pt')
#model.load_state_dict(torch.load(model_file, map_location={'cuda:0': 'cpu'}))
#model.eval()

controller = Controller(num_objects, embedding_dim, 3).to(device)

#if exists(ctrl_file):
#    ctrl_state = torch.load(ctrl_file, map_location={'cuda:0': str(device)})
#    print("Loading Controller with reward {}".format(
#        ctrl_state['reward']))
#    controller.load_state_dict(ctrl_state['state_dict'])
    
swm_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
controller_params = sum(p.numel() for p in controller.parameters() if p.requires_grad)

print("SWM trainable params:", swm_params, model)
print("Controller trainable params:", controller_params, controller)
print("Total trainable params:", swm_params + controller_params)
