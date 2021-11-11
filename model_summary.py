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


def model_summary(config):
    device = torch.device('cpu')
    
    if config.use_wm:
        vae_file, rnn_file= \
            [join(config.exp_dir, m, 'best.tar') for m in ['vae', 'mdrnn']]
            
        vae_state, rnn_state = [
            torch.load(fname, map_location={'cuda:0': str(device)})
            for fname in (vae_file, rnn_file)]
            
        vae = VAE(config.input_dim, config.vae_dim).to(device)
        vae.load_state_dict(vae_state['state_dict'])

        mdrnn = MDRNNCell(config.vae_dim, config.n_agents * config.action_dim, config.mdrnn_dim, 5).to(device)
        mdrnn.load_state_dict(
            {k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()})
            
        vae_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
        mdrnn_params = sum(p.numel() for p in mdrnn.parameters() if p.requires_grad)
            
        print(f"VAE Loss: {vae_state['precision']:.3f}")
        print(f"MDRNN Loss: {rnn_state['precision']:.3f}")
        print("VAE Params:", vae_params)
        print("MDRNN Params:", mdrnn_params)
        print("WM Params:", vae_params + mdrnn_params)

    if config.use_swm:
        input_shape = config.input_dim
        embedding_dim = config.swm_latent_dim
        hidden_dim = config.swm_hidden_dim
        action_dim = config.action_dim
        num_objects = config.n_agents
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
        
        
        #swm_state = torch.load(join('checkpoints', 'model.pt'), map_location={'cuda:0': str(device)})
        #print(swm_state.keys())
        #model.load_state_dict(swm_state['state_dict'])
        
        mlp = model.obj_encoder
        gnn = model.transition_model
        mlp_params = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
        gnn_params = sum(p.numel() for p in gnn.parameters() if p.requires_grad)
        swm_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        #print(f"SWM Loss: {swm_state['precision']:.3f}")
        print("MLP Params:", mlp_params)
        print("GNN Params:", gnn_params)
        print("SWM Params:", mlp_params + gnn_params)
