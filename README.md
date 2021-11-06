# SMAWM

In multi-agent reinforcement learning (MARL), the difficulty of generalising to diverse strategies and adapting to non-stationary behaviour remains a challenge. Inspired by model-based reinforcement learning (MBRL), we propose SMAWM or Structured Multi-Agent World Models, a world model that encompasses other agents in a compositional structure, to provide a strong inductive bias to novel interactions among multiple agents in the environment.

## Set-up
```
pip install -r requirements.txt
```

## Train WM
```
python mpe.py
python vae.py
python mdrnn.py
python controller.py
python wm.py
```

## Train SWM
```
python mpe_swm.py
python swm.py
python controller.py
python wm.py
```

## Project Breakdown

### Dataset Generation
- mpe.py - Generates MPE dataset.
- mpe_spurious.py - Generates MPE dataset for the spurious context.
- 
### World Model
- loaders.py - Utilities for loading data for WM.
- vae.py - Trains the WM variational autoencoder.
- mdrnn.py - Defines the WM mixture density recurrent neural network.
- wm.py - Evaluates WM on the MPE environment.
- controller.py - Trains the WM controller.

### Structured World Model
- learning.py - Utilities for SWM
- swm.py - Defines the SWM.
- swm_controller.py - Trains the SWM controller.
- mpe_swm.py - Evaluates SWM on the MPE environment.

### Utilities
- policies.py - Defines fixed policies for the environment.

### Miscellaneous
- /datasets/mpe - Contains 1000 episodes of length 1000 each, collected from the MPE environment in .npz files.
- /exp_dir
  - /ctrl - Contains trained model of the WM controller.
  - /mdrnn - Contains trained model of the WM mixture density recurrent neural network.
  - /vae - Contains trained model of the WM variational autoencoder.
- agents.py - NOT USED.
- dev.py - Useless file for random testing/development.
- misc.py - Utilities for ?
- models.py - NOT USED?
- requirements.txt - List of dependencies.
- test.py - Useless file for random testing/development.
