# Structured Multi-Agent World Models

**Winner of CS4246 Project Competition 2021. Read the paper: [PDF](https://jetnew.io/assets/pdf/new2021structured.pdf)**

In multi-agent reinforcement learning, the difficulty of generalising to diverse interactions remains a challenge. Inspired by model-based reinforcement learning, we present Structured Multi-Agent World Models (SMAWM), a world model that encompasses other agents in a compositional structure, to provide a strong inductive bias for generalising to novel interactions among multiple agents in the environment. We show that reinforcement learning with the agent-factored state representation outperforms that with a purely connectionist world model despite using much fewer parameters. We further show that SMAWM learns an effective representation that is capable of much higher accuracy in forward prediction for planning, and propose future extensions that can likely scale SMAWM to environments of higher complexity.

## Project Breakdown

* Generate datasets - `generate_dataset.py`
* Train world model - `python train_wm.py`
* Train SMAWM - `python train_swm.py`
* Train RL agents - `python train_rl.py`
* Run experiments - `python run_experiments.py`
* Run analysis - `python run_analysis.py`

## Results

1. Performance of SMAWM exceeds World Models across all settings of interest.
2. Performance of SMAWM does not change significantly as parameter count increases.
3. SMAWM has higher prediction accuracy than World Models for very short time steps.

## Future Extensions

1. Opponent modeling to explicitly model joint action or joint policies of agents.
2. Graph-VRNN to overcome environment stochasticity and partial observability.
3. SMAWM as a model-based reinforcement learning method with online planning.
