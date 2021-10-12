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