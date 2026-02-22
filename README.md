# [ICRA2026] End-to-end RL for Humanoid Robot Table Tennis


### [**ArXiv**](https://arxiv.org/abs/2509.21690)   | [**Video**](https://www.youtube.com/watch?v=vzXuCIXpLaE) |  [**TRACE lab**](https://www.thetracelab.com/) 


## 📝 Overview
This is the official implementation of ICRA 2026 paper: [Towards Versatile Humanoid Table Tennis: Unified Reinforcement Learning with Prediction Augmentation ](https://arxiv.org/abs/2509.21690) 


**TL;DR:**

- Goal: Train a single, end-to-end RL policy that directly maps ball position observations + robot proprioception to whole-body joint commands so a humanoid can both move (footwork) and strike (paddle control) fast enough for table tennis. 


- Key idea: Add prediction augmentation to make the policy proactive and trainable:

- A lightweight learned ball-hitting-position predictor estimates a future “target hitting point” (post-bounce apex) from recent ball positions, and this predicted target is fed into the actor to guide footwork before the ball arrives. 

- In simulation, a physics-based predictor provides accurate future ball states to (i) supervise the learned predictor and (ii) build dense, immediate rewards (instead of sparse “did you return the ball?” signals). 

- Results (sim): High performance across long/short/mixed serves, reporting hit rate ≥ 96% and success/return rate ≥ 92% (mixed serves ~94% success) (We tested under IsaacSim 4.5.0, upgrading IsaacSim to 5.0.0+ degrades success rate). Ablations show that both the predictor and prediction-based rewards are crucial. 

## Simulation Demos
- 🔴 **Red ball:** the table tennis ball in simulation.
- 🟢 **Green ball (physics-based future ball prediction):** The analytical/physics predictor is visualized as the green ball.
- 🟡 **Yellow ball (learned prediction):** The learned predictor output is visualized as the yellow ball.
- 🔵 **Blue ball:** A heuristic target body position for forehand hitting.

We only use learned prediction 🟡 before hitting, and after hitting is considered OOD for the predictor, explaining why it is drifting.

[2_predictor.webm](https://github.com/user-attachments/assets/30f7c1fb-e437-478f-a0a7-2f607d5957bd)

We construct a hitting heuristic (forehand only) of target body position 🔵, visualized by a blue ball, which encourages the robot to move proactively by reward design.

[3_pre-contact.webm](https://github.com/user-attachments/assets/a2d9412a-61ce-436d-9be0-dd907fb8af78)

More clips with caption and hardware experiments can be found in our paper [video](youtube.com/watch?v=vzXuCIXpLaE&feature=youtu.be).

## Key Features

- Environment: Table tennis simulation with configurable serving range and air drag implementation(legged_lab.physics.aerodynamics.py).
- Algorithm: Baseline PPO from [RSL_RL](https://github.com/leggedrobotics/rsl_rl) and proposed prediction-augmented RL (rsl_rl.runners.on_policy_predictor_regression_runner.py)
- Supported Robots 🤖 : **Booster T1** 


## Installation


- Install Isaac Lab by following the [conda installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html). We tested our policy using IsaacSim 4.5.0 and we notice a significant performance drop of the same training config in updated IsaacSim 5.0+. 
```bash
conda create -n ttrl python=3.11
conda activate ttrl

pip install --upgrade pip

pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```
- Verify installation and accept EULA
```bash
isaacsim
```

- Clone this repository separately from the Isaac Lab installation (i.e., outside the `IsaacLab` directory):

- Using a Python interpreter that has Isaac Lab installed, install the library

```bash
cd TTRL-ICRA2026 
pip install -e .
```
- Using the same Python interpreter, install the customized rsl_rl library
```bash
cd rsl_rl
pip install -e .
cd ..
```


- Verify that the extension is correctly installed by running the following command:

```bash
python legged_lab/scripts/train.py --task=t1_tt --logger=tensorboard --num_envs=64
```


## Training and Evaluation

An optional auxiliary predictor can be enabled for the `t1_tt` task to learn a small MLP that predicts the ball's future pose. Use the `--predictor` flag to switch the runner to the predictor-augmented version.

- Train with predictor enabled:

```bash
python -m legged_lab.scripts.train \
  --task=t1_tt \
  --num_envs=4096 \
  --headless \
  --logger=tensorboard \
  --predictor
```
- (Training log) Tensorboard visualization of training record:
```bash
tensorboard --logdir logs
```

- Play with predictor inference and visualization enabled (modify the run and checkpoint accordingly):

```bash
python -m legged_lab.scripts.eval \
  --task=t1_tt_eval \
  --num_envs=16 \
  --load_run 2026-02-14_16-12-07 \
  --checkpoint model_9000.pt \
  --predictor
```

Notes:
- The predictor runner saves its weights inside the training checkpoint. When playing, pass `--predictor` to load these weights and run inference each step. If a checkpoint was trained without `--predictor`, it won’t contain predictor weights.
- Predictor hyperparameters can be configured under the agent config (e.g., `T1TableTennisAgentCfg.predictor`).
- During play, the predictor’s output is fed to the environment, and a separate orange sphere visualizes the predicted ball position.









## Acknowledgements

This repository is built upon prior work,s including:

- **[LeggedLab](https://github.com/Hellod035/LeggedLab)** – Built by Wandong Sun, this project provides the foundation and framework structure.  
- **[IsaacLab](https://github.com/isaac-sim/IsaacLab)** – The modular, reusable IsaacLab components greatly simplify environment and agent definitions.  
- **[legged_gym](https://github.com/leggedrobotics/legged_gym)** – Inspired the environment architecture and code modularity.
- **[RSL_RL](https://github.com/leggedrobotics/rsl_rl)** - Base PPO runner implementation.


## Citation



```text
@article{hu2025towards,
  title={Towards versatile humanoid table tennis: Unified reinforcement learning with prediction augmentation},
  author={Hu, Muqun and Chen, Wenxi and Li, Wenjing and Mandali, Falak and He, Zijian and Zhang, Renhong and Krisna, Praveen and Christian, Katherine and Benaharon, Leo and Ma, Dizhi and others},
  journal={arXiv preprint arXiv:2509.21690},
  year={2025}
}
```


