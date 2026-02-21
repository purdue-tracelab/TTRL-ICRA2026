# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved. BSD-3-Clause.

import argparse
import os

import torch
from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry
from legged_lab.rl.epo import EpoCfg, EpoRunner
import legged_lab.utils.cli_args as cli_args


parser = argparse.ArgumentParser(description="Play an EPO-trained policy (leader gene).")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--gene_id", type=int, default=0, help="Gene id to use for all envs (default leader=0)")
parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions")
parser.add_argument("--predictor", action="store_true", help="Use predictor-augmented EPO runner and export predictor")

# RSL-RL style args for loading
cli_args.add_rsl_rl_args(parser)
# Provide friendly aliases for checkpoint args
parser.add_argument("--load_checkpoint", dest="checkpoint", type=str, default=None,
                    help="Alias for --checkpoint")
parser.add_argument("--checkpoints", dest="checkpoint", type=str, default=None,
                    help="Alias for --checkpoint")

# App args
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()


def main():
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Import after app launch
    from isaaclab_tasks.utils import get_checkpoint_path
    import legged_lab.envs  # noqa: F401 ensures tasks register
    from legged_lab.utils.cli_args import update_rsl_rl_cfg

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)

    # Resolve checkpoint path
    log_root_path = os.path.join("logs", agent_cfg.experiment_name or "epo_experiment")
    log_root_path = os.path.abspath(log_root_path)
    # Prefer CLI-provided values directly
    load_run = args_cli.load_run if hasattr(args_cli, "load_run") else None
    load_ckpt = args_cli.checkpoint if hasattr(args_cli, "checkpoint") else None
    resume_path = get_checkpoint_path(log_root_path, load_run, load_ckpt)
    print(f"[INFO] Loading checkpoint: {resume_path}")

    # Minimal EPO cfg for play (values won't affect inference)
    epo_cfg = EpoCfg(population_size=8, param_size=32)

    # Build runner to construct policy with correct shapes, then load weights
    agent_cfg_dict = agent_cfg.to_dict() if hasattr(agent_cfg, "to_dict") else agent_cfg
    if args_cli.predictor:
        from legged_lab.rl.epo import EpoPredictorRunner
        runner = EpoPredictorRunner(env, agent_cfg_dict, epo_cfg=epo_cfg, log_dir=os.path.dirname(resume_path), device=agent_cfg.device)
    else:
        runner = EpoRunner(env, agent_cfg_dict, epo_cfg=epo_cfg, log_dir=os.path.dirname(resume_path), device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)

    # When using predictor runner, export predictor TorchScript for deployment
    if args_cli.predictor:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        os.makedirs(export_model_dir, exist_ok=True)
        try:
            try:
                orig_device = next(runner._predictor.parameters()).device
            except StopIteration:
                orig_device = torch.device("cpu")
            runner._predictor.to("cpu").eval()
            ts_predictor = torch.jit.script(runner._predictor)
            out_path = os.path.join(export_model_dir, "predictor.pt")
            ts_predictor.save(out_path)
            runner._predictor.to(orig_device)
            print(f"[Predictor] Exported TorchScript to: {out_path}")
        except Exception as e:
            print(f"[Predictor] Export failed: {e}")

    # Force all envs to use the selected gene (leader by default)
    runner.agent_id_per_env = torch.zeros(env.num_envs, dtype=torch.long, device=env.device) + int(args_cli.gene_id)

    # Optional keyboard for teleop while watching
    if not args_cli.headless:
        try:
            from legged_lab.utils.keyboard import Keyboard

            _keyboard = Keyboard(env)  # noqa: F841
        except Exception:
            pass

    # Roll policy
    actor_obs, extras = env.get_observations()
    critic_obs = extras["observations"]["critic"]
    deterministic = bool(args_cli.deterministic)

    # Track per-term episodic averages
    from collections import defaultdict
    num_envs = env.num_envs
    term_sums = defaultdict(lambda: torch.zeros(num_envs, device=env.device))  # per-env running sums
    reward_sums = torch.zeros(num_envs, device=env.device)
    term_episode_values = defaultdict(list)  # per-term episodic sums (from per-env accumulation)
    scalar_term_episode_values = defaultdict(list)  # episodic scalar logs (Episode_Reward/*)
    reward_episode_values = []
    episodes_since_print = 0
    print_interval = 20

    while simulation_app.is_running():
        with torch.inference_mode():
            # append id (constant) as last column
            ids = runner.agent_id_per_env
            id_col = ids.float().unsqueeze(1)
            actor_obs_id = torch.cat([actor_obs, id_col], dim=1)
            critic_obs_id = torch.cat([critic_obs, id_col], dim=1)
            actions, _, _ = runner.policy.act(actor_obs_id, critic_obs_id, deterministic=deterministic)
            actor_obs, rewards, dones, extras = env.step(actions)
            critic_obs = extras["observations"]["critic"]

            # accumulate per-step logs per env for episodic averaging
            logs = extras.get("log", {}) if isinstance(extras, dict) else {}
            if isinstance(logs, dict):
                for k, v in logs.items():
                    # Per-env vectors: accumulate into running sums
                    if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == num_envs:
                        term_sums[k] += v.to(env.device)
                    # Scalars: treat as per-episode scalar terms that environment already aggregated
                    elif isinstance(v, torch.Tensor) and v.dim() == 0:
                        scalar_term_episode_values[k].append(float(v.item()))
                    elif isinstance(v, (float, int)):
                        scalar_term_episode_values[k].append(float(v))
            reward_sums += rewards.to(env.device)

            # when an env is done, record episode values and reset running sums
            done_envs = dones.nonzero(as_tuple=False).flatten()
            if len(done_envs) > 0:
                for env_id in done_envs.tolist():
                    for k in list(term_sums.keys()):
                        term_episode_values[k].append(float(term_sums[k][env_id].item()))
                        term_sums[k][env_id] = 0.0
                    reward_episode_values.append(float(reward_sums[env_id].item()))
                    reward_sums[env_id] = 0.0
                    episodes_since_print += 1

            # periodic printout of per-term averages
            if episodes_since_print >= print_interval:
                print("\n==== EPO Play: Average episodic rewards over last", episodes_since_print, "episodes ====")
                if len(reward_episode_values) > 0:
                    print(f"Total reward: {sum(reward_episode_values)/len(reward_episode_values):.4f}")
                # Print Episode_* scalar terms (environment-provided), show Episode_Reward/* first if present
                # Episode_Reward terms
                for k, vals in scalar_term_episode_values.items():
                    if "Episode_Reward/" in k and len(vals) > 0:
                        print(f"{k}: {sum(vals)/len(vals):.6f}")
                # Other Episode/* scalar terms
                for k, vals in scalar_term_episode_values.items():
                    if k.startswith("Episode/") and "Episode_Reward/" not in k and len(vals) > 0:
                        print(f"{k}: {sum(vals)/len(vals):.6f}")
                # Per-env accumulated terms (if any present as vectors)
                for k, vals in term_episode_values.items():
                    if len(vals) > 0:
                        print(f"{k}: {sum(vals)/len(vals):.6f}")
                # reset window
                term_episode_values.clear()
                scalar_term_episode_values.clear()
                reward_episode_values.clear()
                episodes_since_print = 0

    simulation_app.close()


if __name__ == "__main__":
    main()
