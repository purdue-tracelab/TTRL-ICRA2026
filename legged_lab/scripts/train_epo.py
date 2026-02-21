import argparse
import os
from datetime import datetime

import torch
from isaaclab.app import AppLauncher
from legged_lab.utils import task_registry
from legged_lab.rl.epo import EpoCfg, EpoRunner, ExtraParamActorCriticCfg
import legged_lab.utils.cli_args as cli_args


parser = argparse.ArgumentParser(description="Train an RL agent with EPO (gene table) + PPO.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--predictor", action="store_true", help="Use predictor-augmented EPO runner and train auxiliary predictor")

# Reuse rsl_rl args for PPO
cli_args.add_rsl_rl_args(parser)

# EPO-specific args
parser.add_argument("--epo_population", type=int, default=8)
parser.add_argument("--epo_param_size", type=int, default=32)
parser.add_argument("--epo_elite_frac", type=float, default=0.25)
parser.add_argument("--epo_mutation_std", type=float, default=0.03)
parser.add_argument("--epo_merges", type=int, default=1)
parser.add_argument("--epo_reassign", type=str, default="round_robin", choices=["round_robin", "random", "static"])
parser.add_argument("--epo_reassign_every", type=int, default=100)
parser.add_argument("--epo_steps", type=int, default=None, help="Rollout steps per env per iteration (defaults to agent cfg).")
parser.add_argument("--epo_leader_id", type=int, default=0)
# Leader off-policy mixing (default: True). Provide --no-epo_leader_offpolicy to disable.
parser.add_argument(
    "--epo_leader_offpolicy",
    dest="epo_leader_offpolicy",
    action="store_true",
    help="Leader mixes in off-policy samples",
)
parser.add_argument(
    "--no-epo_leader_offpolicy",
    dest="epo_leader_offpolicy",
    action="store_false",
    help="Disable leader off-policy mixing",
)
parser.set_defaults(epo_leader_offpolicy=True)
parser.add_argument("--epo_diversity_check", action="store_true", help="Enable diversity gate for EPO stage")

# App args
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()


def main():
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Import IsaacLab utilities after the app is launched to ensure isaacsim is available
    from isaaclab.utils.io import dump_yaml
    from isaaclab_tasks.utils import get_checkpoint_path
    # Import tasks to ensure registration occurs before querying task_registry
    import legged_lab.envs  # noqa: F401

    # torch perf flags
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)
    env_class = task_registry.get_task_class(env_class_name)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    # sync seed and devices
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.scene.seed = seed
        agent_cfg.seed = seed

    env = env_class(env_cfg, args_cli.headless)

    # build EPO config
    epo_cfg = EpoCfg(
        population_size=args_cli.epo_population,
        param_size=args_cli.epo_param_size,
        elite_fraction=args_cli.epo_elite_frac,
        mutation_std=args_cli.epo_mutation_std,
        merges_per_iter=args_cli.epo_merges,
        leader_id=args_cli.epo_leader_id,
        leader_uses_offpolicy=bool(args_cli.epo_leader_offpolicy),
        diversity_check=bool(args_cli.epo_diversity_check),
        reassign_strategy=args_cli.epo_reassign,
        reassign_every_iters=args_cli.epo_reassign_every,
        num_steps_per_env=args_cli.epo_steps,
    )

    # Log directory
    log_root_path = os.path.join("logs", agent_cfg.experiment_name or "epo_experiment")
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    # ensure log directory exists early
    os.makedirs(log_dir, exist_ok=True)

    # Convert agent_cfg to dict for the runner
    agent_cfg_dict = agent_cfg.to_dict() if hasattr(agent_cfg, "to_dict") else agent_cfg

    # Choose runner implementation
    if args_cli.predictor:
        from legged_lab.rl.epo import EpoPredictorRunner
        runner = EpoPredictorRunner(env, agent_cfg_dict, epo_cfg=epo_cfg, log_dir=log_dir, device=agent_cfg.device)
    else:
        runner = EpoRunner(env, agent_cfg_dict, epo_cfg=epo_cfg, log_dir=log_dir, device=agent_cfg.device)
    # Save checkpoints directly under the run directory (not a subfolder)
    runner.checkpoint_dir = log_dir

    # resume from checkpoint if requested
    if getattr(agent_cfg, "resume", False):
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path, load_optimizer=True)

    # learn
    # optionally dump configs for reproducibility
    try:
        os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        # save epo cfg as well
        from dataclasses import asdict as _asdict
        dump_yaml(os.path.join(log_dir, "params", "epo.yaml"), _asdict(epo_cfg))
    except Exception:
        pass

    runner.learn(num_learning_iterations=agent_cfg.max_iterations)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
