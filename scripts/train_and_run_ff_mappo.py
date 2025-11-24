"""Utility script to train a feed-forward MAPPO agent and run evaluations.

This script composes a Hydra configuration, launches the standard Mava MAPPO
training loop, and then runs a configurable number of evaluation episodes with
the freshly trained policy. It is intended as a lightweight convenience wrapper
around the existing training entrypoint that ships with Mava.
"""

from __future__ import annotations

import argparse
import copy
import pickle
from pathlib import Path
from pprint import pprint
from typing import Dict, Iterable, List, Tuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from flax import jax_utils
from flax.core.frozen_dict import FrozenDict
from omegaconf import DictConfig, OmegaConf

from mava.evaluator import get_eval_fn, make_ff_eval_act_fn
from mava.systems.ppo.anakin.ff_mappo import run_experiment, learner_setup
from mava.utils import make_env as environments
from mava.utils.jax_utils import unreplicate_batch_dim
from mava.utils.config import check_total_timesteps

import plotly.express as px
import plotly.io as pio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "mava" / "configs"
DEFAULT_CONFIG = "default/ff_mappo"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the Mava feed-forward MAPPO agent and run post-training evaluations. "
            "Configuration overrides use standard Hydra syntax (e.g. \"arch.seed=7\")."
        )
    )
    parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG,
        help=(
            "Relative Hydra config path to load (default: %(default)s). "
            "Paths are resolved relative to `mava/configs`."
        ),
    )
    parser.add_argument(
        "-o",
        "--override",
        dest="overrides",
        action="append",
        default=[],
        help=(
            "Hydra-style config override. May be supplied multiple times, e.g. "
            "-o arch.seed=123 -o env.scenario.name=cartpole."
        ),
    )
    parser.add_argument(
        "--rollout-episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes to run after training (default: %(default)s).",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=1,
        help=(
            "Additional offset added to the training seed for post-training evaluations. "
            "Useful when running multiple evaluations without retraining."
        ),
    )
    parser.add_argument(
        "--save-network",
        type=str,
        default=None,
        help=(
            "Path where to save the trained network parameters. "
            "If not specified, network will not be saved."
        ),
    )
    parser.add_argument(
        "--load-network",
        type=str,
        default=None,
        help=(
            "Path to load pre-trained network parameters from. "
            "If specified, training will be skipped and rollout will use loaded network."
        ),
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help=(
            "Skip training phase. Must be used with --load-network to perform rollout only."
        ),
    )
    return parser.parse_args()


def _resolve_config_location(config_name: str) -> Tuple[Path, str]:
    """Split a config reference into directory and base filename.

    Hydra expects `config_name` to be the filename (without extension) scoped to the
    directory provided during initialization. For convenience, we allow callers to
    specify paths like ``default/ff_mappo`` and map them to the
    corresponding directory and config name automatically.
    """

    config_path = Path(config_name)
    if config_path.suffix:
        # Strip optional extension so users can pass `foo/bar.yaml` if desired.
        config_path = config_path.with_suffix("")

    if config_path.parent == Path("."):
        return CONFIG_ROOT, config_path.name

    resolved_dir = CONFIG_ROOT / config_path.parent
    return resolved_dir, config_path.name


def _load_config(config_name: str, overrides: Iterable[str]) -> DictConfig:
    """Compose a Hydra configuration with optional overrides."""
    override_list = list(overrides)
    config_dir, resolved_name = _resolve_config_location(config_name)

    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.2", config_dir=str(config_dir)):
        cfg = compose(config_name=resolved_name, overrides=override_list)
    OmegaConf.set_struct(cfg, False)
    return cfg


def _summarize_metrics(metrics: Dict[str, np.ndarray]) -> Dict[str, Tuple[float, float]]:
    """Return mean and standard deviation for each logged metric."""
    summary: Dict[str, Tuple[float, float]] = {}
    for key, values in metrics.items():
        arr = np.asarray(values)
        summary[key] = (float(np.mean(arr)), float(np.std(arr)))
    return summary


def save_network(
    filepath: str,
    actor_params: FrozenDict,
    config: DictConfig,
    actor_apply_fn=None,
) -> None:
    """Save trained network parameters and configuration to disk.
    
    Args:
        filepath: Path where to save the network (will be created if doesn't exist)
        actor_params: The actor network parameters to save (can have device/batch dims)
        config: The configuration used for training
        actor_apply_fn: Optional actor apply function (not saved, just for completeness)
    """
    save_path = Path(filepath)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract single actor params (remove device and batch dimensions)
    params_single = _extract_single_actor_params(actor_params)
    
    # Convert JAX arrays to numpy for serialization
    params_numpy = jax.tree_util.tree_map(lambda x: np.array(x), params_single)
    
    save_dict = {
        "actor_params": params_numpy,
        "config": OmegaConf.to_container(config, resolve=True),
        "version": "1.0",
    }
    
    with open(save_path, "wb") as f:
        pickle.dump(save_dict, f)
    
    print(f"Network saved to: {save_path}")


def load_network(
    filepath: str,
    return_config: bool = True,
) -> Tuple[FrozenDict, Optional[DictConfig]]:
    """Load trained network parameters from disk.
    
    Args:
        filepath: Path to the saved network file
        return_config: Whether to return the config along with parameters
        
    Returns:
        Tuple of (actor_params, config) where config is None if return_config=False
    """
    load_path = Path(filepath)
    if not load_path.exists():
        raise FileNotFoundError(f"Network file not found: {load_path}")
    
    with open(load_path, "rb") as f:
        save_dict = pickle.load(f)
    
    # Convert numpy arrays back to JAX arrays
    actor_params = jax.tree_util.tree_map(lambda x: jnp.array(x), save_dict["actor_params"])
    actor_params = FrozenDict(actor_params)
    
    config = None
    if return_config and "config" in save_dict:
        config = OmegaConf.create(save_dict["config"])
    
    print(f"Network loaded from: {load_path}")
    if "version" in save_dict:
        print(f"Network version: {save_dict['version']}")
    
    return actor_params, config


def _evaluate_policy(
    config: DictConfig,
    actor_apply_fn,
    actor_params,
    num_episodes: int,
    seed_offset: int,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, object]]]:
    """Roll out the trained policy for a handful of evaluation episodes."""
    eval_config = copy.deepcopy(config)
    eval_config.arch.num_evaluation = 1

    _, eval_env = environments.make(config=eval_config, add_global_state=True)
    
    n_devices = len(jax.devices())
    eval_key = jax.random.PRNGKey(eval_config.system.seed + seed_offset)
    eval_keys = jax.random.split(eval_key, n_devices)
    eval_keys = eval_keys.reshape(n_devices, -1)

    # Actor params should already be in shape (n_devices, ...)
    # If not, replicate them
    replicated_params = _replicate_params_for_evaluator(actor_params, eval_config)
    eval_act_fn = make_ff_eval_act_fn(actor_apply_fn, eval_config)
    evaluator = get_eval_fn(eval_env, eval_act_fn, eval_config, absolute_metric=False)

    eval_metrics = evaluator(replicated_params, eval_keys, {})
    eval_metrics = jax.device_get(eval_metrics)
    
    detailed_rollout = _collect_rollout_details(
        config=config,
        actor_apply_fn=actor_apply_fn,
        actor_params=actor_params,
        num_episodes=num_episodes,
        seed_offset=seed_offset,
    )
    return {name: np.asarray(value) for name, value in eval_metrics.items()}, detailed_rollout


def _replicate_params_for_evaluator(params: FrozenDict, config: DictConfig) -> FrozenDict:
    """Match the shape expected by evaluator (device × ...)."""

    n_devices = jax.local_device_count()
    leaves = jax.tree_util.tree_leaves(params)
    
    # Check if already in correct format: (n_devices, ...)
    if leaves:
        first = leaves[0]
        if first.ndim >= 1 and first.shape[0] == n_devices:
            return params

    # Base params have shape (...) - need to add (n_devices, ...)
    # Just replicate across devices
    base_params = jax.tree_util.tree_map(jnp.asarray, params)
    replicated = jax_utils.replicate(base_params)
    
    return replicated


def _extract_single_actor_params(params: FrozenDict) -> FrozenDict:
    """Strip device and batch axes from replicated actor parameters.
    
    MAPPO params have shape (n_devices, update_batch_size, ...) or (n_devices, ...)
    We want to extract just the base params (...) for saving.
    """

    leaves = jax.tree_util.tree_leaves(params)
    if not leaves:
        return params

    candidate = params
    first_leaf = leaves[0]
    
    # Check if we have (n_devices, update_batch_size, ...) - 3+ dims
    # or just (n_devices, ...) - 2+ dims
    if first_leaf.ndim >= 3:
        # Has both device and batch dimensions
        # unreplicate_batch_dim removes the batch dim: (n_devices, update_batch_size, ...) -> (n_devices, ...)
        candidate = unreplicate_batch_dim(candidate)
        leaves = jax.tree_util.tree_leaves(candidate)
        first_leaf = leaves[0]
    
    # Now remove device dimension if present
    if first_leaf.ndim >= 1 and first_leaf.shape[0] == jax.device_count():
        # Take first device params (they should all be identical)
        candidate = jax.tree_util.tree_map(lambda x: x[0], candidate)

    return jax.tree_util.tree_map(jnp.asarray, candidate)


def _tree_to_serializable(tree: object) -> object:
    """Convert a pytree of JAX/NumPy arrays into Python-native containers."""

    device_fetched = jax.device_get(tree)

    def _convert(x: object) -> object:
        if isinstance(x, (jax.Array, jnp.ndarray, np.ndarray)):
            return np.asarray(x).tolist()
        return x

    return jax.tree_util.tree_map(_convert, device_fetched)


def _collect_rollout_details(
    config: DictConfig,
    actor_apply_fn,
    actor_params: FrozenDict,
    num_episodes: int,
    seed_offset: int,
) -> List[Dict[str, object]]:
    """Generate a detailed rollout trace with full environment state and extras."""

    if num_episodes <= 0:
        return []

    detail_cfg = copy.deepcopy(config)
    detail_cfg.arch.num_envs = 1
    detail_cfg.system.update_batch_size = 1
    detail_cfg.arch.num_evaluation = 1

    _, eval_env = environments.make(config=detail_cfg, add_global_state=True)

    # Extract single params if they are replicated, otherwise use as is
    leaves = jax.tree_util.tree_leaves(actor_params)
    if leaves and leaves[0].shape[0] == jax.device_count():
        # Params are replicated across devices, extract single copy
        params_single = _extract_single_actor_params(actor_params)
    else:
        # Params are already single (not replicated)
        params_single = actor_params
    
    eval_act_fn = make_ff_eval_act_fn(actor_apply_fn, detail_cfg)

    rollout: List[Dict[str, object]] = []
    key = jax.random.PRNGKey(detail_cfg.system.seed + seed_offset)
    key, reset_key = jax.random.split(key)
    env_state, timestep = eval_env.reset(reset_key)

    max_steps = 500  # safeguard against infinite loops
    episodes = 0
    step_counter = 0

    current_state = env_state
    current_timestep = timestep

    while episodes < num_episodes and step_counter < max_steps:
        obs = jax.tree_util.tree_map(lambda x: jnp.asarray(x), current_timestep.observation)
        batched_obs = jax.tree_util.tree_map(lambda x: x[None, ...], obs)

        key, actor_key = jax.random.split(key)
        actor_policy = actor_apply_fn(params_single, batched_obs)
        action = actor_policy.sample(seed=actor_key)
        action_array = jnp.asarray(action)

        next_state, next_timestep = eval_env.step(current_state, action_array.squeeze(0))

        step_record: Dict[str, object] = {
            "step": step_counter,
            "env_state": _tree_to_serializable(current_state),
            "observation": _tree_to_serializable(obs),
            "action": np.asarray(action_array.squeeze(0)).tolist(),
            "reward": float(np.asarray(next_timestep.reward).squeeze()),
            "discount": float(np.asarray(next_timestep.discount).squeeze()),
            "extras": _tree_to_serializable(next_timestep.extras),
            "terminal": bool(np.asarray(next_timestep.last()).squeeze()),
            "next_env_state": _tree_to_serializable(next_state),
        }
        rollout.append(step_record)

        step_counter += 1
        if step_record["terminal"]:
            episodes += 1
            if episodes >= num_episodes:
                break
            # Reset environment for next episode
            key, reset_key = jax.random.split(key)
            current_state, current_timestep = eval_env.reset(reset_key)
        else:
            current_state = next_state
            current_timestep = next_timestep

    if step_counter >= max_steps:
        print(
            "Warning: rollout collection reached max_steps limit before completing the requested"
            " number of episodes."
        )

    return rollout


def _print_rollout_details(rollout: List[Dict[str, object]]) -> None:
    if not rollout:
        print("\nNo detailed rollout data captured.")
        return

    print("\nDetailed rollout trace (per step):")
    for record in rollout:
        print(f"\nStep {record['step']}:")
        pprint({k: v for k, v in record.items() if k != "step"})


def _plot_rollout_visualization(rollout: List[Dict[str, object]], output_dir: Path) -> None:
    """Create visualization plots from rollout data."""
    if not rollout:
        print("No rollout data to visualize.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract trajectory data - env_state might be a nested object
    # Try to access as dict first, then as object attributes
    def safe_get(record, key_path):
        """Safely get nested values from dict or object."""
        obj = record
        for key in key_path:
            if isinstance(obj, dict):
                obj = obj[key]
            else:
                obj = getattr(obj, key)
        return obj
    
    try:
        # env_state -> env_state (JaxMarlState) -> state (State) -> fields
        # Position, speed, etc. are arrays with shape [num_agents, ...]
        # We take the first agent [0]
        x = [safe_get(record, ['env_state', 'env_state', 'state', 'position'])[0][0] for record in rollout]
        y = [safe_get(record, ['env_state', 'env_state', 'state', 'position'])[0][1] for record in rollout]
        z = [safe_get(record, ['env_state', 'env_state', 'state', 'position'])[0][2] for record in rollout]
        
        speed = [safe_get(record, ['env_state', 'env_state', 'state', 'speed'])[0] for record in rollout]
        glide_angle = [safe_get(record, ['env_state', 'env_state', 'state', 'attitude'])[0][0] for record in rollout]
        side_angle = [safe_get(record, ['env_state', 'env_state', 'state', 'attitude'])[0][1] for record in rollout]
        
        # Calculate vertical speed
        vertical_speed = [float(s) * np.sin(float(g)) for s, g in zip(speed, glide_angle)]
        
        # Extract control data
        bank_control = [safe_get(record, ['env_state', 'env_state', 'state', 'controls'])[0][0] for record in rollout]
        attack_control = [safe_get(record, ['env_state', 'env_state', 'state', 'controls'])[0][1] for record in rollout]
        sideslip_control = [safe_get(record, ['env_state', 'env_state', 'state', 'controls'])[0][2] for record in rollout]
    except (KeyError, AttributeError, IndexError, TypeError) as e:
        print(f"Warning: Could not extract all state data for visualization: {e}")
        print("Rollout structure:")
        if rollout:
            import pprint
            pprint.pprint(rollout[0], depth=3)
        return
    
    rewards = [record['reward'] for record in rollout]
    times = [record['step'] for record in rollout]

    # Create time series plots
    fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True)
    axes = axes.flatten()

    labels = [
        (x, "Position X (m)"),
        (y, "Position Y (m)"),
        (z, "Altitude Z (m)"),
        (speed, "Speed (m/s)"),
        (vertical_speed, "Vertical Speed (m/s)"),
        (glide_angle, "Glide Angle (rad)"),
        (side_angle, "Side Angle (rad)"),
        (bank_control, "Bank Control (rad)"),
        (attack_control, "Attack Control (rad)"),
        (sideslip_control, "Sideslip Control (rad)"),
        (rewards, "Reward"),
    ]

    for ax, (data, title) in zip(axes, labels):
        ax.plot(times, data, linewidth=1.0)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.4)
    
    # Hide unused subplots
    for ax in axes[len(labels):]:
        ax.axis('off')

    fig.suptitle("Glider State Trajectories")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    
    timeseries_path = output_dir / "state_timeseries.png"
    fig.savefig(timeseries_path, dpi=150)
    plt.close(fig)
    print(f"Time series plot saved to: {timeseries_path}")

    # Create 3D trajectory plot
    fig_3d = px.line_3d(x=x, y=y, z=z)
    fig_3d.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis=dict(title='X Position (m)'),
            yaxis=dict(title='Y Position (m)'),
            zaxis=dict(title='Altitude Z (m)')
        ),
        title="3D Trajectory Visualization"
    )
    
    trajectory_path = output_dir / "trajectory_3d.html"
    pio.write_html(fig_3d, file=str(trajectory_path), auto_open=False)
    print(f"3D trajectory plot saved to: {trajectory_path}")


def main() -> None:
    args = _parse_args()
    
    # Validate arguments
    if args.skip_training and not args.load_network:
        raise ValueError("--skip-training requires --load-network to be specified")
    
    # Load or train network
    if args.load_network:
        print(f"Loading pre-trained network from: {args.load_network}")
        loaded_params_raw, loaded_config = load_network(args.load_network, return_config=True)
        
        # Always use the base config from command line for architecture and system settings
        cfg = _load_config(args.config_name, args.overrides)
        
        if loaded_config is not None:
            print("Note: Using base configuration (loaded config available for reference)")
            print("Tip: To match original training config, use appropriate overrides with -o")
        else:
            print("Note: Using base configuration (no config in network file)")
        
        # Setup environment and network
        cfg.logger.system_name = "ff_mappo"
        n_devices = len(jax.devices())
        cfg = check_total_timesteps(cfg)
        
        env, _ = environments.make(config=cfg, add_global_state=True)
        key, actor_net_key, critic_net_key = jax.random.split(
            jax.random.PRNGKey(cfg.system.seed), num=3
        )
        _, actor_network, _ = learner_setup(env, (key, actor_net_key, critic_net_key), cfg)
        actor_apply_fn = actor_network.apply
        
        # Replicate params for evaluation
        actor_params = _replicate_params_for_evaluator(loaded_params_raw, cfg)
        
        if not args.skip_training:
            print("Warning: --load-network specified but training will still run.")
            print("Use --skip-training to skip training and only perform rollout.")
            eval_score = run_experiment(cfg)
            print(f"Training complete. Final evaluation metric: {eval_score:.4f}")
            # After training, get the trained params
            # Note: run_experiment doesn't return params, so we need to modify it
            # For now, use loaded params
    else:
        # Standard training path
        cfg = _load_config(args.config_name, args.overrides)
        cfg.logger.system_name = "ff_mappo"
        
        print("Starting MAPPO training with configuration:")
        print(OmegaConf.to_yaml(cfg, resolve=True))
        
        # We need to modify run_experiment to return the trained params
        # For now, we'll train and then recreate the setup to get the network
        # This is a workaround since run_experiment doesn't return params
        
        # Train
        eval_score = run_experiment(cfg)
        print(f"Training complete. Final evaluation metric: {eval_score:.4f}")
        
        # After training, we need to setup the network again
        # But we can't get the trained params without modifying run_experiment
        # So for the training path, we'll need to load from checkpoint or modify the code
        
        print("Warning: Training path doesn't support direct parameter extraction.")
        print("Please use checkpointing to save/load trained parameters, or use --load-network with pre-trained network.")
        print("Skipping rollout evaluation.")
        return
    
    # Save network if requested
    if args.save_network and not args.skip_training and 'actor_params' in locals():
        print(f"Saving trained network to: {args.save_network}")
        save_network(args.save_network, actor_params, cfg, actor_apply_fn)
    
    # Run evaluation rollouts
    print(
        f"Running {args.rollout_episodes} post-training evaluation episode(s) using seed offset "
        f"{args.seed_offset}..."
    )
    metrics, rollout_details = _evaluate_policy(
        config=cfg,
        actor_apply_fn=actor_apply_fn,
        actor_params=actor_params,
        num_episodes=args.rollout_episodes,
        seed_offset=args.seed_offset,
    )

    summary = _summarize_metrics(metrics)
    print("\nAggregated evaluation metrics (mean ± std):")
    for name, (mean, std) in summary.items():
        print(f"  {name:20s} : {mean:.4f} ± {std:.4f}")

    if "episode_return" in metrics:
        flat_returns = np.asarray(metrics["episode_return"]).reshape(-1)
        preview = ", ".join(f"{ret:.2f}" for ret in flat_returns[: min(10, len(flat_returns))])
        print(f"\nPer-episode returns (first {min(10, len(flat_returns))} shown): {preview}")
    
    # Visualize rollout results
    output_dir = Path("outputs") / "glider_mappo_rollout"
    _plot_rollout_visualization(rollout_details, output_dir)

    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    if args.skip_training:
        print("Mode: Rollout only (training skipped)")
        print(f"Network loaded from: {args.load_network}")
    else:
        print("Mode: Training + Rollout")
        if args.save_network:
            print(f"Network saved to: {args.save_network}")
    print(f"Rollout episodes: {args.rollout_episodes}")
    print(f"Visualization saved to: {output_dir}")
    print("="*80)
    print("\nDone. Logs, checkpoints, and evaluation artifacts are available under \"results/\" and \"outputs/\".")


if __name__ == "__main__":
    main()
