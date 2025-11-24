# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Feed-forward MAPPO experiment with post-training rollout logging."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chex
import hydra
import jax
import jax.numpy as jnp
import numpy as np
from colorama import Fore, Style
from flax import jax_utils as flax_jax_utils
from flax.core.frozen_dict import FrozenDict
from jumanji.types import TimeStep
from omegaconf import DictConfig, OmegaConf

from mava.evaluator import get_eval_fn, make_ff_eval_act_fn
from mava.systems.ppo.anakin import ff_mappo as base_ff_mappo
from mava.types import ActorApply, MarlEnv, Metrics
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.config import check_total_timesteps
from mava.utils.jax_utils import unreplicate_batch_dim, unreplicate_n_dims
from mava.utils.logger import LogEvent, MavaLogger
from mava.wrappers.auto_reset_wrapper import AutoResetWrapper
from mava.wrappers.episode_metrics import get_final_step_metrics
from jax import tree


def _select_observation_for_logging(timestep: TimeStep) -> Any:
    """Return the observation that corresponds to the state after the transition."""

    extras = getattr(timestep, "extras", {}) or {}
    if hasattr(extras, "get") and AutoResetWrapper.OBS_IN_EXTRAS_KEY in extras:
        return extras[AutoResetWrapper.OBS_IN_EXTRAS_KEY]
    return timestep.observation


def _maybe_extract_positions(timestep: TimeStep, observation: Any) -> Optional[np.ndarray]:
    """Attempt to extract agent positions from the timestep extras or observation."""

    candidate_keys = (
        "agent_positions",
        "positions",
        "agent_coords",
        "coords",
        "agent_locations",
    )
    extras = getattr(timestep, "extras", {}) or {}
    env_metrics = extras.get("env_metrics", {}) if hasattr(extras, "get") else {}

    for source in (env_metrics, extras):
        if hasattr(source, "get"):
            for key in candidate_keys:
                value = source.get(key)
                if value is not None:
                    return np.asarray(value)

    for attr in ("agent_positions", "agents_positions"):
        if hasattr(observation, attr):
            value = getattr(observation, attr)
            if value is not None:
                return np.asarray(value)

    return None


def _to_serialisable(array: Any) -> Any:
    if array is None:
        return None
    return np.asarray(array).tolist()


def _ensure_batched_observation(observation: Any) -> Tuple[Any, bool]:
    """Ensure the observation has a batch dimension for the actor network."""

    agents_view = getattr(observation, "agents_view", None)
    if agents_view is None and hasattr(observation, "observation"):
        agents_view = getattr(observation.observation, "agents_view", None)

    if agents_view is None or agents_view.ndim >= 3:
        return observation, False

    batched_observation = tree.map(
        lambda x: x if x is None else jnp.expand_dims(x, axis=0), observation
    )
    return batched_observation, True


def _select_action(
    actor_apply_fn: ActorApply,
    actor_params: FrozenDict,
    observation: Any,
    deterministic: bool,
    key: chex.PRNGKey,
) -> Any:
    batched_observation, added_batch = _ensure_batched_observation(observation)
    policy = actor_apply_fn(actor_params, batched_observation)
    action = policy.mode() if deterministic else policy.sample(seed=key)
    if added_batch:
        action = jnp.squeeze(action, axis=0)
    return action


def _rollout_single_episode(
    env: MarlEnv,
    actor_apply_fn: ActorApply,
    actor_params: FrozenDict,
    rollout_cfg: Dict[str, Any],
    key: chex.PRNGKey,
    episode_index: int,
) -> Tuple[Dict[str, Any], chex.PRNGKey]:
    """Simulate a single episode to record agent trajectories."""

    key, reset_key = jax.random.split(key)
    env_state, timestep = env.reset(reset_key)

    initial_obs = _select_observation_for_logging(timestep)
    initial_positions = _maybe_extract_positions(timestep, initial_obs)
    agent_paths: List[List[Any]] = [[] for _ in range(env.num_agents)]
    positions_tracked = initial_positions is not None
    if initial_positions is not None:
        for agent_id, position in enumerate(np.asarray(initial_positions)):
            agent_paths[agent_id].append(_to_serialisable(position))

    episode_steps: List[Dict[str, Any]] = []
    total_reward = np.zeros(env.num_agents, dtype=float)
    done = False
    step_idx = 0

    while (not done) and step_idx < rollout_cfg["max_steps"]:
        key, action_key = jax.random.split(key)
        action = _select_action(
            actor_apply_fn,
            actor_params,
            timestep.observation,
            rollout_cfg["deterministic_actions"],
            action_key,
        )
        env_state, next_timestep = env.step(env_state, action)
        logged_obs = _select_observation_for_logging(next_timestep)

        reward = np.asarray(next_timestep.reward)
        total_reward += reward

        positions = _maybe_extract_positions(next_timestep, logged_obs)
        if positions is not None:
            for agent_id, position in enumerate(np.asarray(positions)):
                agent_paths[agent_id].append(_to_serialisable(position))
        else:
            positions_tracked = False

        record: Dict[str, Any] = {
            "step": step_idx + 1,
            "action": _to_serialisable(action),
            "reward": _to_serialisable(reward),
            "positions": _to_serialisable(positions),
        }
        if rollout_cfg.get("store_observations", False) and hasattr(logged_obs, "agents_view"):
            record["agents_view"] = _to_serialisable(logged_obs.agents_view)
        if getattr(logged_obs, "step_count", None) is not None:
            record["step_count"] = _to_serialisable(logged_obs.step_count)

        episode_steps.append(record)
        done = bool(np.all(np.asarray(next_timestep.last())))
        timestep = next_timestep
        step_idx += 1

    episode_record = {
        "episode_index": episode_index,
        "num_steps": len(episode_steps),
        "episode_return_per_agent": _to_serialisable(total_reward),
        "terminated": done,
        "termination_reason": "env_done" if done else "rollout_limit",
        "initial_positions": _to_serialisable(initial_positions),
        "agent_paths": agent_paths if positions_tracked else None,
        "steps": episode_steps,
    }

    return episode_record, key


def _collect_rollout_trajectories(
    env: MarlEnv,
    actor_apply_fn: ActorApply,
    actor_params: FrozenDict,
    rollout_cfg: Dict[str, Any],
    key: chex.PRNGKey,
) -> Tuple[List[Dict[str, Any]], bool, chex.PRNGKey]:
    episodes: List[Dict[str, Any]] = []
    positions_logged = False

    for episode_index in range(rollout_cfg["num_episodes"]):
        episode_record, key = _rollout_single_episode(
            env,
            actor_apply_fn,
            actor_params,
            rollout_cfg,
            key,
            episode_index,
        )
        episodes.append(episode_record)
        positions_logged = positions_logged or episode_record["agent_paths"] is not None

    return episodes, positions_logged, key


def _persist_rollouts(
    episodes: List[Dict[str, Any]],
    rollout_cfg: Dict[str, Any],
    system_name: str,
) -> Optional[Path]:
    if not episodes:
        return None

    output_dir = rollout_cfg["output_dir"]
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    file_path = output_dir / f"{system_name}_rollouts_{timestamp}.json"
    payload = {
        "system_name": system_name,
        "timestamp": timestamp,
        "num_episodes": len(episodes),
        "episodes": episodes,
    }
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    return file_path


def _log_rollout_metrics(
    logger: MavaLogger,
    episodes: List[Dict[str, Any]],
    t: int,
    eval_step: int,
) -> None:
    lengths = np.asarray([episode["num_steps"] for episode in episodes], dtype=float)
    terminated = np.asarray([episode["terminated"] for episode in episodes], dtype=float)
    metrics: Metrics = {
        "rollout/episode_length": lengths,
        "rollout/terminated": terminated,
    }
    logger.log(metrics, t, eval_step, LogEvent.MISC)


def _print_rollout_summary(
    episodes: List[Dict[str, Any]],
    file_path: Optional[Path],
    positions_logged: bool,
) -> None:
    print(f"{Fore.CYAN}{Style.BRIGHT}Rollout summary{Style.RESET_ALL}")
    for episode in episodes:
        episode_return = episode["episode_return_per_agent"]
        print(
            f"  Episode {episode['episode_index']}: steps={episode['num_steps']} | "
            f"terminated={episode['terminated']} | return={episode_return}"
        )
    if file_path is not None:
        print(f"Saved rollout trajectories to: {file_path}")
    if not positions_logged:
        print(
            f"{Fore.YELLOW}Warning:{Style.RESET_ALL} could not extract explicit agent positions. "
            "Full observations were stored instead so trajectories can be reconstructed manually."
        )


def _get_rollout_config(config: DictConfig, env: MarlEnv) -> Dict[str, Any]:
    rollout_section = OmegaConf.select(config, "rollout", default=None)
    user_cfg: Dict[str, Any] = {}
    if rollout_section is not None:
        user_cfg = OmegaConf.to_container(rollout_section, resolve=True)  # type: ignore[arg-type]

    default_cfg: Dict[str, Any] = {
        "enabled": True,
        "num_episodes": 1,
        "max_steps": env.time_limit,
        "deterministic_actions": True,
        "output_dir": Path("rollouts"),
        "store_observations": False,
    }
    default_cfg.update(user_cfg)
    default_cfg["num_episodes"] = max(1, int(default_cfg["num_episodes"]))
    default_cfg["max_steps"] = max(1, int(default_cfg["max_steps"]))
    return default_cfg


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment and records rollout trajectories after training."""

    _config.logger.system_name = "ff_mappo_with_rollout"
    config = copy.deepcopy(_config)

    n_devices = len(jax.devices())

    # Create the environments for train and eval.
    env, eval_env = environments.make(config=config, add_global_state=True)

    # PRNG keys.
    key, key_e, actor_net_key, critic_net_key = jax.random.split(
        jax.random.PRNGKey(config.system.seed), num=4
    )

    # Setup learner.
    learn, actor_network, learner_state = base_ff_mappo.learner_setup(
        env, (key, actor_net_key, critic_net_key), config
    )

    # Setup evaluator.
    eval_keys = jax.random.split(key_e, n_devices)
    eval_act_fn = make_ff_eval_act_fn(actor_network.apply, config)
    evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=False)

    # Calculate total timesteps.
    config = check_total_timesteps(config)
    assert (
        config.system.num_updates > config.arch.num_evaluation
    ), "Number of updates per evaluation must be less than total number of updates."
    assert (
        config.arch.num_envs % config.system.num_minibatches == 0
    ), "Number of envs must be divisibile by number of minibatches."

    config.system.num_updates_per_eval = config.system.num_updates // config.arch.num_evaluation
    steps_per_rollout = (
        n_devices
        * config.system.num_updates_per_eval
        * config.system.rollout_length
        * config.system.update_batch_size
        * config.arch.num_envs
    )

    # Logger setup
    logger = MavaLogger(config)
    logger.log_config(OmegaConf.to_container(config, resolve=True))

    # Set up checkpointer
    save_checkpoint = config.logger.checkpointing.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(
            metadata=config,
            model_name=config.logger.system_name,
            **config.logger.checkpointing.save_args,
        )

    # Run experiment for a total number of evaluations.
    max_episode_return = -jnp.inf
    best_params: Optional[FrozenDict] = None
    rollout_cfg = _get_rollout_config(config, eval_env)
    latest_metrics: Metrics = {}

    for eval_step in range(config.arch.num_evaluation):
        # Train.
        start_time = time.time()
        learner_output = learn(learner_state)
        jax.block_until_ready(learner_output)

        # Log the results of the training.
        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        episode_metrics, ep_completed = get_final_step_metrics(learner_output.episode_metrics)
        episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time

        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        if ep_completed:
            logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
        logger.log(learner_output.train_metrics, t, eval_step, LogEvent.TRAIN)

        # Prepare for evaluation.
        trained_params = unreplicate_batch_dim(learner_state.params.actor_params)
        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys).reshape(n_devices, -1)

        # Evaluate.
        eval_metrics = evaluator(trained_params, eval_keys, {})
        logger.log(eval_metrics, t, eval_step, LogEvent.EVAL)
        episode_return = jnp.mean(eval_metrics["episode_return"])
        latest_metrics = eval_metrics

        if save_checkpoint:
            checkpointer.save(
                timestep=steps_per_rollout * (eval_step + 1),
                unreplicated_learner_state=unreplicate_n_dims(learner_output.learner_state),
                episode_return=episode_return,
            )

        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_params = copy.deepcopy(trained_params)
            max_episode_return = episode_return

        learner_state = learner_output.learner_state

    eval_performance = float(jnp.mean(latest_metrics[config.env.eval_metric]))

    # Absolute metric evaluation if requested.
    if config.arch.absolute_metric and best_params is not None:
        abs_metric_evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=True)
        eval_keys = jax.random.split(key, n_devices)
        eval_metrics = abs_metric_evaluator(best_params, eval_keys, {})
        t = int(steps_per_rollout * (config.arch.num_evaluation))
        logger.log(eval_metrics, t, config.arch.num_evaluation - 1, LogEvent.ABSOLUTE)

    # Post-training rollout logging.
    rollout_path: Optional[Path] = None
    if rollout_cfg.get("enabled", True):
        rollout_key, key_e = jax.random.split(key_e)
        if best_params is not None:
            rollout_params = flax_jax_utils.unreplicate(best_params)
        else:
            host_params = unreplicate_batch_dim(learner_state.params.actor_params)
            rollout_params = flax_jax_utils.unreplicate(host_params)
        episodes, positions_logged, _ = _collect_rollout_trajectories(
            eval_env,
            actor_network.apply,
            rollout_params,
            rollout_cfg,
            rollout_key,
        )

        if episodes:
            rollout_path = _persist_rollouts(episodes, rollout_cfg, config.logger.system_name)
            t = int(steps_per_rollout * config.arch.num_evaluation)
            _log_rollout_metrics(logger, episodes, t, config.arch.num_evaluation)
            _print_rollout_summary(episodes, rollout_path, positions_logged)

    logger.stop()

    if rollout_path is not None:
        print(f"{Fore.GREEN}{Style.BRIGHT}Trajectories written to {rollout_path}{Style.RESET_ALL}")

    return eval_performance


@hydra.main(
    config_path="../../../configs/default",
    config_name="ff_mappo.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    OmegaConf.set_struct(cfg, False)
    eval_performance = run_experiment(cfg)
    print(f"{Fore.CYAN}{Style.BRIGHT}MAPPO rollout experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
