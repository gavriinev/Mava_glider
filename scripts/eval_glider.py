
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Dict, List, Any

from mava.utils import make_env as environments
from mava.networks import FeedForwardActor as Actor
from mava.networks import FeedForwardValueNet as Critic
from mava.systems.ppo.types import Params
from mava.utils.checkpointing import Checkpointer
from mava.utils.network_utils import get_action_head

# Set matplotlib backend
matplotlib.use("Agg")

def plot_state_history(history: Dict[str, np.ndarray], output_dir: Path, num_agents: int) -> Dict[str, Path]:
    """Generate timeseries and 3D trajectory plots from recorded history."""

    output_dir.mkdir(parents=True, exist_ok=True)

    times = history["time"]
    positions = history["positions"]  # (steps, num_agents, 3)
    speeds = history["speeds"]  # (steps, num_agents)
    vertical_speeds = history["vertical_speeds"]  # (steps, num_agents)
    attitudes = history["attitudes"]  # (steps, num_agents, 2)
    controls = history["controls"]  # (steps, num_agents, 3)
    rewards = history["rewards"]  # (steps, num_agents)
    distances_to_thermal = history["distances_to_thermal"]  # (steps, num_agents)
    # min_inter_agent_distances = history["min_inter_agent_distances"]  # (steps, num_agents)

    # Create color palette for agents
    colors = plt.cm.tab10(np.linspace(0, 1, num_agents))

    # Plot timeseries for each metric
    fig, axes = plt.subplots(4, 3, figsize=(18, 14), sharex=True)
    axes = axes.flatten()

    # Plot metrics for each agent
    metrics = [
        (positions[:, :, 0], "Position X (m)"),
        (positions[:, :, 1], "Position Y (m)"),
        (positions[:, :, 2], "Altitude Z (m)"),
        (speeds[:, :], "Speed (m/s)"),
        (vertical_speeds[:, :], "Vertical Speed (m/s)"),
        (attitudes[:, :, 0], "Glide Angle (rad)"),
        (attitudes[:, :, 1], "Side Angle (rad)"),
        (controls[:, :, 0], "Bank Control (rad)"),
        (controls[:, :, 1], "Attack Control (rad)"),
        (controls[:, :, 2], "Sideslip Control (rad)"),
        (rewards[:, :], "Reward"),
        (distances_to_thermal[:, :], "Distance to Nearest Thermal (m)"),
        # (min_inter_agent_distances[:, :], "Min Inter-Agent Distance (m)"),
    ]

    for idx, (data, title) in enumerate(metrics):
        ax = axes[idx]
        for agent_idx in range(num_agents):
            ax.plot(times, data[:, agent_idx], linewidth=1.0, 
                   color=colors[agent_idx], label=f"Agent {agent_idx}")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.4)
        if idx == 0:
            ax.legend()


    for ax in axes[9:11]:
        ax.set_xlabel("Step")

    fig.suptitle(f"Multi-Agent Glider State Trajectories ({num_agents} agents)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    timeseries_path = output_dir / "ma_state_timeseries.png"
    fig.savefig(timeseries_path, dpi=150)
    plt.close(fig)

    # Create 3D trajectory plot using matplotlib
    fig3d = plt.figure(figsize=(12, 10))
    ax3d = fig3d.add_subplot(111, projection="3d")
    
    for agent_idx in range(num_agents):
        ax3d.plot(
            positions[:, agent_idx, 0], 
            positions[:, agent_idx, 1], 
            positions[:, agent_idx, 2],
            linewidth=2.0,
            color=colors[agent_idx],
            label=f"Agent {agent_idx}"
        )
        # Mark start and end
        ax3d.scatter(
            positions[0, agent_idx, 0], 
            positions[0, agent_idx, 1], 
            positions[0, agent_idx, 2],
            s=100, marker='o', color=colors[agent_idx]
        )
        ax3d.scatter(
            positions[-1, agent_idx, 0], 
            positions[-1, agent_idx, 1], 
            positions[-1, agent_idx, 2],
            s=100, marker='x', color=colors[agent_idx]
        )
    
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_title("Multi-Agent Glider Position Trajectories")
    ax3d.legend()
    ax3d.grid(True, linestyle="--", alpha=0.3)

    pos3d_path = output_dir / "ma_position_3d.png"
    fig3d.savefig(pos3d_path, dpi=150)
    plt.close(fig3d)

    # Create interactive 3D plot with plotly
    fig_plotly = go.Figure()
    
    for agent_idx in range(num_agents):
        # Add trajectory
        fig_plotly.add_trace(go.Scatter3d(
            x=positions[:, agent_idx, 0],
            y=positions[:, agent_idx, 1],
            z=positions[:, agent_idx, 2],
            mode='lines',
            name=f'Agent {agent_idx}',
            line=dict(width=4)
        ))
        
        # Add start marker
        fig_plotly.add_trace(go.Scatter3d(
            x=[positions[0, agent_idx, 0]],
            y=[positions[0, agent_idx, 1]],
            z=[positions[0, agent_idx, 2]],
            mode='markers',
            name=f'Agent {agent_idx} Start',
            marker=dict(size=8, symbol='circle'),
            showlegend=False
        ))
        
        # Add end marker
        fig_plotly.add_trace(go.Scatter3d(
            x=[positions[-1, agent_idx, 0]],
            y=[positions[-1, agent_idx, 1]],
            z=[positions[-1, agent_idx, 2]],
            mode='markers',
            name=f'Agent {agent_idx} End',
            marker=dict(size=8, symbol='x'),
            showlegend=False
        ))

    fig_plotly.update_layout(
        title=f"Multi-Agent Glider 3D Trajectories ({num_agents} agents)",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode='data'
        ),
        showlegend=True
    )

    plotly_path = output_dir / "ma_simulation_3d.html"
    pio.write_html(fig_plotly, file=str(plotly_path), auto_open=False)

    # Create top-down view (XY plane)
    fig_topdown = plt.figure(figsize=(10, 10))
    ax_top = fig_topdown.add_subplot(111)
    
    for agent_idx in range(num_agents):
        ax_top.plot(
            positions[:, agent_idx, 0], 
            positions[:, agent_idx, 1],
            linewidth=2.0,
            color=colors[agent_idx],
            label=f"Agent {agent_idx}"
        )
        # Mark start and end
        ax_top.scatter(
            positions[0, agent_idx, 0], 
            positions[0, agent_idx, 1],
            s=100, marker='o', color=colors[agent_idx]
        )
        ax_top.scatter(
            positions[-1, agent_idx, 0], 
            positions[-1, agent_idx, 1],
            s=100, marker='x', color=colors[agent_idx]
        )
    
    ax_top.set_xlabel("X (m)")
    ax_top.set_ylabel("Y (m)")
    ax_top.set_title("Multi-Agent Glider Trajectories (Top View)")
    ax_top.legend()
    ax_top.grid(True, linestyle="--", alpha=0.3)
    ax_top.set_aspect('equal')

    topdown_path = output_dir / "ma_position_topdown.png"
    fig_topdown.savefig(topdown_path, dpi=150)
    plt.close(fig_topdown)

    return {
        "timeseries": timeseries_path,
        "position_3d": pos3d_path,
        "position_topdown": topdown_path,
        "plotly_3d": plotly_path,
    }

def get_glider_state(state):
    """Unwrap Mava/Jumanji state wrappers to find the underlying GliderMA state."""
    curr = state
    # Limit depth to avoid infinite loops
    for _ in range(10):
        if hasattr(curr, 'position') and hasattr(curr, 'speed'):
            return curr
        if hasattr(curr, 'env_state'):
            curr = curr.env_state
        elif hasattr(curr, 'state'):
            curr = curr.state
        else:
            break
    return state

@hydra.main(config_path="../mava/configs/default", config_name="ff_ippo.yaml", version_base="1.2")
def main(cfg: DictConfig):
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)
    
    print(f"Initializing environment with config: {cfg.env.env_name}")
    
    # Create env
    # We use the eval_env which is typically what we want for testing
    env, eval_env = environments.make(config=cfg, add_global_state=True)
    
    # Set num_agents
    cfg.system.num_agents = eval_env.num_agents
    
    # Setup networks
    key = jax.random.PRNGKey(cfg.system.seed)
    key, actor_key, critic_key = jax.random.split(key, 3)
    
    actor_torso = hydra.utils.instantiate(cfg.network.actor_network.pre_torso)
    action_head, _ = get_action_head(eval_env.action_spec)
    actor_action_head = hydra.utils.instantiate(action_head, action_dim=eval_env.action_dim)
    actor_network = Actor(torso=actor_torso, action_head=actor_action_head)
    
    critic_torso = hydra.utils.instantiate(cfg.network.critic_network.pre_torso)
    critic_network = Critic(torso=critic_torso, centralised_critic=True)
    
    # Init params
    obs = eval_env.observation_spec.generate_value()
    # Add batch dim for init
    init_x = jax.tree_map(lambda x: x[jnp.newaxis, ...], obs)
    
    actor_params = actor_network.init(actor_key, init_x)
    critic_params = critic_network.init(critic_key, init_x)
    params = Params(actor_params, critic_params)
    
    # Load checkpoint
    if cfg.logger.checkpointing.load_model:
        print(f"Loading checkpoint from {cfg.logger.system_name}...")
        checkpointer = Checkpointer(
            model_name=cfg.logger.system_name,
            **cfg.logger.checkpointing.load_args,
        )
        params, _ = checkpointer.restore_params(input_params=params)
        print("Checkpoint loaded successfully.")
    else:
        print("WARNING: logger.checkpointing.load_model is False. Using random weights.")

    # Rollout
    print("Starting rollout...")
    
    # Reset
    key, reset_key = jax.random.split(key)
    # Create a batch of 1 key
    reset_keys = jnp.stack([reset_key])
    
    # Vmap reset to get batched state (batch_size=1)
    state, timestep = jax.vmap(eval_env.reset)(reset_keys)
    
    history = {
        "time": [],
        "positions": [],
        "speeds": [],
        "vertical_speeds": [],
        "attitudes": [],
        "controls": [],
        "rewards": [],
        "dones": [],
        "collisions": [],
        "out_of_bounds": [],
        "low_speed": [],
        "distances_to_thermal": [],
        # "min_inter_agent_distances": [],
    }
    
    # JIT the actor apply
    actor_apply = jax.jit(actor_network.apply)
    
    # Vmap step function
    env_step = jax.vmap(eval_env.step)
    
    num_steps = 200 # Default rollout length
    if hasattr(cfg.system, 'rollout_length'):
        num_steps = cfg.system.rollout_length
        
    print(f"Simulating for {num_steps} steps...")
    
    for step in range(num_steps):
        # Get action
        key, action_key = jax.random.split(key)
        
        # Select action (Greedy for evaluation)
        # timestep.observation is (1, num_agents, obs_dim)
        pi = actor_apply(params.actor_params, timestep.observation)
        action = pi.mode() # (1, num_agents)
        
        # Step
        next_state, next_timestep = env_step(state, action)
        
        # Extract info
        # Note: state and timestep are batched (dim 0 is env index). We take index 0.
        
        glider_state = get_glider_state(next_state)
        
        if not hasattr(glider_state, 'position'):
             print("Warning: Could not find 'position' in state. Skipping step data collection.")
             state = next_state
             timestep = next_timestep
             continue
             
        # Append to history (taking 0-th env)
        # Ensure shapes are (num_agents, ...)
        
        history["time"].append(step)

        distances_to_thermal = np.array(glider_state.distances_to_thermal[0])
        if distances_to_thermal.ndim == 0: distances_to_thermal = distances_to_thermal[np.newaxis]
        history["distances_to_thermal"].append(distances_to_thermal)

        # min_inter_agent_distances = np.array(glider_state.min_inter_agent_distances[0])
        # if min_inter_agent_distances.ndim == 0: min_inter_agent_distances = min_inter_agent_distances[np.newaxis]
        # history["min_inter_agent_distances"].append(min_inter_agent_distances)

        pos = np.array(glider_state.position[0])
        if pos.ndim == 1: pos = pos[np.newaxis, :]
        history["positions"].append(pos)
        
        speed = np.array(glider_state.speed[0])
        if speed.ndim == 0: speed = speed[np.newaxis] # Handle scalar
        history["speeds"].append(speed)
        
        # Calculate vertical speeds
        # speed * sin(glide_angle)
        # attitude is (1, num_agents, 2) -> [0] gives (num_agents, 2)
        att = np.array(glider_state.attitude[0])
        if att.ndim == 1: att = att[np.newaxis, :]
        
        
        vertical_speed = np.array(glider_state.vertical_speed[0])
        if vertical_speed.ndim == 0: vertical_speed = vertical_speed[np.newaxis] # Handle scalar
        history["vertical_speeds"].append(vertical_speed)
        
        history["attitudes"].append(att)
        
        ctrl = np.array(glider_state.controls[0])
        if ctrl.ndim == 1: ctrl = ctrl[np.newaxis, :]
        history["controls"].append(ctrl)
        
        rew = np.array(next_timestep.reward[0])
        if rew.ndim == 0: rew = rew[np.newaxis]
        history["rewards"].append(rew)
        
        d = np.array(next_timestep.last()[0])
        if d.ndim == 0: d = d[np.newaxis]
        history["dones"].append(d) 
        
        # Try to find failure info
        # Check extras
        extras = next_timestep.extras
        
        def find_key(d, key_name):
            if key_name in d: return d[key_name]
            for k, v in d.items():
                if isinstance(v, dict):
                    res = find_key(v, key_name)
                    if res is not None: return res
            return None

        collisions = find_key(extras, 'collisions')
        if collisions is not None:
             c = np.array(collisions[0])
             if c.ndim == 0: c = c[np.newaxis]
             history["collisions"].append(c)
        
        out_of_bounds = find_key(extras, 'out_of_bounds')
        if out_of_bounds is not None:
             o = np.array(out_of_bounds[0])
             if o.ndim == 0: o = o[np.newaxis]
             history["out_of_bounds"].append(o)
             
        low_speed = find_key(extras, 'low_speed')
        if low_speed is not None:
             l = np.array(low_speed[0])
             if l.ndim == 0: l = l[np.newaxis]
             history["low_speed"].append(l)

        state = next_state
        timestep = next_timestep
        
        # Check done
        if np.any(timestep.last()[0]):
            print(f"Episode finished at step {step}")
            break
            
    # Convert history to numpy arrays
    for k, v in history.items():
        if len(v) > 0:
            history[k] = np.array(v)
        else:
            history[k] = np.array([])
        
    # Plot
    output_dir = Path("outputs/eval_plots")
    print(f"Generating plots in {output_dir}...")
    plot_state_history(history, output_dir, cfg.system.num_agents)
    print(f"Done.")

if __name__ == "__main__":
    main()
