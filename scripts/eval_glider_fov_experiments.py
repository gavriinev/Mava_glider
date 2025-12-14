
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.io as pio
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pickle
import itertools

from mava.utils import make_env as environments
from mava.networks import RecurrentActor as Actor
from mava.networks import RecurrentValueNet as Critic
from mava.networks.base import ScannedRNN
from mava.systems.ppo.types import Params, HiddenStates
from mava.utils.checkpointing import Checkpointer
from mava.utils.network_utils import get_action_head
from mava.utils.wind import thermal_centers, wind_at

# Set matplotlib backend
matplotlib.use("Agg")

DEG2RAD = jnp.pi / 180.0


def get_glider_state(state):
    """Unwrap Mava/Jumanji state wrappers to find the underlying GliderMA state."""
    curr = state
    # Limit depth to avoid infinite loops
    for depth in range(10):
        # Check for glider state attributes
        if hasattr(curr, 'position') and hasattr(curr, 'ground_speed'):
            return curr
        # Try different wrapper attributes
        if hasattr(curr, 'env_state'):
            curr = curr.env_state
        elif hasattr(curr, 'state'):
            curr = curr.state
        elif hasattr(curr, 'env'):
            curr = curr.env
        elif hasattr(curr, '_state'):
            curr = curr._state
        else:
            # Debug: print available attributes
            if depth == 0:
                print(f"State type: {type(curr)}, attributes: {dir(curr)}")
            break
    return curr


def calculate_proximity_penalty(distances_to_other_agents: np.ndarray) -> float:
    """
    Calculate average proximity penalty from distance history.
    
    Args:
        distances_to_other_agents: (steps, num_agents, num_agents-1) array of distances
        
    Returns:
        Average proximity penalty across all agents and timesteps
    """
    # Get minimum distance to any other agent for each agent at each timestep
    min_distances = np.min(distances_to_other_agents, axis=2)  # (steps, num_agents)
    
    # Calculate penalty: higher penalty for closer distances
    # Using exponential decay: penalty increases as distance decreases
    collision_distance = 1.0
    
    proximity_threshold = 20.0
        
        
    # Exponential interpolation between -1 and -100
    # At d=20: penalty=-1, at d=1: penalty=-100
    # Formula: penalty = -1 * exp(k * (20 - d)) where k = ln(100) / 19
    k = 0.25 * jnp.log(400.0) / (proximity_threshold - collision_distance)
    proximity_penalty = jnp.where(
        min_distances < proximity_threshold,
        -jnp.exp(k * (proximity_threshold - min_distances)),
        0.0
    )
    
    # Return average penalty
    return float(np.sum(proximity_penalty))


def run_single_experiment(
    cfg: DictConfig,
    params: Params,
    actor_network: Actor,
    field_of_view: float,
    field_of_view_rotation: float,
    num_steps: int = 190
) -> Tuple[Dict[str, np.ndarray], float, float]:
    """
    Run a single experiment with specified FOV parameters.
    
    Returns:
        history: Dictionary containing state history
        proximity_penalty: Average proximity penalty for this experiment
        sum_vertical_speed: Sum of vertical speeds across all agents and timesteps
    """
    print(f"  Running experiment: FOV={field_of_view:.0f}°, FOV_rotation={field_of_view_rotation:.0f}°")
    
    # Modify config before creating environment
    cfg_copy = OmegaConf.to_container(cfg, resolve=True)
    cfg_copy = OmegaConf.create(cfg_copy)
    OmegaConf.set_struct(cfg_copy, False)
    
    # Set FOV parameters in config kwargs
    cfg_copy.env.kwargs['field_of_view'] = field_of_view * DEG2RAD
    cfg_copy.env.kwargs['field_of_view_rotation'] = field_of_view_rotation * DEG2RAD
    
    # Create environment with modified config
    env, eval_env = environments.make(config=cfg_copy, add_global_state=True)
    
    # Setup
    key = jax.random.PRNGKey(cfg.system.seed)
    
    # Reset
    key, reset_key = jax.random.split(key)
    reset_keys = jnp.stack([reset_key])
    
    # Vmap reset to get batched state (batch_size=1)
    state, timestep = jax.vmap(eval_env.reset)(reset_keys)
    
    history = {
        "time": [],
        "positions": [],
        "ground_speeds": [],
        "air_speeds": [],
        "vertical_speeds": [],
        "wind_vertical_speeds": [],
        "attitudes": [],
        "controls": [],
        "angle_from_wind": [],
        "rewards": [],
        "dones": [],
        "collisions": [],
        "out_of_bounds": [],
        "low_speed": [],
        "distances_to_thermal": [],
        "distances_to_other_agents": [],
        "thermal_centers": [],
    }
    
    # JIT the actor apply
    actor_apply = jax.jit(actor_network.apply)
    
    # Vmap step function
    env_step = jax.vmap(eval_env.step)
    
    # Initialize hidden states for recurrent network
    num_agents = eval_env.num_agents
    hidden_state_dim = cfg_copy.network.hidden_state_dim
    policy_hstate = ScannedRNN.initialize_carry((1, num_agents), hidden_state_dim)
    
    for step in range(num_steps):
        # Get action
        key, action_key = jax.random.split(key)
        
        # Select action (Greedy for evaluation)
        obs_with_seq = jax.tree.map(lambda x: x[jnp.newaxis, ...], timestep.observation)
        done_env = timestep.last()  # (1,)
        done_broadcasted = jnp.broadcast_to(done_env[:, jnp.newaxis], (1, num_agents))
        done_with_seq = done_broadcasted[jnp.newaxis, ...]
        
        obs_done = (obs_with_seq, done_with_seq)
        policy_hstate, pi = actor_apply(params.actor_params, policy_hstate, obs_done)
        
        # Greedy action
        action = pi.mode()
        action_squeezed = action[0, ...]  # Remove only sequence dim, keep batch dim (1, num_agents, action_dim)
        
        # Step environment
        state, timestep = env_step(state, action_squeezed)
        
        # Extract glider state
        glider_state = get_glider_state(state)
        
        # Record history (remove batch dimension [0])
        history["time"].append(step)
        history["positions"].append(np.array(glider_state.position[0]))
        history["ground_speeds"].append(np.array(glider_state.ground_speed[0]))
        history["air_speeds"].append(np.array(glider_state.air_speed[0]))
        history["vertical_speeds"].append(np.array(glider_state.vertical_speed[0]))
        history["attitudes"].append(np.array(glider_state.attitude[0]))
        history["controls"].append(np.array(glider_state.controls[0]))
        history["angle_from_wind"].append(np.array(glider_state.angle_from_wind[0]))
        history["rewards"].append(np.array(timestep.reward[0]))
        history["dones"].append(np.array(timestep.last()[0]))
        history["distances_to_thermal"].append(np.array(glider_state.distances_to_thermal[0]))
        history["distances_to_other_agents"].append(np.array(glider_state.distances_to_other_agents[0]))
        
        # Calculate thermal center for this step (using first agent's altitude)
        positions_batch = glider_state.position[0]  # (num_agents, 3)
        step_time = float(step + 8)
        altitude = float(positions_batch[0, 2])  # First agent's altitude
        thermal_center_full = thermal_centers(eval_env.params.wind_model, altitude, step_time)
        thermal_center_xyz = np.array(thermal_center_full.reshape(-1, 3)[0])  # (3,) - x, y, z
        history["thermal_centers"].append(thermal_center_xyz)
        
        # Calculate wind vertical speed at each agent's position
        wind_vertical_at_step = []
        for agent_idx in range(positions_batch.shape[0]):
            agent_pos = positions_batch[agent_idx]  # (3,)
            wind_vec = wind_at(eval_env.params.wind_model, agent_pos, step_time)  # (3,) - wind velocity
            wind_vertical_at_step.append(float(wind_vec[2]))  # z-component is vertical
        wind_vertical_at_step = np.array(wind_vertical_at_step)
        history["wind_vertical_speeds"].append(wind_vertical_at_step)
        
        if timestep.last()[0]:
            break
    
    # Convert history to numpy arrays
    for k, v in history.items():
        if len(v) > 0:
            history[k] = np.array(v)
    
    # Calculate proximity penalty
    if len(history["distances_to_other_agents"]) > 0:
        proximity_penalty = calculate_proximity_penalty(history["distances_to_other_agents"])
    else:
        proximity_penalty = 0.0
    
    # Calculate sum of vertical speeds (sum across all agents and timesteps)
    if len(history["vertical_speeds"]) > 0:
        sum_vertical_speed = float(np.sum(history["vertical_speeds"]))
    else:
        sum_vertical_speed = 0.0
    
    return history, proximity_penalty, sum_vertical_speed


@hydra.main(config_path="../mava/configs/default", config_name="rec_mappo.yaml", version_base="1.2")
def main(cfg: DictConfig):
    # Allow dynamic attributes
    OmegaConf.set_struct(cfg, False)
    
    print(f"Initializing environment with config: {cfg.env.env_name}")
    
    # Create env for network setup
    env, eval_env = environments.make(config=cfg, add_global_state=True)
    
    # Set num_agents
    cfg.system.num_agents = eval_env.num_agents
    
    # Setup networks
    key = jax.random.PRNGKey(cfg.system.seed)
    key, actor_key, critic_key = jax.random.split(key, 3)
    
    actor_pre_torso = hydra.utils.instantiate(cfg.network.actor_network.pre_torso)
    actor_post_torso = hydra.utils.instantiate(cfg.network.actor_network.post_torso)
    action_head, _ = get_action_head(eval_env.action_spec)
    actor_action_head = hydra.utils.instantiate(action_head, action_dim=eval_env.action_dim)
    actor_network = Actor(
        pre_torso=actor_pre_torso,
        post_torso=actor_post_torso, 
        action_head=actor_action_head,
        hidden_state_dim=cfg.network.hidden_state_dim
    )
    
    critic_pre_torso = hydra.utils.instantiate(cfg.network.critic_network.pre_torso)
    critic_post_torso = hydra.utils.instantiate(cfg.network.critic_network.post_torso)
    critic_network = Critic(
        pre_torso=critic_pre_torso,
        post_torso=critic_post_torso,
        centralised_critic=True,
        hidden_state_dim=cfg.network.hidden_state_dim
    )
    
    # Init params
    obs = eval_env.observation_spec.generate_value()
    init_x = jax.tree.map(lambda x: x[jnp.newaxis, jnp.newaxis, ...], obs)
    
    init_policy_hstate = ScannedRNN.initialize_carry((1, cfg.system.num_agents), cfg.network.hidden_state_dim)
    init_critic_hstate = ScannedRNN.initialize_carry((1, cfg.system.num_agents), cfg.network.hidden_state_dim)
    init_done = jnp.zeros((1, 1, cfg.system.num_agents), dtype=bool)
    init_obs_done = (init_x, init_done)
    
    actor_params = actor_network.init(actor_key, init_policy_hstate, init_obs_done)
    critic_params = critic_network.init(critic_key, init_critic_hstate, init_obs_done)
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
    
    # Define experiment parameters
    fov_values = [120.0, 90.0, 60.0, 30.0, 10.0]  # degrees
    fov_rotation_values = [0.0, 30.0, 45.0, 60.0, 90.0]  # degrees
    
    num_steps = 190
    if hasattr(cfg.system, 'rollout_length'):
        num_steps = cfg.system.rollout_length
    
    # Create output directory
    output_dir = Path("outputs/fov_experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nRunning {len(fov_values) * len(fov_rotation_values)} experiments...")
    print(f"FOV values: {fov_values}")
    print(f"FOV rotation values: {fov_rotation_values}")
    
    # Store results
    results = []
    all_histories = {}
    
    # Run experiments
    for fov, fov_rot in itertools.product(fov_values, fov_rotation_values):
        exp_name = f"fov_{fov:.0f}_rot_{fov_rot:.0f}"
        
        history, proximity_penalty, sum_vertical_speed = run_single_experiment(
            cfg=cfg,
            params=params,
            actor_network=actor_network,
            field_of_view=fov,
            field_of_view_rotation=fov_rot,
            num_steps=num_steps
        )
        
        results.append({
            'fov': fov,
            'fov_rotation': fov_rot,
            'proximity_penalty': proximity_penalty,
            'sum_vertical_speed': sum_vertical_speed
        })
        
        all_histories[exp_name] = history
        
        print(f"    Proximity penalty: {proximity_penalty:.4f}, Sum vertical speed: {sum_vertical_speed:.4f}")
    
    # Save all results and histories
    results_file = output_dir / "experiment_results.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump({
            'results': results,
            'histories': all_histories
        }, f)
    print(f"\nSaved results to {results_file}")
    
    # Create 3D plot
    print("\nGenerating 3D plot...")
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract data for plotting
    fov_plot = [r['fov'] for r in results]
    fov_rot_plot = [r['fov_rotation'] for r in results]
    penalty_plot = [r['proximity_penalty'] for r in results]
    
    # Create scatter plot
    scatter = ax.scatter(
        fov_plot, 
        fov_rot_plot, 
        penalty_plot,
        c=penalty_plot,
        cmap='viridis',
        s=100,
        alpha=0.6,
        edgecolors='black'
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Proximity Penalty', rotation=270, labelpad=20)
    
    # Set labels
    ax.set_xlabel('Field of View (degrees)', fontsize=12)
    ax.set_ylabel('FOV Rotation (degrees)', fontsize=12)
    ax.set_zlabel('Proximity Penalty', fontsize=12)
    ax.set_title('Proximity Penalty vs FOV Parameters', fontsize=14, pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save plot
    plot_file = output_dir / "proximity_penalty_3d.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Saved 3D plot to {plot_file}")
    plt.close()
    
    # Create surface plot as well
    print("Generating surface plot...")
    
    # Prepare data for surface plot
    fov_unique = sorted(set(fov_plot))
    fov_rot_unique = sorted(set(fov_rot_plot))
    
    # Create meshgrid
    FOV, FOV_ROT = np.meshgrid(fov_unique, fov_rot_unique)
    PENALTY = np.zeros_like(FOV)
    
    # Fill penalty values
    for r in results:
        i = fov_rot_unique.index(r['fov_rotation'])
        j = fov_unique.index(r['fov'])
        PENALTY[i, j] = r['proximity_penalty']
    
    # Create surface plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(
        FOV, 
        FOV_ROT, 
        PENALTY,
        cmap='viridis',
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Add colorbar
    cbar = plt.colorbar(surf, ax=ax, pad=0.1, shrink=0.5)
    cbar.set_label('Proximity Penalty', rotation=270, labelpad=20)
    
    # Set labels
    ax.set_xlabel('Field of View (degrees)', fontsize=12)
    ax.set_ylabel('FOV Rotation (degrees)', fontsize=12)
    ax.set_zlabel('Proximity Penalty', fontsize=12)
    ax.set_title('Proximity Penalty Surface Plot', fontsize=14, pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save plot
    surface_plot_file = output_dir / "proximity_penalty_surface.png"
    plt.savefig(surface_plot_file, dpi=150, bbox_inches='tight')
    print(f"Saved surface plot to {surface_plot_file}")
    plt.close()
    
    # Create surface plot for mean vertical speeds
    print("Generating vertical speed surface plot...")
    
    # Prepare data for vertical speed surface plot
    VERTICAL_SPEED = np.zeros_like(FOV)
    
    # Fill vertical speed values
    for r in results:
        i = fov_rot_unique.index(r['fov_rotation'])
        j = fov_unique.index(r['fov'])
        VERTICAL_SPEED[i, j] = r['sum_vertical_speed']
    
    # Create surface plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(
        FOV, 
        FOV_ROT, 
        VERTICAL_SPEED,
        cmap='RdYlGn',
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Add colorbar
    cbar = plt.colorbar(surf, ax=ax, pad=0.1, shrink=0.5)
    cbar.set_label('Sum Vertical Speed (m/s)', rotation=270, labelpad=20)
    
    # Set labels
    ax.set_xlabel('Field of View (degrees)', fontsize=12)
    ax.set_ylabel('FOV Rotation (degrees)', fontsize=12)
    ax.set_zlabel('Sum Vertical Speed (m/s)', fontsize=12)
    ax.set_title('Sum Vertical Speed Surface Plot', fontsize=14, pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save plot
    vspeed_surface_plot_file = output_dir / "vertical_speed_surface.png"
    plt.savefig(vspeed_surface_plot_file, dpi=150, bbox_inches='tight')
    print(f"Saved vertical speed surface plot to {vspeed_surface_plot_file}")
    plt.close()
    
    # Create interactive Plotly 3D surface plots
    print("\nGenerating interactive Plotly plots...")
    
    # Interactive surface plot for Proximity Penalty
    fig_proximity = go.Figure(data=[go.Surface(
        x=FOV,
        y=FOV_ROT,
        z=PENALTY,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Proximity Penalty", x=1.1),
        hovertemplate='FOV: %{x}°<br>FOV Rot: %{y}°<br>Proximity Penalty: %{z:.4f}<extra></extra>'
    )])
    
    fig_proximity.update_layout(
        title='Interactive Proximity Penalty Surface Plot',
        scene=dict(
            xaxis_title='Field of View (degrees)',
            yaxis_title='FOV Rotation (degrees)',
            zaxis_title='Proximity Penalty',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        width=1000,
        height=800
    )
    
    proximity_interactive_file = output_dir / "proximity_penalty_interactive.html"
    pio.write_html(fig_proximity, file=str(proximity_interactive_file), auto_open=False)
    print(f"Saved interactive proximity penalty plot to {proximity_interactive_file}")
    
    # Interactive surface plot for Vertical Speed
    fig_vspeed = go.Figure(data=[go.Surface(
        x=FOV,
        y=FOV_ROT,
        z=VERTICAL_SPEED,
        colorscale='RdYlGn',
        showscale=True,
        colorbar=dict(title="Sum Vertical Speed (m/s)", x=1.1),
        hovertemplate='FOV: %{x}°<br>FOV Rot: %{y}°<br>Sum Vert Speed: %{z:.4f} m/s<extra></extra>'
    )])
    
    fig_vspeed.update_layout(
        title='Interactive Sum Vertical Speed Surface Plot',
        scene=dict(
            xaxis_title='Field of View (degrees)',
            yaxis_title='FOV Rotation (degrees)',
            zaxis_title='Sum Vertical Speed (m/s)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        width=1000,
        height=800
    )
    
    vspeed_interactive_file = output_dir / "vertical_speed_interactive.html"
    pio.write_html(fig_vspeed, file=str(vspeed_interactive_file), auto_open=False)
    print(f"Saved interactive vertical speed plot to {vspeed_interactive_file}")
    
    # Create combined interactive scatter plot
    fig_combined = go.Figure()
    
    # Add proximity penalty scatter
    fig_combined.add_trace(go.Scatter3d(
        x=fov_plot,
        y=fov_rot_plot,
        z=penalty_plot,
        mode='markers',
        name='Proximity Penalty',
        marker=dict(
            size=8,
            color=penalty_plot,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Proximity Penalty", x=1.15, len=0.5, y=0.75)
        ),
        hovertemplate='FOV: %{x}°<br>FOV Rot: %{y}°<br>Proximity: %{z:.4f}<extra></extra>'
    ))
    
    fig_combined.update_layout(
        title='Interactive 3D Scatter: Proximity Penalty vs FOV Parameters',
        scene=dict(
            xaxis_title='Field of View (degrees)',
            yaxis_title='FOV Rotation (degrees)',
            zaxis_title='Proximity Penalty',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        width=1000,
        height=800
    )
    
    combined_interactive_file = output_dir / "combined_scatter_interactive.html"
    pio.write_html(fig_combined, file=str(combined_interactive_file), auto_open=False)
    print(f"Saved interactive combined scatter plot to {combined_interactive_file}")
    
    # Print summary table
    print("\n" + "="*90)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*90)
    print(f"{'FOV (deg)':<12} {'FOV Rot (deg)':<15} {'Proximity Penalty':<20} {'Sum Vert Speed (m/s)':<20}")
    print("-"*90)
    for r in results:
        print(f"{r['fov']:<12.0f} {r['fov_rotation']:<15.0f} {r['proximity_penalty']:<20.4f} {r['sum_vertical_speed']:<20.4f}")
    print("="*90)
    
    print(f"\nAll experiments completed!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
