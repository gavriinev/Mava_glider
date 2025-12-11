"""Roll out the multi-agent glider environment and visualize the results."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from jaxmarl.environments.glider_ma import GliderMA

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_rollout(num_steps: int = 20, seed: int = 1, num_agents: int = 3) -> Dict[str, np.ndarray]:
    """Simulate the multi-agent glider environment and collect state history."""

    env = GliderMA(num_agents=num_agents)
    key = jax.random.PRNGKey(seed)

    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)
    init_state = state

    history: Dict[str, List[np.ndarray]] = {
        "time": [],
        "positions": [],  # Will store all agents' positions
        "ground_speeds": [],
        "air_speeds": [],
        "vertical_speeds": [],
        "attitudes": [],
        "controls": [],
        "angle_from_wind": [],
        "rewards": [],
        "dones": [],
        "out_of_bounds": [],
        "low_speed": [],
        "observations": [],
        "distances_to_thermal": [],
        "distances_to_other_agents": [],
    }

    for step_idx in range(num_steps):
        key, step_key = jax.random.split(key)
        
        # Simple heuristic action: slight bank to circle
        actions = {
            agent: jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32) 
            for agent in env.agents
        }

        obs, next_state, rewards, dones, info = env.step_env(step_key, state, actions)
        
        # Print observations for each step
        print(f"\n--- Step {step_idx} ---")
        for agent_idx, agent in enumerate(env.agents):
            print(f"Agent {agent_idx} obs shape: {obs[agent].shape}, obs: {obs[agent]}")

        history["time"].append(step_idx)
        history["positions"].append(np.asarray(next_state.position, dtype=np.float32))
        history["ground_speeds"].append(np.asarray(next_state.ground_speed, dtype=np.float32))
        history["air_speeds"].append(np.asarray(next_state.air_speed, dtype=np.float32))
        history["vertical_speeds"].append(np.asarray(next_state.vertical_speed, dtype=np.float32))
        
        history["attitudes"].append(np.asarray(next_state.attitude, dtype=np.float32))
        history["controls"].append(np.asarray(next_state.controls, dtype=np.float32))
        history["angle_from_wind"].append(np.asarray(next_state.angle_from_wind, dtype=np.float32))
        
        # Convert rewards dict to array
        rewards_array = np.array([rewards[agent] for agent in env.agents], dtype=np.float32)
        history["rewards"].append(rewards_array)
        
        # Store info
        history["dones"].append(np.asarray(dones, dtype=bool))
        history["out_of_bounds"].append(np.asarray(info["out_of_bounds"], dtype=bool))
        history["low_speed"].append(np.asarray(info["low_speed"], dtype=bool))
        history["distances_to_thermal"].append(np.asarray(next_state.distances_to_thermal, dtype=np.float32))
        history["distances_to_other_agents"].append(np.asarray(next_state.distances_to_other_agents, dtype=np.float32))

        history["observations"].append(np.asarray(obs, dtype=dict))

        state = next_state

        if bool(dones["__all__"]):
            print(f"Episode ended at step {step_idx}")
            break

    return {
        "time": np.array(history["time"]),
        "positions": np.stack(history["positions"]),  # (steps, num_agents, 3)
        "ground_speeds": np.stack(history["ground_speeds"]),  # (steps, num_agents)
        "air_speeds": np.stack(history["air_speeds"]),  # (steps, num_agents)
        "vertical_speeds": np.stack(history["vertical_speeds"]),  # (steps, num_agents)
        "attitudes": np.stack(history["attitudes"]),  # (steps, num_agents, 2)
        "controls": np.stack(history["controls"]),  # (steps, num_agents, 3)
        "angle_from_wind": np.stack(history["angle_from_wind"]),  # (steps, num_agents)
        "rewards": np.stack(history["rewards"]),  # (steps, num_agents)
        "dones": np.stack(history["dones"]),  # (steps, num_agents)
        "out_of_bounds": np.stack(history["out_of_bounds"]),  # (steps, num_agents)
        "low_speed": np.stack(history["low_speed"]),  # (steps, num_agents)
        "initial_state": init_state,
        "observations": np.stack(history["observations"]),  # (steps, num_agents, obs_dim)
        "distances_to_thermal": np.stack(history["distances_to_thermal"]),  # (steps, num_agents)
        "distances_to_other_agents": np.stack(history["distances_to_other_agents"]),  # (steps, num_agents, num_agents-1)
    }


def plot_state_history(history: Dict[str, np.ndarray], output_dir: Path, num_agents: int) -> Dict[str, Path]:
    """Generate timeseries and 3D trajectory plots from recorded history."""

    output_dir.mkdir(parents=True, exist_ok=True)

    times = history["time"]
    positions = history["positions"]  # (steps, num_agents, 3)
    ground_speeds = history["ground_speeds"]  # (steps, num_agents)
    air_speeds = history["air_speeds"]  # (steps, num_agents)
    vertical_speeds = history["vertical_speeds"]  # (steps, num_agents)
    attitudes = history["attitudes"]  # (steps, num_agents, 2)
    controls = history["controls"]  # (steps, num_agents, 3)
    angle_from_wind = history["angle_from_wind"]  # (steps, num_agents)
    rewards = history["rewards"]  # (steps, num_agents)
    distances_to_thermal = history["distances_to_thermal"]  # (steps, num_agents)
    distances_to_other_agents = history["distances_to_other_agents"]  # (steps, num_agents, num_agents-1)

    # Create color palette for agents
    colors = plt.cm.tab10(np.linspace(0, 1, num_agents))

    # Plot timeseries for each metric
    fig, axes = plt.subplots(5, 3, figsize=(18, 18), sharex=True)
    axes = axes.flatten()

    # Plot metrics for each agent
    metrics = [
        (positions[:, :, 0], "Position X (m)"),
        (positions[:, :, 1], "Position Y (m)"),
        (positions[:, :, 2], "Altitude Z (m)"),
        (ground_speeds[:, :], "Ground Speed (m/s)"),
        (air_speeds[:, :], "Air Speed (m/s)"),
        (vertical_speeds[:, :], "Vertical Speed (m/s)"),
        (attitudes[:, :, 0], "Glide Angle (rad)"),
        (attitudes[:, :, 1], "Side Angle (rad)"),
        (controls[:, :, 0], "Bank Control (rad)"),
        (controls[:, :, 1], "Attack Control (rad)"),
        (controls[:, :, 2], "Sideslip Control (rad)"),
        (angle_from_wind[:, :], "Angle from Wind (rad)"),
        (rewards[:, :], "Reward"),
        (distances_to_thermal[:, :], "Distance to Thermal (m)"),
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

    # Plot min distances to other agents (for each agent, plot min distance to any other agent)
    if distances_to_other_agents.shape[2] > 0:  # Only if there are other agents
        ax = axes[14]
        min_distances_to_others = np.min(distances_to_other_agents, axis=2)  # (steps, num_agents)
        for agent_idx in range(num_agents):
            ax.plot(times, min_distances_to_others[:, agent_idx], linewidth=1.0,
                   color=colors[agent_idx], label=f"Agent {agent_idx}")
        ax.set_title("Min Distance to Other Agents (m)")
        ax.grid(True, linestyle="--", alpha=0.4)

    for ax in axes[12:]:
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


def main() -> None:
    """Roll out the multi-agent glider environment, plot results, and save figures."""
    
    num_agents = 4
    num_steps = 20
    
    print(f"Running multi-agent glider rollout with {num_agents} agents for {num_steps} steps...")
    history = run_rollout(num_steps=num_steps, seed=42, num_agents=num_agents)

    # Print initial state
    print("\n" + "="*80)
    print("INITIAL STATE:")
    print("="*80)
    init_state = history['initial_state']
    print(f"Positions:\n{np.asarray(init_state.position)}")
    print(f"Ground speeds: {np.asarray(init_state.ground_speed)}")
    print(f"Air speeds: {np.asarray(init_state.air_speed)}")
    print(f"Vertical speeds: {np.asarray(init_state.vertical_speed)}")
    print(f"Attitudes (glide_angle, side_angle):\n{np.asarray(init_state.attitude)}")
    print(f"Controls (bank, attack, sideslip):\n{np.asarray(init_state.controls)}")
    print(f"Angle from wind: {np.asarray(init_state.angle_from_wind)}")
    print(f"Distances to thermal: {np.asarray(init_state.distances_to_thermal)}")
    print(f"Distances to other agents:\n{np.asarray(init_state.distances_to_other_agents)}")
    print(f"Step: {init_state.step}")

    # Print observations (first step)
    print("\n" + "="*80)
    print("OBSERVATIONS (First Step):")
    print("="*80)
    first_obs = history['observations'][0]
    for agent_idx in range(num_agents):
        print(f"\nAgent {agent_idx} observation shape: {first_obs[f'agent_{agent_idx}'].shape}")
        print(f"Agent {agent_idx} observation:\n{first_obs[f'agent_{agent_idx}']}")

    # Print final state statistics
    print("\n" + "="*80)
    print("FINAL STATE STATISTICS:")
    print("="*80)
    print(f"Completed {len(history['time'])} steps")
    print(f"\nFinal positions:\n{history['positions'][-1]}")
    print(f"Final ground speeds: {history['ground_speeds'][-1]}")
    print(f"Final air speeds: {history['air_speeds'][-1]}")
    print(f"Final vertical speeds: {history['vertical_speeds'][-1]}")
    print(f"Final attitudes:\n{history['attitudes'][-1]}")
    print(f"Final controls:\n{history['controls'][-1]}")
    print(f"Final angle from wind: {history['angle_from_wind'][-1]}")
    print(f"Final distances to thermal: {history['distances_to_thermal'][-1]}")
    print(f"Final distances to other agents:\n{history['distances_to_other_agents'][-1]}")
    print(f"\nTotal rewards per agent: {np.sum(history['rewards'], axis=0)}")
    print(f"Mean rewards per agent: {np.mean(history['rewards'], axis=0)}")
    print("="*80 + "\n")
    
    output_dir = Path("outputs") / "glider_ma_rollout"
    figure_paths = plot_state_history(history, output_dir, num_agents)

    print("\nSaved plots:")
    for name, path in figure_paths.items():
        print(f" - {name}: {path}")


if __name__ == "__main__":
    main()
