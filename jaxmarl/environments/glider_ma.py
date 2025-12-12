"""
Multi-Agent Glider Environment for JaxMARL

This environment extends the single-agent Glider environment to support multiple gliders
navigating in a shared thermal soaring environment. Agents must learn to exploit thermals
efficiently while avoiding collisions with other gliders.

Based on:
- Single-agent Glider environment from gymnax/Stoix
- JaxMARL multi-agent environment API
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct
from typing import Dict, Tuple, Any, Optional
from functools import partial
import chex

from jaxmarl.environments.multi_agent_env import MultiAgentEnv, State as BaseState
from jaxmarl.environments.spaces import Box

# Import wind utilities from Stoix if available, otherwise define locally
from mava.utils.wind import WindModel, thermal_centers, wind_at


DEG2RAD = jnp.pi / 180.0
RAD2DEG = 180.0 / jnp.pi


@struct.dataclass
class State(BaseState):
    """Multi-agent glider state"""
    
    # Per-agent states - shape [num_agents, ...]
    position: chex.Array  # (num_agents, 3) - (x, y, z) for each agent

    ground_speed: chex.Array  # (num_agents,) - ground speed for each agent
    air_speed: chex.Array  # (num_agents,) - airspeed for each agent
    vertical_speed: chex.Array  # (num_agents,) - vertical speed for each agent 

    attitude: chex.Array  # (num_agents, 2) - (glide_angle, side_angle) for each agent
    controls: chex.Array  # (num_agents, 3) - (bank_angle, attack_angle, sideslip_angle) for each agent
    angle_from_wind: chex.Array  # (num_agents,) - angle from wind for each agent

    distances_to_thermal: chex.Array  # (num_agents,) - distance to nearest thermal for each agent

    distances_to_other_agents: chex.Array  # (num_agents, num_agents - 1) -  distances from each agent to all other agents

    # History buffers for observations - per agent
    speeds_history: chex.Array  # (num_agents, history_seconds, 2) includes air_speed and vertical speed
    controls_history: chex.Array  # (num_agents, history_seconds, 3) includes bank, attack, sideslip
    angle_from_wind_history: chex.Array  # (num_agents, history_seconds) includes angle from wind
    attitude_history: chex.Array  # (num_agents, history_seconds, 2) includes glide_angle and side_angle
    distances_to_other_agents_history: chex.Array  # (num_agents, history_seconds, num_agents - 1) includes distances to other agents
    
    # wind_velocity_history: chex.Array  # (num_agents, 1) - wind velocity magnitude history
    relative_positions_local_history: chex.Array  # (num_agents, history_seconds, num_agents - 1, 3) - history of positions of other agents in each agent's local frame
    
    # Shared state
    done: chex.Array  # (num_agents,) - done flag for each agent
    step: int  # current step
    

    


@struct.dataclass
class EnvParams:
    """Environment parameters for multi-agent glider"""
    
    # Multi-agent specific
    num_agents: int = 3
    
    # Time parameters
    dt: float = 0.01
    max_steps_in_episode: int = 200
    history_seconds: int = struct.field(pytree_node=False, default=8)
    
    # Physical constants (same as single-agent)
    g: float = 9.81
    rho: float = 1.225
    mass: float = 5.0
    wingspan: float = 2.5
    aspect_ratio: float = 16.0
    
    # Aerodynamic coefficients
    e: float = 0.95
    a0: float = 0.1 * RAD2DEG
    alpha0: float = -2.5 * DEG2RAD
    V_H: float = 0.4
    V_V: float = 0.02
    C_d_0: float = 0.01
    C_d_L: float = 0.05
    C_L_min: float = 0.4
    C_D_F: float = 0.008
    C_D_T: float = 0.01
    C_D_E: float = 0.002
    
     # Operational bounds
    horizontal_bound: float = 5_000.0
    vertical_bounds: tuple[float, float] = (0.0, 1_000.0)
    ground_speed_bounds: tuple[float, float] = (0.0, 100.0)
    air_speed_bounds: tuple[float, float] = (0.0, 100.0)
    vertical_speed_bounds: tuple[float, float] = (-100.0, 30.0)
    glide_limits: tuple[float, float] = (-25.0 * DEG2RAD, 45.0 * DEG2RAD)
    side_limits: tuple[float, float] = (-jnp.pi, jnp.pi)
    bank_limits: tuple[float, float] = (-50.0 * DEG2RAD, 50.0 * DEG2RAD)
    attack_limits: tuple[float, float] = (-30.0 * DEG2RAD, 30.0 * DEG2RAD)
    sideslip_limits: tuple[float, float] = (-50.0 * DEG2RAD, 50.0 * DEG2RAD)
    angle_from_wind_limits: tuple[float, float] = (-180.0 * DEG2RAD, 180.0 * DEG2RAD)
    # wind_velocity_limits: tuple[float, float] = (0.0, 20.0)
    distance_to_other_agents_limits: tuple[float, float] = (0.0, 1_000.0)
    relative_position_limits: tuple[float, float] = (-1_000.0, 1_000.0)  # bounds for relative positions in local frame
    
    # Control increments per unit action
    action_deltas: tuple[float, float, float] = (
        15.0 * DEG2RAD,
        10.0 * DEG2RAD,
        3.0 * DEG2RAD,
    )
    
    # Initial conditions
    initial_altitude: float = 500.0
    initial_speed: float = 10.0
    initial_spawn_radius: float = 100.0  # spawn agents in a circle

    collision_distance: float = 1.0  # minimum distance between agents
    collision_penalty: float = -1000.0

    vertical_speed_weight: float = 1.0
    distance_to_other_weight: float = 1.0
    
    # Field of view for agent observation (in radians, total angle)
    # Controls both horizontal (XY plane) and vertical (cone angle) FOV
    field_of_view: float = 120.0 * DEG2RAD  # 120 degrees FOV
    field_of_view_rotation: float = 0.0 * DEG2RAD  # FOV rotation relative to forward direction (0 = forward, positive = right)
    
    # Wind model
    wind_model: WindModel = struct.field(default_factory=WindModel.default)

def calculate_inter_agent_distances(positions: jax.Array) -> jax.Array:
    """
    Calculate distances from each agent to all other agents (excluding self).
    
    Args:
        positions: (num_agents, 3) array of agent positions
        
    Returns:
        (num_agents, num_agents - 1) array where row i contains distances 
        from agent i to all other agents (excluding itself)
    """
    # positions: (num_agents, 3)
    num_agents = positions.shape[0]
    
    # Expand dims to create (num_agents, num_agents, 3) difference matrix
    diff = positions[:, None, :] - positions[None, :, :]
    # Calculate euclidean distance
    dist = jnp.linalg.norm(diff, axis=-1)  # (num_agents, num_agents)
    
    # Remove diagonal using boolean indexing with where
    # Create indices for all elements
    i_indices = jnp.arange(num_agents)[:, None]  # (num_agents, 1)
    j_indices = jnp.arange(num_agents)[None, :]  # (1, num_agents)
    
    # Create mask where diagonal elements are False
    mask = i_indices != j_indices  # (num_agents, num_agents)
    
    # Use where to select non-diagonal elements, then reshape
    # For each row, select num_agents-1 elements (all except diagonal)
    distances_to_others = jnp.where(
        mask,
        dist,
        jnp.inf  # Fill diagonal with inf temporarily
    )
    
    # Sort each row and take first num_agents-1 elements (excluding the inf on diagonal)
    distances_to_others = jnp.sort(distances_to_others, axis=1)[:, :num_agents-1]
    
    return distances_to_others


def detect_collisions(positions: jax.Array, collision_distance: float) -> Tuple[jax.Array, jax.Array]:
    """
    Detect collisions between agents based on minimum distance threshold.
    
    Args:
        positions: (num_agents, 3) array of agent positions
        collision_distance: minimum distance threshold for collision detection
        
    Returns:
        collision_mask: (num_agents,) boolean array indicating which agents are in collision
        min_distances: (num_agents,) array of minimum distance to nearest agent for each agent
    """
    num_agents = positions.shape[0]
    
    # Calculate pairwise distances
    diff = positions[:, None, :] - positions[None, :, :]
    dist = jnp.linalg.norm(diff, axis=-1)  # (num_agents, num_agents)
    
    # Mask out self-distances by setting diagonal to infinity
    mask = jnp.eye(num_agents, dtype=bool)
    dist_masked = jnp.where(mask, jnp.inf, dist)
    
    # Find minimum distance to any other agent for each agent
    min_distances = jnp.min(dist_masked, axis=1)
    
    # Check if any agent is closer than collision distance
    collision_mask = min_distances < collision_distance
    
    return collision_mask, min_distances

# Helper functions from single-agent environment
def _wing_surface_area(params: EnvParams) -> jax.Array:
    return (params.wingspan**2) / params.aspect_ratio


def _tail_moment_arm(params: EnvParams) -> jax.Array:
    return 0.28 * params.wingspan


def _mean_aerodynamic_chord(params: EnvParams) -> jax.Array:
    return 1.03 * params.wingspan / params.aspect_ratio


def _fuselage_area(params: EnvParams) -> jax.Array:
    span = params.wingspan
    return 0.01553571429 * span**2 + 0.01950357142 * span - 0.01030412685


def _horizontal_tail_surface(params: EnvParams) -> jax.Array:
    S = _wing_surface_area(params)
    lt = _tail_moment_arm(params)
    c_bar = _mean_aerodynamic_chord(params)
    return params.V_H * c_bar * S / lt


def _vertical_tail_surface(params: EnvParams) -> jax.Array:
    S = _wing_surface_area(params)
    lt = _tail_moment_arm(params)
    return params.V_V * params.wingspan * S / lt


def _lift_curve_slope(params: EnvParams) -> jax.Array:
    return params.a0 / (1 + params.a0 / (jnp.pi * params.e * params.aspect_ratio))


def _yaw_curve_slope(params: EnvParams) -> jax.Array:
    S = _wing_surface_area(params)
    S_V = jnp.maximum(_vertical_tail_surface(params), 1e-6)
    AR_V = 0.5 * params.aspect_ratio
    return (params.a0 / (1 + params.a0 / (jnp.pi * params.e * AR_V))) * (S_V / S)


def calculate_forces(
    params: EnvParams, speed: jax.Array, attack_angle: jax.Array, sideslip_angle: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return (drag, lift, side) force magnitudes."""
    S = _wing_surface_area(params)
    S_V = jnp.maximum(_vertical_tail_surface(params), 1e-6)
    S_F = _fuselage_area(params)
    S_T = _horizontal_tail_surface(params)

    C_L_alpha = _lift_curve_slope(params)
    C_C_beta = _yaw_curve_slope(params)

    attack_offset = attack_angle - params.alpha0
    C_L = C_L_alpha * attack_offset
    C_C = C_C_beta * sideslip_angle

    base_drag = (
        params.C_D_F * S_F / S
        + params.C_D_T * (S_T + S_V) / S
        + params.C_D_E
        + params.C_d_0
    )
    induced_drag = (C_L**2 + (C_C**2) * (S / S_V)) / (jnp.pi * params.e * params.aspect_ratio)
    C_D = base_drag + params.C_d_L * (C_L - params.C_L_min) ** 2 + induced_drag

    q = 0.5 * params.rho * speed**2

    drag = q * S * C_D
    lift = q * S * C_L
    side = q * S * C_C

    return drag, lift, side


def calculate_angle_from_wind(bank_angle: jax.Array, side_angle: jax.Array, params: EnvParams) -> jax.Array:
    """Calculate angle from wind based on bank angle, side angle and wind direction."""
    horizontal_wind = params.wind_model.horizontal_wind
    wind_angle = jnp.arctan2(horizontal_wind[1], horizontal_wind[0])
    
    bank_angle_deg = bank_angle * RAD2DEG
    side_angle_deg = side_angle * RAD2DEG
    wind_angle_deg = wind_angle * RAD2DEG
    
    sign_factor = 2.0 * (bank_angle_deg >= 0) - 1.0
    angle_diff = ((side_angle_deg - wind_angle_deg) % 360.0) - 180.0
    angle_from_wind_deg = sign_factor * angle_diff
    
    return angle_from_wind_deg * DEG2RAD


def calculate_wind_velocity(params: EnvParams) -> jax.Array:
    """Calculate wind velocity magnitude from horizontal wind components."""
    horizontal_wind = params.wind_model.horizontal_wind
    return jnp.linalg.norm(horizontal_wind)


def _rotation_matrix_x(angle: jax.Array) -> jax.Array:
    """Rotation matrix around X axis"""
    cos_a, sin_a = jnp.cos(angle), jnp.sin(angle)
    return jnp.array([[1, 0, 0],
                      [0, cos_a, -sin_a],
                      [0, sin_a, cos_a]])


def _rotation_matrix_y(angle: jax.Array) -> jax.Array:
    """Rotation matrix around Y axis"""
    cos_a, sin_a = jnp.cos(angle), jnp.sin(angle)
    return jnp.array([[cos_a, 0, sin_a],
                      [0, 1, 0],
                      [-sin_a, 0, cos_a]])


def _rotation_matrix_z(angle: jax.Array) -> jax.Array:
    """Rotation matrix around Z axis"""
    cos_a, sin_a = jnp.cos(angle), jnp.sin(angle)
    return jnp.array([[cos_a, -sin_a, 0],
                      [sin_a, cos_a, 0],
                      [0, 0, 1]])


def calculate_relative_positions_local_frame(
    agent_position: jax.Array,
    agent_attitude: jax.Array,
    agent_control: jax.Array,
    other_positions: jax.Array,
    field_of_view: float,
    field_of_view_rotation: float
) -> jax.Array:
    """
    Calculate positions of other agents in the local coordinate frame of the reference agent.
    Agents outside the field of view are assigned coordinates of 1000.0.
    
    The local frame is defined such that:
    - X-axis points in the direction of the agent's velocity vector (horizontal projection)
    - Y-axis points to the right (perpendicular to X in horizontal plane)
    - Z-axis points upward (vertical, same as inertial frame)
    
    Args:
        agent_position: (3,) position of the reference agent [x, y, z]
        agent_attitude: (2,) attitude of the reference agent [glide_angle, side_angle]
        agent_control: (3,) controls of the reference agent [bank, attack, sideslip]
        other_positions: (num_others, 3) positions of other agents
        field_of_view: (float) field of view angle in radians (total angle, e.g., 120 degrees = +/- 60 degrees)
                       Controls both horizontal (XY plane) and vertical (cone angle) FOV
        field_of_view_rotation: (float) rotation of FOV direction relative to forward (X-axis)
                                Positive values rotate FOV to the right, negative to the left
        
    Returns:
        (num_others, 3) positions of other agents in reference agent's local frame.
        For agents outside FOV, returns [1000.0, 1000.0, 1000.0]
    """
    glide_angle, side_angle = agent_attitude
    
    # Only rotate around Z-axis (yaw) to align X-axis with velocity direction in horizontal plane
    # side_angle determines the heading direction in the horizontal plane
    R_z_side = _rotation_matrix_z(side_angle)
    
    # Calculate relative positions in inertial frame
    relative_positions_inertial = other_positions - agent_position[None, :]
    
    # Transform to local frame (only horizontal rotation)
    # Apply rotation to each relative position vector
    relative_positions_local = jnp.dot(relative_positions_inertial, R_z_side.T)
    
    # Calculate FOV center direction by rotating forward direction (X-axis) by field_of_view_rotation
    # FOV center in XY plane
    fov_center_x = jnp.cos(field_of_view_rotation)
    fov_center_y = jnp.sin(field_of_view_rotation)
    
    # Check horizontal FOV: angle in XY plane from FOV center direction
    # Calculate angle between each agent's position and FOV center direction
    horizontal_angles = jnp.arccos(
        jnp.clip(
            (relative_positions_local[:, 0] * fov_center_x + relative_positions_local[:, 1] * fov_center_y) / 
            jnp.maximum(jnp.sqrt(relative_positions_local[:, 0]**2 + relative_positions_local[:, 1]**2), 1e-6),
            -1.0, 1.0
        )
    )
    half_fov = field_of_view / 2.0
    within_horizontal_fov = horizontal_angles <= half_fov
    
    # Check vertical FOV: cone angle from FOV center direction (3D angle)
    # FOV center direction in 3D (rotated in horizontal plane only)
    fov_center_dir = jnp.array([fov_center_x, fov_center_y, 0.0], dtype=jnp.float32)
    
    # Calculate 3D angle between each agent's position and FOV center direction
    position_norms = jnp.maximum(jnp.linalg.norm(relative_positions_local, axis=1), 1e-6)
    fov_center_norm = jnp.linalg.norm(fov_center_dir)
    
    # Dot product divided by norms gives cosine of angle
    dot_products = jnp.sum(relative_positions_local * fov_center_dir[None, :], axis=1)
    cone_angles = jnp.arccos(jnp.clip(dot_products / (position_norms * fov_center_norm), -1.0, 1.0))
    
    within_vertical_fov = cone_angles <= half_fov
    
    # Agent must be within BOTH horizontal and vertical FOV to be visible
    within_fov = within_horizontal_fov & within_vertical_fov
    
    # For agents outside FOV, set coordinates to 1000.0
    invisible_position = jnp.array([1000.0, 1000.0, 1000.0], dtype=jnp.float32)
    relative_positions_local = jnp.where(
        within_fov[:, None],
        relative_positions_local,
        invisible_position[None, :]
    )
    
    return relative_positions_local


def _step_per_sec_single_agent(
    position: jax.Array,
    ground_speed: jax.Array, 
    attitude: jax.Array,
    controls: jax.Array,
    time: int,
    params: EnvParams
) -> tuple[jax.Array, jax.Array, jax.Array]:
    bank, attack, sideslip = controls

    unit_x = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
    unit_y = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32)
    unit_z = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)

    R_x_bank = _rotation_matrix_x(bank)
    R_z_v_to_b = _rotation_matrix_z(-sideslip)
    R_y_v_to_b = _rotation_matrix_y(attack)
    R_v_to_b = R_y_v_to_b @ R_z_v_to_b

    dt = jnp.asarray(params.dt, dtype=jnp.float32)
    mass = jnp.asarray(params.mass, dtype=jnp.float32)
    g = jnp.asarray(params.g, dtype=jnp.float32)
    
    def _safe_cos(angle: jax.Array) -> jax.Array:
        cos_val = jnp.cos(angle)
        return jnp.where(
            jnp.abs(cos_val) < 1e-6,
            jnp.where(cos_val >= 0, 1e-6, -1e-6),
            cos_val,
        )

    def body_fun(step_idx: int, carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        position, ground_speed, attitude, current_time = carry

        ground_speed = jnp.maximum(ground_speed, 1e-6)
        glide_angle, side_angle = attitude

        R_y_glide = _rotation_matrix_y(-glide_angle)
        R_z_side = _rotation_matrix_z(side_angle)
        R_i_to_v_positive_glide = R_z_side @ R_y_glide @ R_x_bank

        v_vector_in_i = R_i_to_v_positive_glide @ (ground_speed * unit_x)

        w_vector_in_i = wind_at(params.wind_model, position, current_time)

        relative_velocity = v_vector_in_i - w_vector_in_i
        relative_velocity_norm = jnp.linalg.norm(relative_velocity)
        safe_relative_norm = jnp.maximum(relative_velocity_norm, 1e-6)
        relative_velocity_direction = relative_velocity / safe_relative_norm

        glide_angle_in_w = jnp.arcsin(jnp.clip(relative_velocity_direction[2], -1.0, 1.0))
        cos_glide_in_w = _safe_cos(glide_angle_in_w)
        side_angle_in_w = jnp.sign(relative_velocity_direction[1] / cos_glide_in_w) * jnp.arccos(
            jnp.clip(relative_velocity_direction[0] / cos_glide_in_w, -1.0, 1.0)
        )

        R_y_att = _rotation_matrix_y(glide_angle)
        R_z_att = _rotation_matrix_z(side_angle)
        R_i_to_v = R_z_att @ R_y_att @ R_x_bank

        R_z_b_to_m = _rotation_matrix_z(-side_angle_in_w)
        R_y_b_to_m = _rotation_matrix_y(-glide_angle_in_w)
        R_b_to_m = R_y_b_to_m @ R_z_b_to_m

        third_col_of_m = R_b_to_m @ (R_i_to_v @ (R_v_to_b @ unit_z))
        second_col_of_m = R_b_to_m @ (R_i_to_v @ (R_v_to_b @ unit_y))
        first_col_of_m = R_b_to_m @ (R_i_to_v @ (R_v_to_b @ unit_x))

        attack_angle_in_w = jnp.arcsin(jnp.clip(third_col_of_m[0], -1.0, 1.0))
        cos_attack_in_w = _safe_cos(attack_angle_in_w)
        bank_angle_in_w = jnp.sign(-third_col_of_m[1] / cos_attack_in_w) * jnp.arccos(
            jnp.clip(third_col_of_m[2] / cos_attack_in_w, -1.0, 1.0)
        )
        sideslip_angle_in_w = jnp.sign(second_col_of_m[0] / cos_attack_in_w) * jnp.arccos(
            jnp.clip(first_col_of_m[0] / cos_attack_in_w, -1.0, 1.0)
        )

        d_w, l_w, c_w = calculate_forces(
            params, safe_relative_norm, attack_angle_in_w, sideslip_angle_in_w
        )

        R_v_to_i = R_i_to_v.T
        R_x_w = _rotation_matrix_x(bank_angle_in_w)
        R_y_w = _rotation_matrix_y(glide_angle_in_w)
        R_z_w = _rotation_matrix_z(side_angle_in_w)
        R_i_to_w = R_z_w @ R_y_w @ R_x_w

        forces_temp = R_i_to_w @ jnp.array([-d_w, -c_w, -l_w], dtype=jnp.float32)
        forces_in_v = R_v_to_i @ forces_temp
        d_v, c_v, l_v = -forces_in_v

        sin_glide = jnp.sin(glide_angle)
        cos_glide = jnp.cos(glide_angle)
        sin_side = jnp.sin(side_angle)
        cos_side = jnp.cos(side_angle)

        dz = ground_speed * sin_glide
        dx = ground_speed * cos_side * cos_glide
        dy = ground_speed * sin_side * cos_glide

        dv = -d_v / mass - g * sin_glide
        inv_speed = 1.0 / jnp.maximum(ground_speed, 1e-6)
        d_glide = (c_v * jnp.sin(bank) + l_v * jnp.cos(bank)) * inv_speed / mass - g * cos_glide * inv_speed
        d_side = (l_v * jnp.sin(bank) - c_v * jnp.cos(bank)) * inv_speed / mass

        position = position + dt * jnp.array([dx, dy, dz], dtype=jnp.float32)
        ground_speed = jnp.clip(ground_speed + dt * dv, 1e-6, 1000)
        glide_angle = glide_angle + dt * d_glide
        side_angle = side_angle + dt * d_side
        attitude = jnp.array([glide_angle, side_angle], dtype=jnp.float32)
        current_time = current_time + dt

        return position, ground_speed, attitude, current_time

    init_time = jnp.asarray(time, dtype=jnp.float32)
    init_carry = (position, ground_speed, attitude, init_time)
    final_position, final_speed, final_attitude, _ = jax.lax.fori_loop(0, 100, body_fun, init_carry)

    return final_position, final_speed, final_attitude


class GliderMA(MultiAgentEnv):
    """Multi-agent glider environment for thermal soaring"""
    
    def __init__(self, num_agents: int = 3, **kwargs):
        super().__init__(num_agents=num_agents)
        
        self.params = EnvParams(num_agents=num_agents, **kwargs)
        self.agents = [f"agent_{i}" for i in range(num_agents)]
        
        # Set up observation and action spaces
        # Observation includes:
        # - speed_history: history_seconds * 2
        # - controls_history: history_seconds * 3
        # - attitude_history: history_seconds * 2
        # - relative_positions_local_history: history_seconds * (num_agents - 1)*3

        obs_size = (self.params.history_seconds * 2 + 
                    self.params.history_seconds * 3 + 
                    self.params.history_seconds * 2 +
                    self.params.history_seconds * (num_agents - 1)*3
                    )
        
        for agent in self.agents:
            self.observation_spaces[agent] = Box(
                low=-1.0,
                high=1.0,
                shape=(obs_size,),
                dtype=jnp.float32
            )
            self.action_spaces[agent] = Box(
                low=-1.0,
                high=1.0,
                shape=(3,),
                dtype=jnp.float32
            )
    
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        """Reset environment for all agents"""
        keys = jax.random.split(key, self.params.num_agents + 1)
        key_angles = keys[0]
        keys_pos = keys[1:]

        # Initialize positions in a circle
        if self.params.num_agents > 1:
            min_radius = self.params.initial_spawn_radius / (2 * jnp.sin(jnp.pi / self.params.num_agents))
            spawn_radius = jnp.maximum(self.params.initial_spawn_radius, min_radius)
        else:
            spawn_radius = self.params.initial_spawn_radius
        angles = jnp.linspace(0, 2 * jnp.pi, self.params.num_agents, endpoint=False)
        x = spawn_radius * jnp.cos(angles)
        y = spawn_radius * jnp.sin(angles)
        z = jnp.full((self.params.num_agents,), self.params.initial_altitude)
        positions = jnp.stack([x, y, z], axis=1) 
        
        # Initialize attitudes with random side angles
        side_angles = jax.random.uniform( key_angles, shape=(self.params.num_agents,), minval=-180.0 * DEG2RAD, maxval=180.0 * DEG2RAD)
        glide_angle = jax.random.uniform(key_angles, shape=(self.params.num_agents,), minval=-4.0 * DEG2RAD, maxval=4.0 * DEG2RAD)
        attitudes = jnp.stack([ glide_angle,side_angles], axis=1)
        
        # Initialize controls to zero
        controls = jnp.zeros((self.params.num_agents, 3), dtype=jnp.float32)
        
        # Calculate initial air speeds
        def calc_initial_air_speed(agent_idx):
            glide_angle, side_angle = attitudes[agent_idx]
            bank_angle = controls[agent_idx, 0]
            
            R_y_glide = _rotation_matrix_y(-glide_angle)
            R_z_side = _rotation_matrix_z(side_angle)
            R_x_bank = _rotation_matrix_x(bank_angle)
            R_i_to_v = R_z_side @ R_y_glide @ R_x_bank
            
            unit_x = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
            v_vector = R_i_to_v @ (ground_speeds[agent_idx] * unit_x)
            
            w_vector = wind_at(self.params.wind_model, positions[agent_idx], self.params.history_seconds)
            
            relative_velocity = v_vector - w_vector
            return jnp.linalg.norm(relative_velocity)
        # Initialize speeds
        ground_speeds = jnp.full((self.params.num_agents,), self.params.initial_speed, dtype=jnp.float32)
        air_speeds = jax.vmap(calc_initial_air_speed)(jnp.arange(self.params.num_agents))
        initial_vertical_speeds = ground_speeds * jnp.sin(attitudes[:, 0])
        vertical_speeds = jnp.full((self.params.num_agents,), initial_vertical_speeds, dtype=jnp.float32)

        # Initialize distances
        distances_to_thermal = jnp.full((self.params.num_agents,), 0.0, dtype=jnp.float32)
        distances_to_other_agents = calculate_inter_agent_distances(positions)
        
        # Initialize history buffers
        speed_history = jnp.zeros((self.params.num_agents, self.params.history_seconds, 2), dtype=jnp.float32)
        speed_history = speed_history.at[:, :, 0].set(air_speeds[:, None])
        speed_history = speed_history.at[:, :, 1].set(initial_vertical_speeds[:, None])
        
        controls_history = jnp.zeros((self.params.num_agents, self.params.history_seconds, 3), dtype=jnp.float32)

        initial_angles_from_wind = jax.vmap( lambda bank, side: calculate_angle_from_wind(bank, side, self.params))(controls[:, 0], attitudes[:, 1])
        angle_from_wind_history = jnp.tile( initial_angles_from_wind[:, None], (1, self.params.history_seconds))
        
        # Initialize attitude history
        attitude_history = jnp.tile(attitudes[:, None, :], (1, self.params.history_seconds, 1))
        
        # Calculate initial wind velocity
        initial_wind_velocity = calculate_wind_velocity(self.params)
        wind_velocity_history = jnp.full(
            (self.params.num_agents, 1),
            initial_wind_velocity,
            dtype=jnp.float32
        )
        
        # Calculate relative positions in local frame for each agent
        def calc_relative_positions_for_agent(agent_idx):
            # Calculate relative positions for ALL agents, then we'll filter later
            agent_pos = positions[agent_idx]
            agent_att = attitudes[agent_idx]
            agent_ctrl = controls[agent_idx]
            
            # Transform all positions (including self) to local frame
            all_relative = calculate_relative_positions_local_frame(
                agent_pos,
                agent_att,
                agent_ctrl,
                positions,
                self.params.field_of_view,
                self.params.field_of_view_rotation
            )  # Shape: (num_agents, 3)
            
            # Remove self by selecting all indices except agent_idx
            # Use jnp.where to avoid boolean indexing
            indices = jnp.arange(self.params.num_agents)
            # Create array that excludes agent_idx: [0,1,2,...,agent_idx-1, agent_idx+1,...,n-1]
            other_indices = jnp.where(
                indices < agent_idx,
                indices,
                indices + 1
            )
            # Take only first num_agents-1 elements (since we shifted indices after agent_idx)
            other_indices = other_indices[:self.params.num_agents - 1]
            
            return all_relative[other_indices]
        
        initial_relative_positions_local = jax.vmap(calc_relative_positions_for_agent)(
            jnp.arange(self.params.num_agents)
        )  # Shape: (num_agents, num_agents - 1, 3)
        
        # Initialize relative_positions_local_history with the same positions for all time steps
        relative_positions_local_history = jnp.tile(
            initial_relative_positions_local[:, None, :, :],
            (1, self.params.history_seconds, 1, 1)
        )  # Shape: (num_agents, history_seconds, num_agents - 1, 3)
        
        # Initialize distances_to_other_agents_history with the same distances for all time steps
        distances_to_other_agents_history = jnp.tile(
            distances_to_other_agents[:, None, :],
            (1, self.params.history_seconds, 1)
        )  # Shape: (num_agents, history_seconds, num_agents - 1)
        
        state = State(
            position=positions,

            ground_speed=ground_speeds,
            air_speed=air_speeds,
            vertical_speed=vertical_speeds,

            attitude=attitudes,
            controls=controls,
            angle_from_wind=initial_angles_from_wind,

            distances_to_thermal=distances_to_thermal,
            distances_to_other_agents=distances_to_other_agents,

            speeds_history=speed_history,
            controls_history=controls_history,
            angle_from_wind_history=angle_from_wind_history,
            attitude_history=attitude_history,
            distances_to_other_agents_history=distances_to_other_agents_history,
            relative_positions_local_history=relative_positions_local_history,

            done=jnp.zeros(self.params.num_agents, dtype=bool),
            step=self.params.history_seconds,
        )
        return self.get_obs(state), state
    
    def step_env(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Step environment for all agents"""
        
        # Convert actions dict to array
        actions_array = jnp.stack([actions[agent] for agent in self.agents])
        actions_array = jnp.clip(actions_array, -1.0, 1.0)
        # Apply action deltas to controls
        delta = actions_array * jnp.array(self.params.action_deltas, dtype=jnp.float32)
        
        new_controls = state.controls + delta
        new_controls = jnp.clip(
            new_controls,
            jnp.array([self.params.bank_limits[0], self.params.attack_limits[0], self.params.sideslip_limits[0]]),
            jnp.array([self.params.bank_limits[1], self.params.attack_limits[1], self.params.sideslip_limits[1]])
        )
        
        # Step physics for each agent
        def step_agent(agent_idx):
            return _step_per_sec_single_agent(
                state.position[agent_idx],
                state.ground_speed[agent_idx],
                state.attitude[agent_idx],
                new_controls[agent_idx],
                state.step,
                self.params
            )
        
        new_positions, new_ground_speeds, new_attitudes = jax.vmap(step_agent)(jnp.arange(self.params.num_agents))
        
        # Calculate air speed for each agent (relative velocity to wind)
        def calc_air_speed(agent_idx):
            glide_angle, side_angle = new_attitudes[agent_idx]
            bank_angle = new_controls[agent_idx, 0]
            
            # Rotation matrices to get velocity vector in inertial frame
            R_y_glide = _rotation_matrix_y(-glide_angle)
            R_z_side = _rotation_matrix_z(side_angle)
            R_x_bank = _rotation_matrix_x(bank_angle)
            R_i_to_v = R_z_side @ R_y_glide @ R_x_bank
            
            # Glider velocity vector in inertial frame
            unit_x = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
            v_vector = R_i_to_v @ (new_ground_speeds[agent_idx] * unit_x)
            
            # Wind velocity vector at current position
            w_vector = wind_at(self.params.wind_model, new_positions[agent_idx], state.step )
            
            # Air speed is the magnitude of relative velocity
            relative_velocity = v_vector - w_vector
            return jnp.linalg.norm(relative_velocity)
        
        air_speeds = jax.vmap(calc_air_speed)(jnp.arange(self.params.num_agents))

        # Calculate vertical speeds 
        vertical_speeds = ((new_positions[:, 2]-state.position[:, 2]) + (air_speeds**2 - state.air_speed**2)/(2*self.params.g) ) / 1

        # Get thermal center position and distances from agents to thermal center
        thermal_center_full = thermal_centers(self.params.wind_model, new_positions[0, 2], jnp.asarray(state.step, dtype=jnp.float32))
        thermal_center_xy = thermal_center_full.reshape(-1, 3)[0, :2]  # Get first thermal, xy only
        distances_to_thermal = jnp.linalg.norm( new_positions[:, :2] - thermal_center_xy[None, :], axis=1)

        # Calculate inter-agent distances and update history
        distances_to_other_agents = calculate_inter_agent_distances(new_positions)

        # Update history buffers
        new_speed_entries = jnp.stack([air_speeds, vertical_speeds], axis=1)
        new_speed_history = jnp.roll(state.speeds_history, shift=-1, axis=1)
        new_speed_history = new_speed_history.at[:, -1, :].set(new_speed_entries)
        
        new_controls_history = jnp.roll(state.controls_history, shift=-1, axis=1)
        new_controls_history = new_controls_history.at[:, -1, :].set(new_controls)
        
        current_angles_from_wind = jax.vmap( lambda bank, side: calculate_angle_from_wind(bank, side, self.params))(new_controls[:, 0], new_attitudes[:, 1])
        new_angle_from_wind_history = jnp.roll(state.angle_from_wind_history, shift=-1, axis=1)
        new_angle_from_wind_history = new_angle_from_wind_history.at[:, -1].set(current_angles_from_wind)
        
        new_attitude_history = jnp.roll(state.attitude_history, shift=-1, axis=1)
        new_attitude_history = new_attitude_history.at[:, -1, :].set(new_attitudes)
        
        new_distances_to_other_agents_history = jnp.roll(state.distances_to_other_agents_history, shift=-1, axis=1)
        new_distances_to_other_agents_history = new_distances_to_other_agents_history.at[:, -1, :].set(distances_to_other_agents)

        step_number = state.step + 1

        # Check boundary conditions for each agent
        out_of_bounds_xy = (
            (jnp.abs(new_positions[:, 0]) > self.params.horizontal_bound) |
            (jnp.abs(new_positions[:, 1]) > self.params.horizontal_bound)
        )
        out_of_bounds_z = (
            (new_positions[:, 2] < self.params.vertical_bounds[0]) |
            (new_positions[:, 2] > self.params.vertical_bounds[1])
        )

        low_speed = new_ground_speeds <= jnp.linalg.norm(self.params.wind_model.horizontal_wind)

        collision_mask, min_distances = detect_collisions(new_positions, self.params.collision_distance)

        # Check if any agent violates conditions - if so, truncate for all agents
        
        
        # Calculate rewards
        
        # Proximity penalty: exponential penalty for getting too close to other agents
        # -1 at 20m, -100 at collision_distance (5m)
        proximity_threshold = 20.0
        
        
        # Exponential interpolation between -1 and -100
        # At d=20: penalty=-1, at d=1: penalty=-100
        # Formula: penalty = -1 * exp(k * (20 - d)) where k = ln(100) / 19
        k = 0.25 * jnp.log(400.0) / (proximity_threshold - self.params.collision_distance)
        proximity_penalty = jnp.where(
            min_distances < proximity_threshold,
            -jnp.exp(k * (proximity_threshold - min_distances)),
            0.0
        )

        # Thermal distance penalty: linearly decreases with distance to thermal
        # At distance = 0m: penalty = 0.0
        # At distance = 50m: penalty = -1
        # Linear formula: penalty = -0.007 * distance
        thermal_distance_penalty = -0.02 * distances_to_thermal

        reward_action = 2.0*vertical_speeds + 1.0*proximity_penalty + 1.0*thermal_distance_penalty
        

        max_steps_f = jnp.asarray(self.params.max_steps_in_episode, dtype=jnp.float32)
        step_f = jnp.asarray(state.step, dtype=jnp.float32)
        low_speed_reward = -(max_steps_f - step_f)
        

        # rewards_array = \
        #     jnp.where( out_of_bounds_xy | out_of_bounds_z, -1000.0,
        #         jnp.where(low_speed, low_speed_reward, \
        #                   reward_action)
        # )


        rewards_array = \
        jnp.where( out_of_bounds_xy | out_of_bounds_z, -1000.0,
            jnp.where(low_speed, low_speed_reward, 
                jnp.where(collision_mask, -1000, \
                          reward_action))
        )

        any_out_of_bounds = jnp.any(out_of_bounds_xy | out_of_bounds_z | low_speed | collision_mask)

        truncated = jnp.full(self.params.num_agents, any_out_of_bounds, dtype=bool)
        terminated = step_number >= self.params.max_steps_in_episode
        done_agents = truncated | terminated

        # Calculate relative positions in local frame for each agent
        def calc_relative_positions_for_agent(agent_idx):
            # Calculate relative positions for ALL agents, then we'll filter later
            agent_pos = new_positions[agent_idx]
            agent_att = new_attitudes[agent_idx]
            agent_ctrl = new_controls[agent_idx]
            
            # Transform all positions (including self) to local frame
            all_relative = calculate_relative_positions_local_frame(
                agent_pos,
                agent_att,
                agent_ctrl,
                new_positions,
                self.params.field_of_view,
                self.params.field_of_view_rotation
            )  # Shape: (num_agents, 3)
            
            # Remove self by selecting all indices except agent_idx
            # Use jnp.where to avoid boolean indexing
            indices = jnp.arange(self.params.num_agents)
            # Create array that excludes agent_idx: [0,1,2,...,agent_idx-1, agent_idx+1,...,n-1]
            other_indices = jnp.where(
                indices < agent_idx,
                indices,
                indices + 1
            )
            # Take only first num_agents-1 elements (since we shifted indices after agent_idx)
            other_indices = other_indices[:self.params.num_agents - 1]
            
            return all_relative[other_indices]
        
        new_relative_positions_local = jax.vmap(calc_relative_positions_for_agent)(
            jnp.arange(self.params.num_agents)
        )  # Shape: (num_agents, num_agents - 1, 3)
        # Roll the history and add new relative positions
        new_relative_positions_local_history = jnp.roll(state.relative_positions_local_history, shift=-1, axis=1)
        new_relative_positions_local_history = new_relative_positions_local_history.at[:, -1, :, :].set(new_relative_positions_local)
        
        # Create new state
        new_state = State(
            position=new_positions,

            ground_speed=new_ground_speeds,
            air_speed=air_speeds,
            vertical_speed=vertical_speeds,

            attitude=new_attitudes,
            controls=new_controls,
            angle_from_wind=current_angles_from_wind,

            distances_to_thermal=distances_to_thermal,
            distances_to_other_agents=distances_to_other_agents,

            speeds_history=new_speed_history,
            controls_history=new_controls_history,
            angle_from_wind_history=new_angle_from_wind_history,
            attitude_history=new_attitude_history,
            distances_to_other_agents_history=new_distances_to_other_agents_history,
            relative_positions_local_history=new_relative_positions_local_history,

            done=done_agents,
            step=step_number,
        )
        
        # Get observations
        obs = self.get_obs(new_state)
        
        # Convert to dicts (keep as JAX arrays for JIT compatibility)
        rewards = {agent: rewards_array[i] for i, agent in enumerate(self.agents)}
        dones = {agent: done_agents[i] for i, agent in enumerate(self.agents)}
        dones["__all__"] = jnp.all(done_agents)
        
        # Info dict
        info = {
            "step": step_number,
            "out_of_bounds": out_of_bounds_xy | out_of_bounds_z,
            "low_speed": low_speed,
            "vertical_speeds": vertical_speeds,
            "positions": new_positions,
            "distances_to_thermal": distances_to_thermal,
        }
        
        return obs, new_state, rewards, dones, info
    
    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Get observations for all agents"""
        
        def normalize(value: jax.Array, bounds: Tuple[float, float]) -> jax.Array:
            """Normalize value from bounds to [-1, 1]"""
            low, high = bounds
            # Scale from [low, high] to [-1, 1]
            return 2.0 * (value - low) / (high - low) - 1.0
            # return value
        
        def get_agent_obs(agent_idx: int) -> chex.Array:
            # Own state history - normalize each component
            speed_hist = state.speeds_history[agent_idx]  # (history_seconds, 2)
            
            # Normalize speed (column 0)
            normalized_speed = normalize(speed_hist[:, 0], self.params.air_speed_bounds)
            
            # Normalize vertical speed (column 1)
            normalized_vspeed = normalize(speed_hist[:, 1], self.params.vertical_speed_bounds)
            
            # Stack and flatten
            normalized_speed_hist = jnp.stack([normalized_speed, normalized_vspeed], axis=1).flatten()
            
            # Normalize controls history
            controls_hist = state.controls_history[agent_idx]  # (history_seconds, 3)
            normalized_bank = normalize(controls_hist[:, 0], self.params.bank_limits)
            normalized_attack = normalize(controls_hist[:, 1], self.params.attack_limits)
            normalized_sideslip = normalize(controls_hist[:, 2], self.params.sideslip_limits)
            normalized_controls_hist = jnp.stack([normalized_bank, normalized_attack, normalized_sideslip], axis=1).flatten()
            
            # Normalize angle from wind history
            angle_from_wind_hist = state.angle_from_wind_history[agent_idx]  # (history_seconds,)
            normalized_angle_from_wind_hist = normalize(angle_from_wind_hist, self.params.angle_from_wind_limits).flatten()
            
            # Normalize attitude history
            attitude_hist = state.attitude_history[agent_idx]  # (history_seconds, 2)
            normalized_glide = normalize(attitude_hist[:, 0], self.params.glide_limits)
            normalized_side = normalize(attitude_hist[:, 1], self.params.side_limits)
            normalized_attitude_hist = jnp.stack([normalized_glide, normalized_side], axis=1).flatten()
    
            # Normalize distances to other agents history
            distances_to_others_hist = state.distances_to_other_agents_history[agent_idx]  # (history_seconds, num_agents - 1)
            normalized_distances_to_others_history = normalize(distances_to_others_hist, self.params.distance_to_other_agents_limits).flatten()
            
            # Get relative positions history of other agents in local frame
            relative_pos_hist = state.relative_positions_local_history[agent_idx]  # (history_seconds, num_agents - 1, 3)
            # Normalize each coordinate (x, y, z) separately
            normalized_relative_pos_hist = normalize(relative_pos_hist, self.params.relative_position_limits).flatten()
            
            obs = jnp.concatenate([
                normalized_speed_hist,
                normalized_controls_hist,
                normalized_attitude_hist,
                normalized_relative_pos_hist,
            ])
            
            return obs
        
        obs_array = jax.vmap(get_agent_obs)(jnp.arange(self.params.num_agents))
        obs_dict = {agent: obs_array[i] for i, agent in enumerate(self.agents)}
        
        return obs_dict
    
    @property
    def name(self) -> str:
        return "GliderMA"
    
    @property
    def agent_classes(self) -> dict:
        return {"agents": self.agents}
