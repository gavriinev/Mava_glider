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
    speed: chex.Array  # (num_agents,) - forward speed for each agent
    attitude: chex.Array  # (num_agents, 2) - (glide_angle, side_angle) for each agent
    controls: chex.Array  # (num_agents, 3) - (bank_angle, attack_angle, sideslip_angle) for each agent
    
    # History buffers for observations - per agent
    speed_history: chex.Array  # (num_agents, history_seconds, 2)
    controls_history: chex.Array  # (num_agents, history_seconds, 3)
    angle_from_wind_history: chex.Array  # (num_agents, history_seconds)
    wind_velocity_history: chex.Array  # (num_agents, 1)
    distance_between_agents_history: chex.Array  # (num_agents, history_seconds, num_agents - 1)
    
    # Shared state
    done: chex.Array  # (num_agents,) - done flag for each agent
    step: int  # current step
    

@struct.dataclass
class EnvParams:
    """Environment parameters for multi-agent glider"""
    
    # Multi-agent specific
    num_agents: int = 3
    collision_distance: float = 50.0  # minimum distance between agents
    collision_penalty: float = -100.0
    
    # Time parameters
    dt: float = 0.01
    max_steps_in_episode: int = 200
    history_seconds: int = struct.field(pytree_node=False, default=1)
    
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
    speed_bounds: tuple[float, float] = (3.0, 30.0)
    glide_limits: tuple[float, float] = (-25.0 * DEG2RAD, 45.0 * DEG2RAD)
    side_limits: tuple[float, float] = (-jnp.pi, jnp.pi)
    bank_limits: tuple[float, float] = (-50.0 * DEG2RAD, 50.0 * DEG2RAD)
    attack_limits: tuple[float, float] = (-30.0 * DEG2RAD, 30.0 * DEG2RAD)
    sideslip_limits: tuple[float, float] = (-50.0 * DEG2RAD, 50.0 * DEG2RAD)
    
    # Control increments per unit action
    action_deltas: tuple[float, float, float] = (
        15.0 * DEG2RAD,
        10.0 * DEG2RAD,
        3.0 * DEG2RAD,
    )
    
    # Initial conditions
    initial_altitude: float = 500.0
    initial_speed: float = 10.0
    initial_spawn_radius: float = 10.0  # spawn agents in a circle
    
    # Reward shaping
    vertical_speed_weight: float = 1.0
    speed_penalty_weight: float = 0.01
    
    # Wind model
    wind_model: WindModel = struct.field(default_factory=WindModel.default)


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


def _step_per_sec_single_agent(
    position: jax.Array,
    speed: jax.Array, 
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
        position, speed, attitude, current_time = carry

        speed = jnp.maximum(speed, 1e-6)
        glide_angle, side_angle = attitude

        R_y_glide = _rotation_matrix_y(-glide_angle)
        R_z_side = _rotation_matrix_z(side_angle)
        R_i_to_v_positive_glide = R_z_side @ R_y_glide @ R_x_bank

        v_vector_in_i = R_i_to_v_positive_glide @ (speed * unit_x)

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

        dz = speed * sin_glide
        dx = speed * cos_side * cos_glide
        dy = speed * sin_side * cos_glide

        dv = -d_v / mass - g * sin_glide
        inv_speed = 1.0 / jnp.maximum(speed, 1e-6)
        d_glide = (c_v * jnp.sin(bank) + l_v * jnp.cos(bank)) * inv_speed / mass - g * cos_glide * inv_speed
        d_side = (l_v * jnp.sin(bank) - c_v * jnp.cos(bank)) * inv_speed / mass

        position = position + dt * jnp.array([dx, dy, dz], dtype=jnp.float32)
        speed = jnp.clip(speed + dt * dv, 1e-6, 1000)
        glide_angle = glide_angle + dt * d_glide
        side_angle = side_angle + dt * d_side
        attitude = jnp.array([glide_angle, side_angle], dtype=jnp.float32)
        current_time = current_time + dt

        return position, speed, attitude, current_time

    init_time = jnp.asarray(time, dtype=jnp.float32)
    init_carry = (position, speed, attitude, init_time)
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
        # - angle_from_wind_history: history_seconds * 1
        # - wind_velocity: 1
        # - distance_between_agents_history: history_seconds * (num_agents - 1)
        obs_size = (self.params.history_seconds * 2 + 
                    self.params.history_seconds * 3 + 
                    self.params.history_seconds * 1 + 
                    1 
                    # + self.params.history_seconds * (num_agents - 1)
                    )
        
        for agent in self.agents:
            self.observation_spaces[agent] = Box(
                low=-1000.0,
                high=1000.0,
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
        angles = jax.random.uniform(
            key_angles, 
            shape=(self.params.num_agents,),
            minval=0.0,
            maxval=2 * jnp.pi
        )
        
        radius = self.params.initial_spawn_radius
        positions = jnp.stack([
            radius * jnp.cos(angles),
            radius * jnp.sin(angles),
            jnp.full((self.params.num_agents,), self.params.initial_altitude, dtype=jnp.float32)
        ], axis=1)
        
        # Initialize attitudes with random side angles
        side_angles = jax.random.uniform(
            key_angles,
            shape=(self.params.num_agents,),
            minval=-100.0 * DEG2RAD,
            maxval=100.0 * DEG2RAD
        )
        attitudes = jnp.stack([
            jnp.full((self.params.num_agents,), 5.0 * DEG2RAD),
            side_angles
        ], axis=1)
        
        # Initialize controls to zero
        controls = jnp.zeros((self.params.num_agents, 3), dtype=jnp.float32)
        
        # Initialize speeds
        speeds = jnp.full((self.params.num_agents,), self.params.initial_speed, dtype=jnp.float32)
        
        # Initialize history buffers
        speed_history = jnp.zeros((self.params.num_agents, self.params.history_seconds, 2), dtype=jnp.float32)
        initial_vertical_speeds = speeds * jnp.sin(attitudes[:, 0])
        speed_history = speed_history.at[:, :, 0].set(speeds[:, None])
        speed_history = speed_history.at[:, :, 1].set(initial_vertical_speeds[:, None])
        
        controls_history = jnp.zeros((self.params.num_agents, self.params.history_seconds, 3), dtype=jnp.float32)
        
        # Calculate initial angles from wind for all agents
        initial_angles_from_wind = jax.vmap(
            lambda bank, side: calculate_angle_from_wind(bank, side, self.params)
        )(controls[:, 0], attitudes[:, 1])
        angle_from_wind_history = jnp.tile(
            initial_angles_from_wind[:, None],
            (1, self.params.history_seconds)
        )
        
        # Calculate initial wind velocity
        initial_wind_velocity = calculate_wind_velocity(self.params)
        wind_velocity_history = jnp.full(
            (self.params.num_agents, 1),
            initial_wind_velocity,
            dtype=jnp.float32
        )
        
        # Calculate initial distances between agents
        def compute_distances_for_agent(agent_idx):
            # Compute distances from this agent to all other agents
            distances = jnp.linalg.norm(
                positions - positions[agent_idx],
                axis=1
            )
            # Remove self-distance by rolling and slicing
            rolled_distances = jnp.roll(distances, -agent_idx)
            other_distances = rolled_distances[1:]  # Shape: (num_agents - 1,)
            return other_distances
        
        initial_distances = jax.vmap(compute_distances_for_agent)(jnp.arange(self.params.num_agents))
        # Shape: (num_agents, num_agents - 1)
        # Tile across history dimension
        distance_between_agents_history = jnp.tile(
            initial_distances[:, None, :],
            (1, self.params.history_seconds, 1)
        )
        
        state = State(
            position=positions,
            speed=speeds,
            attitude=attitudes,
            controls=controls,
            speed_history=speed_history,
            controls_history=controls_history,
            angle_from_wind_history=angle_from_wind_history,
            wind_velocity_history=wind_velocity_history,
            distance_between_agents_history=distance_between_agents_history,
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
                state.speed[agent_idx],
                state.attitude[agent_idx],
                new_controls[agent_idx],
                state.step,
                self.params
            )
        
        new_positions, new_speeds, new_attitudes = jax.vmap(step_agent)(jnp.arange(self.params.num_agents))
        
        # Check for collisions between agents
        def check_collisions():
            distances = jnp.linalg.norm(
                new_positions[:, None, :] - new_positions[None, :, :],
                axis=2
            )
            # Set diagonal to large value to ignore self-distances
            distances = jnp.where(
                jnp.eye(self.params.num_agents, dtype=bool),
                jnp.inf,
                distances
            )
            min_distances = jnp.min(distances, axis=1)
            return min_distances < self.params.collision_distance
        
        collisions = check_collisions()
        
        # Check boundary conditions for each agent
        out_of_bounds_xy = (
            (jnp.abs(new_positions[:, 0]) > self.params.horizontal_bound) |
            (jnp.abs(new_positions[:, 1]) > self.params.horizontal_bound)
        )
        out_of_bounds_z = (
            (new_positions[:, 2] < self.params.vertical_bounds[0]) |
            (new_positions[:, 2] > self.params.vertical_bounds[1])
        )
        low_speed = new_speeds <= jnp.linalg.norm(self.params.wind_model.horizontal_wind)
        
        # Check if any agent violates conditions - if so, truncate for all agents
        any_out_of_bounds = jnp.any(out_of_bounds_xy | out_of_bounds_z | low_speed)
        truncated = jnp.full(self.params.num_agents, any_out_of_bounds, dtype=bool)
        
        step_number = state.step + 1
        terminated = step_number >= self.params.max_steps_in_episode
        done_agents = truncated | terminated
        
        # Calculate rewards
        vertical_speeds = new_speeds * jnp.sin(new_attitudes[:, 0])
        # Convert step_number to float using jnp.asarray instead of float() for JAX compatibility
        thermal_center_full = thermal_centers(self.params.wind_model, new_positions[0, 2], jnp.asarray(step_number, dtype=jnp.float32))
        # Extract xy coordinates of the first thermal: shape (..., num_thermals, 3) -> take first thermal's xy
        # thermal_center_full has shape (1, 1, 3) or similar, we need (2,) for xy coordinates
        thermal_center_xy = thermal_center_full.reshape(-1, 3)[0, :2]  # Get first thermal, xy only
        
        distances_to_thermal = jnp.linalg.norm(
            new_positions[:, :2] - thermal_center_xy[None, :],
            axis=1
        )
        
        reward_action = vertical_speeds + 15.0 / jnp.maximum(distances_to_thermal, 1.0)
        
        max_steps_f = jnp.asarray(self.params.max_steps_in_episode, dtype=jnp.float32)
        step_f = jnp.asarray(step_number, dtype=jnp.float32)
        low_speed_reward = -(max_steps_f - step_f)
        
        # rewards_array = jnp.where(
        #     out_of_bounds_xy | out_of_bounds_z,
        #     -1000.0,
        #     jnp.where(
        #         collisions,
        #         self.params.collision_penalty,
        #         jnp.where(low_speed, low_speed_reward, reward_action)
        #     )
        # )

        rewards_array = jnp.where(
            out_of_bounds_xy | out_of_bounds_z,
            -1000.0,
            jnp.where(low_speed, low_speed_reward, reward_action)
        )
        
        
        # Update history buffers
        new_speed_entries = jnp.stack([new_speeds, vertical_speeds], axis=1)
        new_speed_history = jnp.roll(state.speed_history, shift=-1, axis=1)
        new_speed_history = new_speed_history.at[:, -1, :].set(new_speed_entries)
        
        new_controls_history = jnp.roll(state.controls_history, shift=-1, axis=1)
        new_controls_history = new_controls_history.at[:, -1, :].set(new_controls)
        
        current_angles_from_wind = jax.vmap(
            lambda bank, side: calculate_angle_from_wind(bank, side, self.params)
        )(new_controls[:, 0], new_attitudes[:, 1])
        
        new_angle_from_wind_history = jnp.roll(state.angle_from_wind_history, shift=-1, axis=1)
        new_angle_from_wind_history = new_angle_from_wind_history.at[:, -1].set(current_angles_from_wind)
        
        current_wind_velocity = calculate_wind_velocity(self.params)
        new_wind_velocity_history = jnp.full(
            (self.params.num_agents, 1),
            current_wind_velocity,
            dtype=jnp.float32
        )
        
        # Calculate current distances between agents
        def compute_current_distances_for_agent(agent_idx):
            distances = jnp.linalg.norm(
                new_positions - new_positions[agent_idx],
                axis=1
            )
            rolled_distances = jnp.roll(distances, -agent_idx)
            other_distances = rolled_distances[1:]
            return other_distances
        
        current_distances = jax.vmap(compute_current_distances_for_agent)(jnp.arange(self.params.num_agents))
        new_distance_between_agents_history = jnp.roll(state.distance_between_agents_history, shift=-1, axis=1)
        new_distance_between_agents_history = new_distance_between_agents_history.at[:, -1, :].set(current_distances)
        
        # Create new state
        new_state = State(
            position=new_positions,
            speed=new_speeds,
            attitude=new_attitudes,
            controls=new_controls,
            speed_history=new_speed_history,
            controls_history=new_controls_history,
            angle_from_wind_history=new_angle_from_wind_history,
            wind_velocity_history=new_wind_velocity_history,
            distance_between_agents_history=new_distance_between_agents_history,
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
            "collisions": collisions,
            "out_of_bounds": out_of_bounds_xy | out_of_bounds_z,
            "low_speed": low_speed,
            "vertical_speeds": vertical_speeds,
            "positions": new_positions,
        }
        
        return obs, new_state, rewards, dones, info
    
    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Get observations for all agents"""
        
        def get_agent_obs(agent_idx: int) -> chex.Array:
            # Own state history
            own_speed_hist = state.speed_history[agent_idx].flatten()
            own_controls_hist = state.controls_history[agent_idx].flatten()
            own_angle_hist = state.angle_from_wind_history[agent_idx].flatten()
            own_wind_vel = state.wind_velocity_history[agent_idx].flatten()
            
            # Distance history to other agents
            # distance_hist = state.distance_between_agents_history[agent_idx].flatten()
            
            obs = jnp.concatenate([
                own_speed_hist,
                own_controls_hist,
                own_angle_hist,
                own_wind_vel,
                # distance_hist,
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
