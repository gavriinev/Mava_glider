"""Plot vertical wind speed distribution at different heights."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from mava.utils.wind import WindModel, build_wind_model, wind_at


def plot_vertical_wind_distribution(
    heights=[500, 550, 600],
    x_range=(-200, 200),
    y_range=(-200, 200),
    grid_resolution=100,
    time=0.0,
):
    """Plot vertical wind speed distribution at specified heights.
    
    Parameters
    ----------
    heights : list
        List of heights in meters to visualize
    x_range : tuple
        X-axis range (min, max) in meters
    y_range : tuple
        Y-axis range (min, max) in meters
    grid_resolution : int
        Number of points along each axis
    time : float
        Time point for the simulation
    """
    # Build wind model with default settings
    wind_model = WindModel.default()
    
    # Create grid
    x = np.linspace(x_range[0], x_range[1], grid_resolution)
    y = np.linspace(y_range[0], y_range[1], grid_resolution)
    X, Y = np.meshgrid(x, y)
    
    # Create figure with subplots
    n_heights = len(heights)
    fig, axes = plt.subplots(1, n_heights, figsize=(6 * n_heights, 5))
    if n_heights == 1:
        axes = [axes]
    
    # Calculate and plot for each height
    for idx, height in enumerate(heights):
        # Create positions array
        positions = jnp.stack([
            jnp.array(X.flatten()),
            jnp.array(Y.flatten()),
            jnp.full(X.size, height, dtype=jnp.float32)
        ], axis=1)
        
        # Calculate wind at all positions
        wind_vectors = wind_at(wind_model, positions, time)
        
        # Extract vertical component (z-component)
        vertical_wind = wind_vectors[:, 2].reshape(X.shape)
        
        # Plot
        ax = axes[idx]
        contour = ax.contourf(X, Y, vertical_wind, levels=50, cmap='RdBu_r')
        ax.contour(X, Y, vertical_wind, levels=50, colors='black', linewidths=0.5, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Вертикальная скорость (м/с)', rotation=270, labelpad=20)
        
        # Mark thermal center
        thermal_center = wind_model.centers[0]
        # ax.plot(thermal_center[0], thermal_center[1], 'r*', markersize=15, 
                # label='Центр термика')
        
        # Labels and title
        ax.set_xlabel('X (м)')
        ax.set_ylabel('Y (м)')
        ax.set_title(f'Высота: {height} м')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'outputs/vertical_wind_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"График сохранен в: {output_path}")
    
    plt.show()
    
    return fig, axes


def plot_vertical_wind_profile():
    """Plot vertical wind speed as a function of height at the thermal center."""
    wind_model = WindModel.default()
    thermal_center_xy = wind_model.centers[0, :2]
    
    # Heights from 0 to 2500m
    heights = np.linspace(0, 2000, 100)
    
    # Create positions at thermal center
    positions = jnp.stack([
        jnp.full(len(heights), thermal_center_xy[0], dtype=jnp.float32),
        jnp.full(len(heights), thermal_center_xy[1], dtype=jnp.float32),
        jnp.array(heights, dtype=jnp.float32)
    ], axis=1)
    
    # Calculate wind
    wind_vectors = wind_at(wind_model, positions, 0.0)
    vertical_wind = wind_vectors[:, 2]
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(vertical_wind, heights, linewidth=2)
    ax.axhline(y=500, color='r', linestyle='--', alpha=0.7, label='500 м')
    ax.axhline(y=550, color='g', linestyle='--', alpha=0.7, label='550 м')
    ax.axhline(y=600, color='b', linestyle='--', alpha=0.7, label='600 м')
    
    ax.set_xlabel('Вертикальная скорость (м/с)', fontsize=12)
    ax.set_ylabel('Высота (м)', fontsize=12)
    ax.set_title('Профиль вертикальной скорости в центре термика', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'outputs/vertical_wind_profile.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"График профиля сохранен в: {output_path}")
    
    plt.show()
    
    return fig, ax


if __name__ == "__main__":
    print("Построение графиков распределения вертикальной скорости...")
    
    # Plot distribution at different heights
    print("\n1. Распределение на высотах 500, 550, 600 метров:")
    plot_vertical_wind_distribution(heights=[500, 550, 600])
    
    # Plot vertical profile
    print("\n2. Вертикальный профиль скорости:")
    plot_vertical_wind_profile()
    
    print("\nГотово!")
