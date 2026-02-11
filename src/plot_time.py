#!/usr/bin/env python3
"""
Script to visualize temporal evolution of plasma data
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors
import os

# Load data
data_dir = "../data/"

# Load spatial grid
x_grid = np.loadtxt(data_dir + "x_grid.txt")

# Load time array
time = np.loadtxt(data_dir + "time.txt")

# Load temporal data (each row = one time step, each column = one position)
n_i1_time = np.loadtxt(data_dir + "n_i1_time.txt")
u_i1_time = np.loadtxt(data_dir + "u_i1_time.txt")
n_e1_time = np.loadtxt(data_dir + "n_e1_time.txt")
u_e1_time = np.loadtxt(data_dir + "u_e1_time.txt")
E_time = np.loadtxt(data_dir + "E_time.txt")

# Ensure data is 2D
if n_i1_time.ndim == 1:
    n_i1_time = n_i1_time.reshape(1, -1)
    u_i1_time = u_i1_time.reshape(1, -1)
    n_e1_time = n_e1_time.reshape(1, -1)
    u_e1_time = u_e1_time.reshape(1, -1)
    E_time = E_time.reshape(1, -1)

Nt = len(time)
print(f"Total number of temporal points saved: {Nt}")
print(f"Number of spatial points: {len(x_grid)}")
print(f"Time min: {time[0]*1e9:.3f} ns, max: {time[-1]*1e9:.3f} ns")

def plot_time_evolution():
    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle("Temporal evolution of 1D plasma", fontsize=16)

    # Individual plots at different time steps
    n_snapshots = min(5, Nt)
    snapshot_indices = np.linspace(0, Nt-1, n_snapshots, dtype=int)

    # 1. Ion density
    ax = axes[0, 0]
    for idx in snapshot_indices:
        ax.plot(x_grid, n_i1_time[idx, :], label=f"t={time[idx]*1e9:.2f} ns")
    ax.set_xlabel("Position x [m]")
    ax.set_ylabel(r"$n_{i1}$ [m$^{-3}$]")
    ax.set_title("Ion density perturbation")
    ax.legend()
    ax.grid(True)

    # 2. Ion velocity
    ax = axes[0, 1]
    for idx in snapshot_indices:
        ax.plot(x_grid, u_i1_time[idx, :], label=f"t={time[idx]*1e9:.2f} ns")
    ax.set_xlabel("Position x [m]")
    ax.set_ylabel(r"$u_{i1}$ [m/s]")
    ax.set_title("Ion velocity perturbation")
    ax.legend()
    ax.grid(True)

    # 3. Electron density
    ax = axes[1, 0]
    for idx in snapshot_indices:
        ax.plot(x_grid, n_e1_time[idx, :], label=f"t={time[idx]*1e9:.2f} ns")
    ax.set_xlabel("Position x [m]")
    ax.set_ylabel(r"$n_{e1}$ [m$^{-3}$]")
    ax.set_title("Electron density perturbation")
    ax.legend()
    ax.grid(True)

    # 4. Electron velocity
    ax = axes[1, 1]
    for idx in snapshot_indices:
        ax.plot(x_grid, u_e1_time[idx, :], label=f"t={time[idx]*1e9:.2f} ns")
    ax.set_xlabel("Position x [m]")
    ax.set_ylabel(r"$u_{e1}$ [m/s]")
    ax.set_title("Electron velocity perturbation")
    ax.legend()
    ax.grid(True)

    # 5. Electric field
    ax = axes[2, 0]
    for idx in snapshot_indices:
        ax.plot(x_grid, E_time[idx, :], label=f"t={time[idx]*1e9:.2f} ns")
    ax.set_xlabel("Position x [m]")
    ax.set_ylabel("E [V/m]")
    ax.set_title("Electric field perturbation")
    ax.legend()
    ax.grid(True)

    # 6. Spatio-temporal diagram of n_i1
    ax = axes[2, 1]
    extent = [x_grid[0], x_grid[-1], time[0]*1e6, time[-1]*1e6]
    im = ax.imshow(n_e1_time, aspect='auto', origin='lower', extent=extent, cmap='RdBu_r')
    ax.set_xlabel("Position x [m]")
    ax.set_ylabel("Time [ns]")
    ax.set_title(r"Spatio-temporal diagram $n_{i1}$")
    plt.colorbar(im, ax=ax, label=r"$n_{e1}$ [m$^{-3}$]")

    plt.tight_layout()
    plt.savefig("../image/time_evolution.png", dpi=150)
    print("Figure saved: ../image/time_evolution.png")

def animate_time_evolution():
    # Create animation
    print("\nCreating animation...")
    fig_anim, axes_anim = plt.subplots(2, 2, figsize=(12, 8))
    fig_anim.suptitle("Temporal evolution animation", fontsize=14)

    lines = []
    titles = ["Ion density n_i1", "Ion velocity u_i1", 
            "Electron density n_e1", "Electric field E"]
    data_arrays = [n_i1_time, u_i1_time, n_e1_time, E_time]
    ylabels = ["n_i1 [m⁻³]", "u_i1 [m/s]", "n_e1 [m⁻³]", "E [V/m]"]

    for idx, (ax, data, title, ylabel) in enumerate(zip(axes_anim.flat, data_arrays, titles, ylabels)):
        line, = ax.plot(x_grid, data[0, :], 'b-', lw=2)
        lines.append(line)
        ax.set_xlabel("Position x [m]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)
        
        # Set y limits
        ymin, ymax = data.min(), data.max()
        margin = 0.1 * (ymax - ymin) if ymax != ymin else 1
        ax.set_ylim(ymin - margin, ymax + margin)

    time_text = fig_anim.suptitle('', fontsize=12)

    def animate(frame):
        for line, data in zip(lines, data_arrays):
            line.set_ydata(data[frame, :])
        fig_anim.suptitle(f"Temporal evolution animation - Time: {time[frame]*1e9:.3f} ns", fontsize=14)
        return lines

    anim = FuncAnimation(fig_anim, animate, frames=Nt, interval=50, blit=True, repeat=True)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    anim.save('../image/plasma_animation.gif', writer='pillow', fps=20)
    print("Animation saved: ../image/plasma_animation.gif")
    plt.close(fig_anim)

def fourier_analysis():
    # Fourier analysis
    print("\nPerforming Fourier analysis...")

    var = n_e1_time  # Example with ion density
    dx = x_grid[1] - x_grid[0]
    k = np.fft.fftfreq(len(x_grid), d=dx)

    dt = time[1] - time[0]
    omega = np.fft.fftfreq(len(time), d=dt)

    F_var = np.fft.fft2(var)
    magnitude = np.abs(F_var)

    plt.figure(figsize=(10, 6))
    extent = [k[0], k[-1], omega[0], omega[-1]]
    plt.imshow(magnitude, extent=extent, aspect='auto', origin='lower', cmap='inferno', norm=matplotlib.colors.LogNorm())
    plt.colorbar(label='Magnitude')
    plt.xlabel('Wavenumber k [rad/m]')
    plt.ylabel('Frequency ω [rad/s]')
    plt.xscale('symlog')
    plt.yscale('symlog')
    plt.title('Fourier Transform')
    plt.tight_layout()
    plt.savefig("../image/fourier_analysis.png", dpi=150)
    print("Figure saved: ../image/fourier_analysis.png")

if __name__ == "__main__":
    plot_time_evolution()
    fourier_analysis()
    animate_time_evolution()