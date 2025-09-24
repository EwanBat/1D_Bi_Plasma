import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import numpy as np

x_grid = np.loadtxt("../data/x_grid.txt",delimiter=',')
t_grid = np.loadtxt("../data/t_grid.txt",delimiter=',')
kx_grid = np.loadtxt("../data/kx_grid.txt",delimiter=',')
n1e = np.loadtxt("../data/n1e_real.txt",delimiter=',')
n1i = np.loadtxt("../data/n1i_real.txt",delimiter=',')
Ue1x = np.loadtxt("../data/U1e_real_x.txt",delimiter=',')
Ue1y = np.loadtxt("../data/U1e_real_y.txt",delimiter=',')
Ui1x = np.loadtxt("../data/U1i_real_x.txt",delimiter=',')
Ui1y = np.loadtxt("../data/U1i_real_y.txt",delimiter=',')

def animation_densities():
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    line1, = ax1.plot([], [], 'b-', label='Electron Density Perturbation')
    line2, = ax1.plot([], [], 'r-', label='Ion Density Perturbation')
    ax1.set_xlim(float(np.min(x_grid)), float(np.max(x_grid)))
    ax1.set_ylim(1.5 * np.min((float(np.min(n1i)), float(np.min(n1e)))), 1.5 * np.max((float(np.max(n1i)), float(np.max(n1e)))))
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('Density Perturbation')
    ax1.legend()
    ax1.grid()
    ax1.set_title('Density Perturbations at t=0s')
    
    line3, = ax2.plot([], [], 'g-', label='Wave Number Spectrum')
    ax2.set_xlim(float(np.min(kx_grid)), float(np.max(kx_grid)))
    ax2.set_ylim(0, 1.1 * np.max(np.abs(np.fft.fft(n1e[:, 0]))) )
    ax2.set_xlabel('Wave Number kx (m$^{-1}$)')
    ax2.set_ylabel('Magnitude')
    ax2.legend()
    ax2.grid()
    
    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        return line1, line2, line3
    
    def update(frame):
        line1.set_data(x_grid, n1e[:, frame])
        line2.set_data(x_grid, n1i[:, frame])
        
        n1e_fft = np.fft.fft(n1e[:, frame], n=len(kx_grid))
        line3.set_data(kx_grid, np.abs(n1e_fft))

        ax1.set_title(f'Density Perturbations at t={t_grid[frame]:.2e}s')
        
        return line1, line2, line3
    
    ani = FuncAnimation(fig, update, frames=len(t_grid), init_func=init,
                        blit=True, interval=100)
    ani.save('../src/density_perturbations.mp4', writer='ffmpeg', fps=10)

    plt.tight_layout()

def animation_velocities_electron():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    line1, = ax.plot([], [], 'b-', label='Electron Velocity Perturbation Ue1x')
    line2, = ax.plot([], [], 'r-', label='Electron Velocity Perturbation Ue1y')
    ax.set_xlim(float(np.min(x_grid)), float(np.max(x_grid)))
    ax.set_ylim(1.5 * np.min((float(np.min(Ue1x)), float(np.min(Ue1y)))), 1.5 * np.max((float(np.max(Ue1x)), float(np.max(Ue1y)))))
    ax.set_xlabel('x (m)')
    ax.set_ylabel('Velocity Perturbation')
    ax.legend()
    ax.grid()
    ax.set_title('Electron Velocity Perturbations at t=0s')
    
    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        return line1, line2
    
    def update(frame):
        line1.set_data(x_grid, Ue1x[:, frame])
        line2.set_data(x_grid, Ue1y[:, frame])
        
        ax.set_title(f'Electron Velocity Perturbations at t={t_grid[frame]:.2e}s')
        
        return line1, line2
    
    ani = FuncAnimation(fig, update, frames=len(t_grid), init_func=init,
                        blit=True, interval=100)
    ani.save('../src/electron_velocity_perturbations.mp4', writer='ffmpeg', fps=10)

    plt.tight_layout()

animation_densities()
animation_velocities_electron()