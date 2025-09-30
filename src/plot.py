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
    
    line1, = ax1.plot([], [], color = 'green', label='Electron Density Perturbation')
    line2, = ax1.plot([], [], color = 'orange', label='Ion Density Perturbation')
    ax1.set_xlim(float(np.min(x_grid)), float(np.max(x_grid)))
    ax1.set_ylim(1.5 * np.min((float(np.min(n1i)), float(np.min(n1e)))), 1.5 * np.max((float(np.max(n1i)), float(np.max(n1e)))))
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel(r'Density Perturbation [m^{-3}]')
    ax1.grid()
    ax1.set_title('Density Perturbations at t=0s')
    
    ax2.set_xlim(float(np.min(kx_grid)), float(np.max(kx_grid)))
    ax2.set_ylim(0, 1.1 * np.max(np.abs(np.fft.fft(n1e[:, 0]))) )
    ax2.set_xlabel('Wave Number kx (m$^{-1}$)')
    ax2.set_ylabel('Magnitude')
    ax2.grid()
    
    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        return line1, line2
    
    def update(frame):
        line1.set_data(x_grid, n1e[:, frame])
        line2.set_data(x_grid, n1i[:, frame])
        
        # FFT et spectre
        n1e_fft = np.fft.fftshift(np.fft.fft(n1e[:, frame], n=len(kx_grid)))
        n1i_fft = np.fft.fftshift(np.fft.fft(n1i[:, frame], n=len(kx_grid)))
        kx_shifted = np.fft.fftshift(kx_grid)

        ax2.cla()
        ax2.stem(kx_shifted, np.abs(n1e_fft), linefmt='b-', markerfmt='bo', basefmt='k-', label="Electron")
        ax2.stem(kx_shifted, np.abs(n1i_fft), linefmt='r-', markerfmt='ro', basefmt='k-', label="Ion")
        ax2.set_xlim(float(np.min(kx_grid)), float(np.max(kx_grid)))
        ax2.set_xlabel('Wave Number kx (m$^{-1}$)')
        ax2.set_ylabel('Magnitude')
        ax2.grid()

        ax1.set_title(f'Density Perturbations at t={t_grid[frame]:.2e}s')
        
        return line1, line2

    ax1.legend() 
    
    ani = FuncAnimation(fig, update, frames=len(t_grid), init_func=init,
                        blit=True, interval=100)
    ani.save('../src/density_perturbations.mp4', writer='ffmpeg', fps=10)

    plt.tight_layout()
    print("File save in ../src/density_perturbations.mp4")

def animation_velocities_electron():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    line1, = ax.plot([], [], color = 'green', label=r'U$_{e1x}$')
    line2, = ax.plot([], [], color = 'orange', label=r'U$_{e1y}$')
    ax.set_xlim(float(np.min(x_grid)), float(np.max(x_grid)))
    ax.set_ylim(1.5 * np.min((float(np.min(Ue1x)), float(np.min(Ue1y)))), 1.5 * np.max((float(np.max(Ue1x)), float(np.max(Ue1y)))))
    ax.set_xlabel('x [m]')
    ax.set_ylabel('Velocity Perturbation [m/s]')
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

    ax.legend()

    ani = FuncAnimation(fig, update, frames=len(t_grid), init_func=init,
                        blit=True, interval=100)
    ani.save('../src/electron_velocity_perturbations.mp4', writer='ffmpeg', fps=10)

    plt.tight_layout()
    print("File save in ../src/electron_velocity_perturbations.mp4")

def display_system(save = False):
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle("Evolution of the plasma")
    gs = gridspec.GridSpec(3, 1)

    ax1, ax2, ax3 = fig.add_subplot(gs[0,0]), fig.add_subplot(gs[1,0]), fig.add_subplot(gs[2,0])

    ax1.plot(x_grid, n1e[:,-1], color = "g", label = "Electron")
    ax1.plot(x_grid, n1i[:,-1], color = "orange", label = "Ion")
    ax1.set_xlim(float(np.min(x_grid)), float(np.max(x_grid)))
    ax1.set_ylim(1.5 * np.min((float(np.min(n1i)), float(np.min(n1e)))), 1.5 * np.max((float(np.max(n1i)), float(np.max(n1e)))))
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel(r'Density [m$^{-3}$]')
    ax1.legend()
    ax1.grid()

    ax2.plot(x_grid, Ue1x[:,-1], color = "g", label = "Electron")
    ax2.plot(x_grid, Ui1x[:,-1], color = "orange", label = "Ion")
    ax2.set_xlim(float(np.min(x_grid)), float(np.max(x_grid)))
    ax2.set_ylim(1.5 * np.min((float(np.min(Ue1x)), float(np.min(Ui1x)))), 1.5 * np.max((float(np.max(Ue1x)), float(np.max(Ui1x)))))
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('X-axis Velocity [m/s]')
    ax2.legend()
    ax2.grid()

    ax3.plot(x_grid, Ue1y[:,-1], color = "g", label = "Electron")
    ax3.plot(x_grid, Ui1y[:,-1], color = "orange", label = "Ion")
    ax3.set_xlim(float(np.min(x_grid)), float(np.max(x_grid)))
    ax3.set_ylim(1.5 * np.min((float(np.min(Ue1y)), float(np.min(Ui1y)))), 1.5 * np.max((float(np.max(Ue1y)), float(np.max(Ui1y)))))
    ax3.set_xlabel('x [m]')
    ax3.set_ylabel('Y-axis Velocity [m/s]')
    ax3.legend()
    ax3.grid()

    plt.tight_layout()

    if save:
        plt.savefig("../src/system.png",dpi=300)

animation_densities()
animation_velocities_electron()
display_system(True)