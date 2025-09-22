import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

x_grid = np.loadtxt("../data/x_grid.txt",skiprows=1,delimiter=',')
t_grid = np.loadtxt("../data/t_grid.txt",skiprows=1,delimiter=',')
kx_grid = np.loadtxt("../data/kx_grid.txt",skiprows=1,delimiter=',')
n1e = np.loadtxt("../data/n1e_real.txt",skiprows=1,delimiter=',')
n1i = np.loadtxt("../data/n1i_real.txt",skiprows=1,delimiter=',')
Ue1x = np.loadtxt("../data/U1e_real_x.txt",skiprows=1,delimiter=',')
Ue1y = np.loadtxt("../data/U1e_real_y.txt",skiprows=1,delimiter=',')
Ui1x = np.loadtxt("../data/U1i_real_x.txt",skiprows=1,delimiter=',')
Ui1y = np.loadtxt("../data/U1i_real_y.txt",skiprows=1,delimiter=',')


# Create a gridspec
fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(3, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[2, 0])
ax6 = fig.add_subplot(gs[2, 1])

# Plot each subplot
for step in range(0,len(t_grid),10):
    ax1.cla()
    ax2.cla()
    ax3.cla()
    ax4.cla()
    ax5.cla()
    ax6.cla()

    ax1.plot(x_grid, n1e[:,step], label='Electron Density Perturbation')
    ax1.set_title('Electron Density Perturbation')
    ax1.set_ylabel('n1e')
    ax1.grid()
    
    ax2.plot(x_grid, n1i[:,step], label='Ion Density Perturbation', color='orange')
    ax2.set_title('Ion Density Perturbation')
    ax2.set_ylabel('n1i')
    ax2.grid()

    ax3.plot(x_grid, Ue1x[:,step], label='Electron Velocity Perturbation Ue1x', color='green')
    ax3.set_title('Electron Velocity Perturbation Ue1x')
    ax3.set_ylabel('Ue1x')
    ax3.grid()

    ax4.plot(x_grid, Ue1y[:,step], label='Electron Velocity Perturbation Ue1y', color='red')
    ax4.set_title('Electron Velocity Perturbation Ue1y')
    ax4.set_ylabel('Ue1y')
    ax4.grid()

    ax5.plot(x_grid, Ui1x[:,step], label='Ion Velocity Perturbation Ui1x', color='purple')
    ax5.set_title('Ion Velocity Perturbation Ui1x')
    ax5.set_ylabel('Ui1x')
    ax5.grid()

    ax6.plot(x_grid, Ui1y[:,step], label='Ion Velocity Perturbation Ui1y', color='brown')
    ax6.set_title('Ion Velocity Perturbation Ui1y')
    ax6.set_ylabel('Ui1y')
    ax6.grid()

    plt.suptitle(f'Time = {t_grid[step]:.2f}')
    plt.pause(1)