import sys
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python plot.py <filename>")
    sys.exit(1)

filename = sys.argv[1]
print(f"Loading data from {filename}...")

try:
    data = np.loadtxt(filename)
    print("Données chargées avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du fichier : {e}")

time = data[:, 0]
n1_e = data[:, 1]
U1_e = data[:, 2:4]
n1_i = data[:, 4]
U1_i = data[:, 5:7]
E1 = data[:, 7]
B1 = data[:, 8]

fig = plt.figure(figsize=(12, 10))
gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 1])
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])
ax3 = plt.subplot(gs[3])

ax0.plot(time, n1_e, label='n1_e', color='blue')
ax0.plot(time, n1_i, label='n1_i', color='orange')
ax0.set_xlabel('Time [s]')
ax0.set_ylabel('Density Perturbation [m$\\^3$]')
ax0.legend()
ax0.grid()
ax0.set_title('Density Perturbations over Time')

ax1.plot(time, U1_e[:, 0], label='U1_e_x', color='blue')
ax1.plot(time, U1_e[:, 1], label='U1_e_y', color='cyan')
ax1.plot(time, U1_i[:, 0], label='U1_i_x', color='orange')
ax1.plot(time, U1_i[:, 1], label='U1_i_y', color='red')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Velocity Perturbation [m/s]')
ax1.legend()
ax1.grid()
ax1.set_title('Velocity Perturbations over Time')

ax2.plot(time, E1, label='E1', color='green')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Electric Field Perturbation [V/m]')
ax2.legend()
ax2.grid()
ax2.set_title('Electric Field Perturbation over Time')  

ax3.plot(time, B1, label='B1', color='purple')
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Magnetic Field Perturbation [T]')
ax3.legend()
ax3.grid()
ax3.set_title('Magnetic Field Perturbation over Time')

plt.tight_layout()
plt.savefig('plasma_perturbations.png', dpi=300)
plt.show()

