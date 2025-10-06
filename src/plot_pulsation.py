import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def load_complex_data(filename):
    """Charge des données complexes au format (réel,imaginaire) depuis un fichier CSV"""
    data = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                # Séparer les nombres complexes par les virgules entre parenthèses
                complex_numbers = []
                parts = line.split('),(')
                for i, part in enumerate(parts):
                    # Nettoyer les parenthèses
                    if i == 0:
                        part = part.lstrip('(')
                    if i == len(parts) - 1:
                        part = part.rstrip(')')
                    
                    # Séparer la partie réelle et imaginaire
                    real_imag = part.split(',')
                    if len(real_imag) == 2:
                        real_part = float(real_imag[0])
                        imag_part = float(real_imag[1])
                        complex_numbers.append(complex(real_part, imag_part))
                
                data.append(complex_numbers)
    
    return np.array(data)

kx_grid = np.loadtxt("../data/kx_grid.txt",delimiter=',')
eigenvalues_longitudinal = load_complex_data("../data/eigenvalues_long.txt")
eigenvalues_transverse = load_complex_data("../data/eigenvalues_trans.txt")

def plot_dispersion_relations(save = False):

    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # Plot longitudinal modes - seuls les 2 premiers modes (élimination des doublons)
    ax1.plot(kx_grid, eigenvalues_longitudinal[:, 0].real, 'b-', label='Mode 1 (Real)')
    ax1.plot(kx_grid, eigenvalues_longitudinal[:, 1].real, 'r-', label='Mode 2 (Real)')
    ax1.plot(kx_grid, eigenvalues_longitudinal[:, 2].real, 'g-', label='Mode 3 (Real)')
    ax1.plot(kx_grid, eigenvalues_longitudinal[:, 3].real, 'm-', label='Mode 4 (Real)')
    
    ax1.set_title('Longitudinal Modes Dispersion Relation')
    ax1.set_xlabel(r'Wave Number k$_x$ (m$^{-1}$)')
    ax1.set_ylabel('Frequency ω (rad/s)')
    ax1.legend()
    ax1.grid()
    
    # Plot transverse modes - seuls les 2 premiers modes (élimination des doublons)
    ax2.plot(kx_grid, eigenvalues_transverse[:, 0].real, 'b-', label='Mode 1 (Real)')
    ax2.plot(kx_grid, eigenvalues_transverse[:, 0].imag, 'b--', label='Mode 1 (Imag)')
    ax2.plot(kx_grid, eigenvalues_transverse[:, 2].real, 'g-', label='Mode 3 (Real)')
    ax2.plot(kx_grid, eigenvalues_transverse[:, 2].imag, 'g--', label='Mode 3 (Imag)')

    ax2.set_title('Transverse Modes Dispersion Relation')
    ax2.set_xlabel(r'Wave Number k$_x$ (m$^{-1}$)')
    ax2.set_ylabel('Frequency ω (rad/s)')
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    if save:
        plt.savefig("../image/dispersion_relations.png", dpi=300)

def plot_dispersion_longitudinal(save=True): #Plot longitudinal dispersion relation only, real and imaginary parts split
    
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    # Plot longitudinal modes - Real part (seulement 2 modes uniques)
    ax1.plot(kx_grid, abs(eigenvalues_longitudinal[:, 0].real), 'bo', label='Mode 1 (Real)')
    # ax1.plot(kx_grid, abs(eigenvalues_longitudinal[:, 1].real), 'ro', label='Mode 2 (Real)')
    ax1.plot(kx_grid, abs(eigenvalues_longitudinal[:, 2].real), 'go', label='Mode 3 (Real)')
    # ax1.plot(kx_grid, abs(eigenvalues_longitudinal[:, 3].real), 'mo', label='Mode 4 (Real)')

    ax1.set_title('Longitudinal Modes Dispersion Relation - Real Part')
    ax1.set_xlabel(r'Wave Number k$_x$ (m$^{-1}$)')
    ax1.set_ylabel('Frequency ω (rad/s)')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid()

    # Plot longitudinal modes - Imaginary part (seulement 2 modes uniques)
    ax2.plot(kx_grid, abs(eigenvalues_longitudinal[:, 0].imag), 'bx', label='Mode 1 (Imag)')
    ax2.plot(kx_grid, abs(eigenvalues_longitudinal[:, 2].imag), 'gx', label='Mode 3 (Imag)')

    ax2.set_title('Longitudinal Modes Dispersion Relation - Imaginary Part')
    ax2.set_xlabel(r'Wave Number k$_x$ (m$^{-1}$)')
    ax2.set_ylabel('Frequency ω (rad/s)')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    if save:
        plt.savefig("../image/dispersion_longitudinal.png", dpi=300)
        print("Figure save in ../image/dispersion_longitudinal.png")
    
    plt.show()

def plot_dispersion_transverse(save=True): #Plot transverse dispersion relation only, real and imaginary parts split
    
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Plot transverse modes - Real part (seulement 2 modes uniques)
    ax1.plot(kx_grid, abs(eigenvalues_transverse[:, 0].real), 'bo', label='Mode 1 (Real)')
    ax1.plot(kx_grid, abs(eigenvalues_transverse[:, 2].real), 'go', label='Mode 3 (Real)')

    ax1.set_title('Transverse Modes Dispersion Relation - Real Part')
    ax1.set_xlabel(r'Wave Number k$_x$ (m$^{-1}$)')
    ax1.set_ylabel('Frequency ω (rad/s)')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid()
    
    # Plot transverse modes - Imaginary part (seulement 2 modes uniques)
    ax2.plot(kx_grid, abs(eigenvalues_transverse[:, 0].imag), 'bx', label='Mode 1 (Imag)')
    ax2.plot(kx_grid, abs(eigenvalues_transverse[:, 2].imag), 'gx', label='Mode 3 (Imag)')

    ax2.set_title('Transverse Modes Dispersion Relation - Imaginary Part')
    ax2.set_xlabel(r'Wave Number k$_x$ (m$^{-1}$)')
    ax2.set_ylabel('Frequency ω (rad/s)')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    if save:
        plt.savefig("../image/dispersion_transverse.png", dpi=300)
        print("Figure save in ../image/dispersion_transverse.png")
    plt.show()

# plot_dispersion_relations(save=True)
plot_dispersion_longitudinal(save=True)
plot_dispersion_transverse(save=True)