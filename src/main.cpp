#include <iostream>
#include "../include/Eigen/Dense"
#include "../include/Eigen/src/Core/Matrix.h"
#include <complex>
#include <string>
#include <vector>

#include "../include/field.hpp"
#include "../include/plasma.hpp"
#include "../include/bi_system.hpp"
#include "../include/fourier.hpp"

const double n0 = 1e6; // Density in m^-3
const double T_e = 1e3; // Electron temperature in K
const double T_i = 1e3; // Ion temperature in K
const double qe = -consts::e; // Electron charge in C
const double qi = consts::e; // Ion charge in C
const double me = consts::me; // Electron mass in kg
const double mi = consts::mp; // Ion mass in kg (proton mass)


int main() {
    double t0 = 0; // Initial time in s
    double tf = 1e-5; // Final time in s
    double dt = 5e-8; // Time step in s
    const int Nt = ((tf - t0) / dt) + 1; // Number of time steps
    std::vector<double> t_grid(Nt);
    for (int i = 0; i < Nt; ++i) {t_grid[i] = t0 + i * dt;}
    std::cout << "Number of time steps: " << Nt << std::endl;

    double kxi = 1e-3; // Minimum wave number in m^-1
    double kxf = 1e-1; // Maximum wave number in m^-1
    double dkx = 1e-3; // Wave number step in m^-1
    const int Nkx = ((kxf - kxi) / dkx) + 1; // Number of wave number steps
    std::vector<double> kx_grid(Nkx);
    for (int i = 0; i < Nkx; ++i) {kx_grid[i] = kxi + i * dkx;}
    std::cout << "Number of kx steps: " << Nkx << std::endl;

    double x0 = -2.0 * M_PI / kxi; // Minimum position in m
    double xf = 2.0 * M_PI / kxi; // Maximum position in m
    double dx = 2.0 * M_PI / (dkx * Nkx); // Position step in m
    const int Nx = ((xf - x0) / dx) + 1; // Number of position steps
    std::vector<double> x_grid(Nx);
    for (int i = 0; i < Nx; ++i) {x_grid[i] = x0 + i * dx;}
    std::cout << "Number of x steps: " << Nx << std::endl;

    // Initialize plasma species
    bi_system system;
    system.m_electron.initial_constant(n0, qe, me, T_e);
    system.m_ion.initial_constant(n0, qi, mi, T_i);
    system.setup_system(Nkx, Nt);
    system.print_parameter();

    // Initial perturbations in kx-space
    Eigen::MatrixXcd n1e_k(Nkx, Nt); n1e_k.setZero();
    Eigen::MatrixXcd n1i_k(Nkx, Nt); n1i_k.setZero();
    std::vector<Eigen::MatrixXcd> U1e_k(2); U1e_k[0].resize(Nkx, Nt); U1e_k[0].setZero(); U1e_k[1].resize(Nkx, Nt); U1e_k[1].setZero();
    std::vector<Eigen::MatrixXcd> U1i_k(2); U1i_k[0].resize(Nkx, Nt); U1i_k[0].setZero(); U1i_k[1].resize(Nkx, Nt); U1i_k[1].setZero();
    double kx0 = 5e-2; // Central wave number of the perturbation in m^-1
    double sigma_kx = 1e-2; // Width of the perturbation in m^-1
    double amplitude = 1e-2 * n0; // Amplitude of the density perturbation
    for (int i = 0; i < Nkx; ++i) {
        double kx = kx_grid[i];
        double envelope = amplitude * exp(-0.5 * pow((kx - kx0) / sigma_kx, 2));
        n1e_k(i, 0) = std::complex<double>(envelope, 0);
        n1i_k(i, 0) = std::complex<double>(envelope, 0);
    }
    system.m_electron.initial_pertubation(n1e_k, U1e_k);
    system.m_ion.initial_pertubation(n1i_k, U1i_k);

    // Iterate over time and wave numbers and transform back to real space
    system.iteration(t0, tf, dt, kxi, kxf, dkx);
    system.save_datas(t_grid, kx_grid, x_grid);
    system.clear_system();

    std::string cmd = "python3 ../src/plot.py";
    int result = std::system(cmd.c_str());
    if (result != 0) {
        std::cerr << "Erreur lors de l'affichage Python.\n";
    }    
    return 0;
}