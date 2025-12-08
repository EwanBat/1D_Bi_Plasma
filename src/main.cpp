#include <iostream>
#include <vector>
#include <cmath>
#include "../include/plasma.hpp"
#include "../include/constants.hpp"

int main() {
    // Simulation parameters
    double x0 = 0.0;           // Start position [m]
    double xf = 1e0;         // End position [m]
    int Nx = 200;              // Number of spatial points
    double dx = (xf - x0) / (Nx - 1);
    
    double t0 = 0.0;           // Start time [s]
    double tf = 1e-7;        // End time [s]
    
    std::cout << "=== Simulation Parameters ===" << std::endl;
    std::cout << "Spatial domain: [" << x0 << ", " << xf << "] m" << std::endl;
    std::cout << "Number of spatial points: " << Nx << std::endl;
    std::cout << "Spatial step dx: " << dx << " m" << std::endl;
    std::cout << "Time domain: [" << t0 << ", " << tf << "] s" << std::endl;
    std::cout << "=============================" << std::endl;
    
    // Physical parameters
    PlasmaParams params;
    
    // Species properties
    params.m_i = consts::mp;              // Ion mass (proton) [kg]
    params.m_e = consts::me;              // Electron mass [kg]
    params.q_i = consts::e;               // Ion charge [C]
    params.q_e = -consts::e;              // Electron charge [C]
    params.n_i0 = 1.0e15;                 // Ion background density [m^-3]
    params.n_e0 = 1.0e15;                 // Electron background density [m^-3]
    
    // Temperature and pressure (assuming T ~ 1 eV = 11600 K)
    double T_i = 1000.0;                 // Ion temperature [K]
    double T_e = 1000.0;                 // Electron temperature [K]
    params.P_i0 = params.n_i0 * consts::kB * T_i;  // Ion pressure [Pa]
    params.P_e0 = params.n_e0 * consts::kB * T_e;  // Electron pressure [Pa]
    
    // Adiabatic coefficients
    params.gamma_i = 3.0;                 // Ion adiabatic coefficient (gamma = 3)
    params.gamma_e = 1.0;                 // Electron adiabatic coefficient (gamma = 1)
    
    // Physical constants
    params.epsilon_0 = consts::epsilon0;  // Vacuum permittivity [F/m]
    double Debye_length = std::sqrt(params.epsilon_0 * consts::kB * T_e / (params.n_e0 * consts::e * consts::e));
    double plasma_frequency = std::sqrt(params.n_e0 * consts::e * consts::e / (params.m_e * params.epsilon_0));

    std::cout << "\n=== Physical Parameters ===" << std::endl;
    std::cout << "Background density: " << params.n_i0 << " m^-3" << std::endl;
    std::cout << "Ion pressure: " << params.P_i0 << " Pa" << std::endl;
    std::cout << "Electron pressure: " << params.P_e0 << " Pa" << std::endl;
    std::cout << "Electron Debye lenght " << Debye_length << " m" << std::endl;
    std::cout << "Electron plasma frequency: " << plasma_frequency << " rad/s" << std::endl;
    std::cout << "============================" << std::endl;
    
    // Create spatial grid
    std::vector<double> x_grid(Nx);
    for (int i = 0; i < Nx; ++i) {
        x_grid[i] = x0 + i * dx;
    }
    
    // Initialize plasma system
    PlasmaSystem system(Nx, dx,params);
    
    // Set initial Gaussian perturbation
    double gauss_center = 0.5 * (x0 + xf);  // Center of domain
    double gauss_width = 1e-2;               // Width [m]
    double gauss_amplitude = 1.0e-3;         // Relative amplitude (0.1%)
    
    system.set_gaussian_perturbation(x_grid, gauss_center, gauss_width, gauss_amplitude);
    
    std::cout << "\n=== Initial Perturbation ===" << std::endl;
    std::cout << "Gaussian center: " << gauss_center << " m" << std::endl;
    std::cout << "Gaussian width: " << gauss_width << " m" << std::endl;
    std::cout << "Relative amplitude: " << gauss_amplitude * 100 << "%" << std::endl;
    std::cout << "=============================" << std::endl;
    
    // Time evolution
    system.run_time_evolution(tf, "../data/");

    std::cout << "\n=== Time evolution complete ===" << std::endl;
    std::string cmd= "python3 ../src/plot_time.py";
    int result = std::system(cmd.c_str());
    if (result != 0) {
        std::cerr << "Error executing plotting script." << std::endl;
    }
    return 0;
}
