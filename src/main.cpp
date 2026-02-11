#include <iostream>
#include <vector>
#include <cmath>
#include "../include/plasma.hpp"
#include "../include/constants.hpp"

int main() {
    // ================================ SIMULATION PARAMETERS ========================
    
    // Spatial domain
    double x0 = 0.0;              // Start position [m]
    double xf = 1e-1;              // End position [m]
    int Nx = 200;                 // Number of spatial points
    double dx = (xf - x0) / (Nx + 1);
    
    // Spatial grid
    std::vector<double> x_grid(Nx);
    for (int i = 0; i < Nx; ++i) {
        x_grid[i] = x0 + i * dx;
    }
    
    // Time domain
    double t0 = 0.0;              // Start time [s]
    double tf = 3e-7;             // End time [s] We observe the space charge effect at 1e-7 s
    double dt_data = 1e-9;        // Data saving interval [s]
    
    std::cout << "=== Simulation Parameters ===" << std::endl;
    std::cout << "Spatial domain: [" << x0 << ", " << xf << "] m" << std::endl;
    std::cout << "Number of spatial points: " << Nx << std::endl;
    std::cout << "Spatial step dx: " << dx << " m" << std::endl;
    std::cout << "Time domain: [" << t0 << ", " << tf << "] s" << std::endl;
    std::cout << "=============================" << std::endl;
    
    // ============================== PHYSICAL PARAMETERS ======================
    PlasmaParams params;
    
    // Species properties: ions (protons) and electrons
    params.m_i = consts::mp;                                        // Ion mass [kg]
    params.m_e = consts::me;                                        // Electron mass [kg]
    params.q_i = consts::e;                                         // Ion charge [C]
    params.q_e = -consts::e;                                        // Electron charge [C]
    params.n_i0 = 1.0e15;                                           // Ion background density [m^-3]
    params.n_e0 = 1.0e15;                                           // Electron background density [m^-3]
    
    // Temperature and pressure (assuming T ~ 1 eV ≈ 11600 K)
    params.T_i = 500.0;                                           // Ion temperature [K]
    params.T_e = 15000.0;                                           // Electron temperature [K]
    params.P_i0 = params.n_i0 * consts::k_B * params.T_i;          // Ion pressure [Pa]
    params.P_e0 = params.n_e0 * consts::k_B * params.T_e;          // Electron pressure [Pa]
    
    // Adiabatic coefficients (γ = 3 for ions, γ = 1 for isothermal electrons)
    params.gamma_i = 3.0;
    params.gamma_e = 1.0;
    
    // Physical constants and derived quantities
    params.epsilon_0 = consts::epsilon0;                            // Vacuum permittivity [F/m]
    const double Debye_length = std::sqrt(params.epsilon_0 * consts::k_B * params.T_e 
                                          / (params.n_e0 * consts::e * consts::e));
    const double plasma_frequency = std::sqrt(params.n_e0 * consts::e * consts::e 
                                              / (params.m_e * params.epsilon_0));
    params.cs_e = std::sqrt(consts::k_B * params.T_e / params.m_e); // Electron sound speed [m/s]
    params.cs_i = std::sqrt(consts::k_B * params.T_i / params.m_i); // Ion sound speed [m/s]

    std::cout << "\n=== Physical Parameters ===" << std::endl;
    std::cout << "Background density: " << params.n_i0 << " m^-3" << std::endl;
    std::cout << "Ion pressure: " << params.P_i0 << " Pa" << std::endl;
    std::cout << "Electron pressure: " << params.P_e0 << " Pa" << std::endl;
    std::cout << "Electron Debye length: " << Debye_length << " m" << std::endl;
    std::cout << "Electron plasma frequency: " << plasma_frequency << " rad/s" << std::endl;
    std::cout << "============================" << std::endl;
    
    // ============================== PLASMA SYSTEM INITIALIZATION ======================
    PlasmaSystem system(Nx, dx, params);
    
    // === INITIAL PERTURBATION ===
    double amp_E0 = 1e3; // Amplitude of electric potential perturbation [V]
    Eigen::VectorXd E0(Nx+1);
    for (int i = 0; i < Nx; ++i) {
        E0(i) = amp_E0;
    }    
    E0(Nx) = E0(Nx - 1); // Mirror boundary condition
    system.set_constant_electric_field(E0);
    
    // === TIME EVOLUTION ===
    system.run_time_evolution(tf, dt_data, "../data/");

    std::cout << "\n=== Time evolution complete ===" << std::endl;
    
    // === Run plotting script ===
    std::string cmd = "python3 ../src/plot_time.py";
    int result = std::system(cmd.c_str());
    if (result != 0) {
        std::cerr << "Error executing plotting script." << std::endl;
    }
    
    return 0;
}
