#pragma once

/**
 * @brief Physical constants used in plasma simulations
 */
namespace consts {
    const double c = 299792458;                // Speed of light in vacuum [m/s]
    const double epsilon0 = 8.854187817e-12;   // Permittivity of free space [F/m]
    const double mu0 = 1.2566370614e-6;        // Permeability of free space [H/m]
    const double e = 1.602176634e-19;          // Elementary charge [C]
    const double me = 9.10938356e-31;          // Electron mass [kg]
    const double mp = 1.67262192369e-27;       // Proton mass [kg]
    const double k_B = 1.380649e-23;           // Boltzmann constant [J/K]
}

/**
 * @brief Plasma simulation parameters
 * 
 * Contains all physical parameters for both ion and electron species.
 * Subscript 's' refers to species: 'i' for ions, 'e' for electrons.
 */
struct PlasmaParams {
    // Species properties (s = i for ions, e for electrons)
    double m_i, m_e;           // Mass [kg]
    double q_i, q_e;           // Charge [C]
    double n_i0, n_e0;         // Background density [m^-3]
    double P_i0, P_e0;         // Background pressure [Pa]
    double gamma_i, gamma_e;   // Adiabatic coefficient [-]
    double T_i, T_e;           // Temperature [K]
    double cs_e, cs_i;         // Sound speed [m/s]
    
    // Physical constants
    double epsilon_0;          // Vacuum permittivity [F/m]
};