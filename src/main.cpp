#include <iostream>
#include "../include/Eigen/Dense"
#include "../include/Eigen/src/Core/Matrix.h"
#include <complex>
#include <string>

#include "../include/field.hpp"
#include "../include/plasma.hpp"
#include "../include/system.hpp"

int main() {
    double n0 = 1e6; // Density in m^-3
    double T_e = 1e5; // Electron temperature in K
    double T_i = 1e5; // Ion temperature in K
    double qe = -consts::e; // Electron charge in C
    double qi = consts::e; // Ion charge in C
    double me = consts::me; // Electron mass in kg
    double mi = consts::mp; // Ion mass in kg (proton mass)
    
    Eigen::Vector2cd Ue1(0.0, 0.0); // Initial electron velocity perturbation
    Eigen::Vector2cd Ui1(0.0, 0.0); // Initial ion velocity perturbation
    std::complex<double> ne1(0.0, 0.0); // Initial electron density perturbation
    std::complex<double> ni1(0.0, 0.0); // Initial ion density perturbation
    double E1 = 0.0; // Initial electric field perturbation
    double B1 = 0.0; // Initial magnetic field perturbation
    
    bi_system equilibrium;
    equilibrium.m_electron.initial_constant(n0, qe, me, T_e);
    equilibrium.m_electron.initial_pertubation(n0, ne1, Ue1, qe);
    equilibrium.m_ion.initial_constant(n0, qi, mi, T_i);
    equilibrium.m_ion.initial_pertubation(n0, ni1, Ui1, qi);
    equilibrium.m_field.initial_perturbation(E1, B1);

    double kx = 1e-3; // Wavenumber in m^-1
    equilibrium.initial_wavenumber(kx);

    double t0 = 0.0; // Start time in seconds
    double tf = 1e-2; // End time in seconds
    double dt = 1e-5; // Time step in seconds
    std::string filename = "../data/plasma_data.txt"; // Output file name
    equilibrium.iteration_time(t0, tf, dt, filename);

    std::string cmd = "python3 ../src/plot.py " + filename;
    int result = system(cmd.c_str());
    if (result != 0) {
        std::cerr << "Error executing Python script." << std::endl;
    }
    return 0;
}