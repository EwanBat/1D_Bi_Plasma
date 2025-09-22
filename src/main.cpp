#include <iostream>
#include "../include/Eigen/Dense"
#include "../include/Eigen/src/Core/Matrix.h"
#include <complex>
#include <string>

#include "../include/field.hpp"
#include "../include/plasma.hpp"
#include "../include/bi_system.hpp"

int main() {
    double n0 = 1e6; // Density in m^-3
    double T_e = 1e5; // Electron temperature in K
    double T_i = 1e5; // Ion temperature in K
    double qe = -consts::e; // Electron charge in C
    double qi = consts::e; // Ion charge in C
    double me = consts::me; // Electron mass in kg
    double mi = consts::mp; // Ion mass in kg (proton mass)
    
    double kxi = 1e-3; // Minimum wave number in m^-1
    double kxf = 1e-1; // Maximum wave number in m^-1
    double dkx = 1e-3; // Wave number step in m^-1
    const int Nkx = ((kxf - kxi) / dkx) + 1; // Number of wave number steps

    double t0 = 0; // Initial time in s
    double tf = 1e-2; // Final time in s
    double dt = 1e-4; // Time step in s
    const int Tstep = ((tf - t0) / dt) + 1; // Number of time steps
    
    std::cout << "Number of kx steps: " << Nkx << std::endl;
    std::cout << "Number of time steps: " << Tstep << std::endl;

    // Initialize plasma species
    bi_system system;
    system.m_electron.initial_constant(n0, qe, me, T_e);
    system.m_ion.initial_constant(n0, qi, mi, T_i);
    system.setup_system(Nkx, Tstep);

    // Iterate over time and wave numbers
    system.iteration(t0, tf, dt, kxi, kxf, dkx);

    // std::string filename = "output.csv";
    // std::string cmd = "python3 ../src/plot.py " + filename;
    // int result = system(cmd.c_str());
    // if (result != 0) {
    //     std::cerr << "Error executing Python script." << std::endl;
    // }
    return 0;
}