#pragma once

#include "Eigen/Dense"
#include "Eigen/src/Core/Matrix.h"
#include "constants.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>  // For std::setprecision
#include <chrono>   // For timing measurements

/**
 * @brief Class representing the state of a 1D bi-fluid plasma system
 * 
 * Solves the coupled equations for ion and electron density/velocity perturbations
 * and electric field using finite volume method with MUSCL reconstruction.
 * 
 * Governing equations:
 * 1) Continuity: dn_s/dt + d(n_s*u_s)/dx = 0
 * 2) Momentum: du_s/dt + u_s*du_s/dx + (1/ρ_s)*dP_s/dx = q_s*E/m_s
 * 3) Poisson: d²Φ/dx² = -(q_i*n_i + q_e*n_e)/ε_0, E = -dΦ/dx
 * 
 * where s = i (ions) or e (electrons)
 */
class PlasmaSystem {
public:
    // State variables (perturbations around background values)
    Eigen::VectorXd n_i1;      // Ion density perturbation [m^-3]
    Eigen::VectorXd u_i1;      // Ion velocity [m/s]
    Eigen::VectorXd n_e1;      // Electron density perturbation [m^-3]
    Eigen::VectorXd u_e1;      // Electron velocity [m/s]
    Eigen::VectorXd E;         // Electric field [V/m]
    
    // Grid and time stepping parameters
    int Nx;                    // Number of spatial points
    double dx;                 // Spatial step size [m]
    double dt;                 // Time step size [s] (adaptive via CFL)
    double cfl_factor;         // CFL safety factor (typically 0.5)
    
    // Physical parameters for both species
    PlasmaParams params;
    
private:
    // File handles for time-series output
    std::ofstream file_t, file_n_i1, file_u_i1, file_n_e1, file_u_e1, file_E;
    
    // Temporary storage vectors to avoid repeated allocations
    Eigen::VectorXd Phi, Phi_star;        // Electric potential [V]
    Eigen::VectorXd E_star;               // Electric field at predictor step [V/m]

    // MUSCL reconstruction: left (L) and right (R) states at cell interfaces
    double n_iR, n_iL;         // Ion density states
    double u_iR, u_iL;         // Ion velocity states
    double n_eR, n_eL;         // Electron density states
    double u_eR, u_eL;         // Electron velocity states
    
    // Intermediate states for Heun's method
    Eigen::VectorXd n_e_star, n_i_star;   // Density at predictor step
    Eigen::VectorXd u_e_star, u_i_star;   // Velocity at predictor step

    // Numerical fluxes at cell interfaces
    Eigen::VectorXd F_e_n, F_i_n;         // Density fluxes [m^-2 s^-1]
    Eigen::VectorXd F_i_u, F_e_u;         // Velocity fluxes [m^2 s^-2]

    // Slope limiters for MUSCL reconstruction
    Eigen::VectorXd minmod_i_n, minmod_e_n;   // Density slopes [m^-4]
    Eigen::VectorXd minmod_i_u, minmod_e_u;   // Velocity slopes [s^-1]

public:
    /**
     * @brief Constructor for PlasmaSystem
     * 
     * Initializes all state vectors and temporary storage.
     * Computes initial time step based on CFL condition.
     * 
     * @param nx Number of spatial grid points
     * @param dx_val Spatial step size [m]
     * @param p Physical parameters (masses, charges, etc.)
     * @param cfl CFL safety factor for time step (default: 0.5)
     */
    PlasmaSystem(int nx, double dx_val, const PlasmaParams& p, const double cfl = 0.5) 
        : Nx(nx), dx(dx_val), params(p), cfl_factor(cfl) {
        // Initialize state vectors to zero
        n_i1.resize(Nx+1); n_i1.setZero();
        u_i1.resize(Nx+1); u_i1.setZero();
        n_e1.resize(Nx+1); n_e1.setZero();
        u_e1.resize(Nx+1); u_e1.setZero();
        E.resize(Nx+1); E.setZero();
        
        // Initialize temporary storage vectors
        Phi.resize(Nx+1); Phi_star.resize(Nx+1);
        E_star.resize(Nx+1);

        n_e_star.resize(Nx+1); n_i_star.resize(Nx+1);
        u_e_star.resize(Nx+1); u_i_star.resize(Nx+1);

        F_e_n.resize(Nx+1); F_i_n.resize(Nx+1);
        F_i_u.resize(Nx+1); F_e_u.resize(Nx+1);

        minmod_i_n.resize(Nx+1); minmod_e_n.resize(Nx+1);
        minmod_i_u.resize(Nx+1); minmod_e_u.resize(Nx+1);

        // Compute initial time step based on CFL condition
        compute_time_step();
    }

private:
    /**
     * @brief Compute adaptive time step based on CFL condition
     * 
     * Time step is dt = CFL * dx / max(|u| + c_s)
     * where c_s is the sound speed for each species.
     */
    void compute_time_step() {
        double max_speed = 0.0;
        for (int i = 0; i < Nx+1; ++i) {
            double speed_i = std::abs(u_i1(i)) + params.cs_i;
            double speed_e = std::abs(u_e1(i)) + params.cs_e;
            max_speed = std::max({max_speed, speed_i, speed_e});
        }
        dt = cfl_factor * dx / max_speed;
    }

    /**
     * @brief Solve Poisson equation for electric potential
     * 
     * @param n_i Ion density perturbation [m^-3]
     * @param n_e Electron density perturbation [m^-3]
     * @param Phi_out Output electric potential [V]
     */
    void calc_potential(Eigen::VectorXd& n_i, Eigen::VectorXd& n_e, Eigen::VectorXd& Phi_out);

    /**
     * @brief Compute electric field from potential gradient
     * 
     * @param Phi_in Electric potential [V]
     * @param E_out Output electric field [V/m]
     */
    void calc_electric_field(Eigen::VectorXd& Phi_in, Eigen::VectorXd& E_out);
    
    /**
     * @brief Compute minmod slope limiter for all variables
     * 
     * @param n_i Ion density perturbation [m^-3]
     * @param n_e Electron density perturbation [m^-3]
     * @param u_i Ion velocity [m/s]
     * @param u_e Electron velocity [m/s]
     */
    void minmod(Eigen::VectorXd& n_i, Eigen::VectorXd& n_e, 
                Eigen::VectorXd& u_i, Eigen::VectorXd& u_e);

    /**
     * @brief MUSCL reconstruction at cell interface
     * 
     * @param i Cell interface index
     * @param n_i Ion density perturbation [m^-3]
     * @param n_e Electron density perturbation [m^-3]
     * @param u_i Ion velocity [m/s]
     * @param u_e Electron velocity [m/s]
     */
    void MUSCL_reconstruction(int i, Eigen::VectorXd& n_i, Eigen::VectorXd& n_e,
                             Eigen::VectorXd& u_i, Eigen::VectorXd& u_e);
    
    /**
     * @brief Compute numerical fluxes using Rusanov scheme
     * 
     * @param i Cell interface index
     * @param n_i Ion density perturbation [m^-3]
     * @param n_e Electron density perturbation [m^-3]
     * @param u_i Ion velocity [m/s]
     * @param u_e Electron velocity [m/s]
     */
    void calc_fluxes(int i, Eigen::VectorXd& n_i, Eigen::VectorXd& n_e,
                     Eigen::VectorXd& u_i, Eigen::VectorXd& u_e);

    /**
     * @brief Compute ion pressure gradient
     * 
     * @param i Spatial grid index
     * @param n_i Ion density perturbation [m^-3]
     * @return Pressure gradient [Pa/m]
     */
    double calc_pressure_gradient_i(int i, Eigen::VectorXd& n_i);
    
    /**
     * @brief Compute electron pressure gradient
     * 
     * @param i Spatial grid index
     * @param n_e Electron density perturbation [m^-3]
     * @return Pressure gradient [Pa/m]
     */
    double calc_pressure_gradient_e(int i, Eigen::VectorXd& n_e);

    /**
     * @brief Perform one time step using Heun's method
     */
    void step_euler();

public:
    /**
     * @brief Initialize time evolution output files
     * 
     * Creates and opens output files for spatial grid and time-series data.
     * Files remain open throughout simulation for efficient writing.
     * 
     * @param x_grid Spatial grid coordinates [m]
     * @param prefix Directory path prefix for output files
     */
    void init_time_files(const std::vector<double>& x_grid, const std::string& prefix = "../data/") {
        // Save spatial grid to file
        std::ofstream file_x(prefix + "x_grid.txt");
        if (!file_x.is_open()) {
            std::cerr << "Error: Cannot open " << prefix << "x_grid.txt" << std::endl;
            return;
        }
        for (int i = 0; i < Nx+1; ++i) {
            file_x << x_grid[i] << "\n";
        }
        file_x.close();
        
        // Open time-series files (kept open for duration of simulation)
        file_t.open(prefix + "time.txt");
        file_n_i1.open(prefix + "n_i1_time.txt");
        file_u_i1.open(prefix + "u_i1_time.txt");
        file_n_e1.open(prefix + "n_e1_time.txt");
        file_u_e1.open(prefix + "u_e1_time.txt");
        file_E.open(prefix + "E_time.txt");
        
        if (!file_t.is_open() || !file_n_i1.is_open() || !file_u_i1.is_open() || 
            !file_n_e1.is_open() || !file_u_e1.is_open() || !file_E.is_open()) {
            std::cerr << "Error: Cannot open output files" << std::endl;
            return;
        }
        
        std::cout << "Time-series files initialized in " << prefix << std::endl;
    }
    
    /**
     * @brief Append current state to time evolution files
     * 
     * Writes current time and spatial profile of all variables.
     * Flushes buffers periodically for data safety.
     * 
     * @param t Current simulation time [s]
     */
    void append_time_data(double& t) {
        // Write current time
        file_t << t << "\n";
        
        // Write spatial profiles for all variables
        for (int i = 0; i < Nx+1; ++i) {
            file_n_i1 << n_i1(i);
            file_u_i1 << u_i1(i);
            file_n_e1 << n_e1(i);
            file_u_e1 << u_e1(i);
            file_E << E(i);
            
            if (i < Nx) {  // Add space separator between values
                file_n_i1 << " ";
                file_u_i1 << " ";
                file_n_e1 << " ";
                file_u_e1 << " ";
                file_E << " ";
            }
        }
        
        // Newline after each time step
        file_n_i1 << "\n";
        file_u_i1 << "\n";
        file_n_e1 << "\n";
        file_u_e1 << "\n";
        file_E << "\n";
        
        // Periodic flush for data safety (every 100 steps)
        static int flush_counter = 0;
        if (++flush_counter % 100 == 0) {
            file_t.flush();
            file_n_i1.flush();
            file_u_i1.flush();
            file_n_e1.flush();
            file_u_e1.flush();
            file_E.flush();
        }
    }
    
    /**
     * @brief Close output files
     */
    void close_time_files() {
        if (file_t.is_open()) file_t.close();
        if (file_n_i1.is_open()) file_n_i1.close();
        if (file_u_i1.is_open()) file_u_i1.close();
        if (file_n_e1.is_open()) file_n_e1.close();
        if (file_u_e1.is_open()) file_u_e1.close();
        if (file_E.is_open()) file_E.close();
    }
    
    /**
     * @brief Run time evolution and save data at each time step
     * 
     * Advances the simulation from t=0 to t=tf using Heun's method.
     * Outputs progress to console and saves data at every time step.
     * 
     * @param tf Final simulation time [s]
     * @param prefix Directory path prefix for output files
     */
    void run_time_evolution(double tf, const std::string& prefix = "../data/") {
        double t = 0.0;
        int step = 0;
        std::vector<double> x_grid(Nx+1);
        for (int i = 0; i < Nx+1; ++i) {
            x_grid[i] = i * dx;
        }
        
        std::cout << "\n=== Initializing output files ===" << std::endl;
        init_time_files(x_grid, prefix);
        
        std::cout << "Starting simulation..." << std::endl;
        
        // Start timing the simulation
        auto start_time = std::chrono::high_resolution_clock::now();
        
        while (t < tf) {
            step_euler();
            if (t == 0.0) {
                std::cout << "Initial dt: " << dt << " s" << std::endl;
            }
            t += dt;
            step++;
            append_time_data(t);
            
            // Display progress bar
            std::cout << "\rProgress: " << std::fixed << std::setprecision(2) 
                      << (t/tf)*100.0 << "% | Step: " << step 
                      << std::flush;
        }
        
        // Stop timing and compute duration
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        close_time_files();
        
        std::cout << std::endl << "Data saved in " << prefix << " | " << step << " steps" << std::endl;
        std::cout << "Simulation time: " << duration.count() / 1000.0 << " s" << std::endl;
        std::cout << "=============================" << std::endl;
    }
    
    /**
     * @brief Set initial Gaussian perturbation for electric field
     * 
     * Initializes electric field with Gaussian profile.
     * Density and velocity perturbations remain zero.
     * 
     * @param x_grid Spatial grid coordinates [m]
     * @param x0 Center position of Gaussian [m]
     * @param sigma_x Width (standard deviation) of Gaussian [m]
     * @param amplitude Amplitude of density perturbation (unused in current implementation)
     * @param E_amplitude Amplitude of electric field [V/m]
     */
    void set_gaussian_perturbation(const std::vector<double>& x_grid, 
                                   double x0, double sigma_x, double amplitude,
                                   double E_amplitude) {
        for (int i = 0; i < Nx; ++i) {
            const double x = x_grid[i];
            const double gauss = amplitude * std::exp(-0.5 * std::pow((x - x0) / sigma_x, 2));
            
            // Initialize electric field with Gaussian profile
            E(i) = E_amplitude * std::exp(-0.5 * std::pow((x - x0) / sigma_x, 2));
        }
    }
};
