#pragma once

#include "Eigen/Dense"
#include "Eigen/src/Core/Matrix.h"
#include "constants.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>  // Pour std::setprecision

/**
 * @brief Class representing the state of a 1D fluid plasma system
 * 
 * Solves the coupled equations for ion and electron density/velocity perturbations
 * and electric field using finite differences with mirror boundary conditions.
 * 
 * Equations:
 * 1) dn_s1/dt + (n_s0 + n_s1) du_s1/dx + u_s1 dn_s1/dx = 0
 * 2) du_s1/dt + u_s1 du_s1/dx + (1/(n_s*m_s)) * dP_s/dx = q_s E / m_s
 * 3) dE/dx = 1/epsilon_0 (q_i n_i1 + q_e n_e1)
 * where s = i (ions) or e (electrons)
 */
class PlasmaSystem {
public:
    // State variables (all are real-valued, spatial grid)
    Eigen::VectorXd n_i1;      // Ion density perturbation [m^-3]
    Eigen::VectorXd u_i1;      // Ion velocity perturbation [m/s]
    Eigen::VectorXd n_e1;      // Electron density perturbation [m^-3]
    Eigen::VectorXd u_e1;      // Electron velocity perturbation [m/s]
    Eigen::VectorXd E;         // Electric field [V/m]
    
    // Grid parameters
    int Nx;                    // Number of spatial points
    double dx;                 // Spatial step [m]
    double dt;                 // Time step [s] (computed via CFL)
    double cfl_factor;         // CFL safety factor (typically 0.5)
    
    // Physical parameters
    PlasmaParams params;
    
private:
    // Membres pour les fichiers ouverts
    std::ofstream file_t, file_n_i1, file_u_i1, file_n_e1, file_u_e1, file_E;
    
    // Vecteurs temporaires pour éviter les réallocations
    Eigen::VectorXd Phi, Phi_star;        // Electric potential [V]
    Eigen::VectorXd E_star;        // Electric field at star step [V/m]

    // Temporary vectors for calculations
    Eigen::VectorXd n_iR, n_iL; // Right and left states for ions density
    Eigen::VectorXd u_iR, u_iL; // Right and left states for ions velocity
    Eigen::VectorXd n_eR, n_eL; // Right and left states for electrons density
    Eigen::VectorXd u_eR, u_eL; // Right and left states for electrons velocity
    Eigen::VectorXd n_e_star, n_i_star; // Star states for ions and electrons density
    Eigen::VectorXd u_e_star, u_i_star; // Star states for ions and electrons velocity

    // Fluxes
    Eigen::VectorXd F_e_n, F_i_n; // Density fluxes
    Eigen::VectorXd F_i_u, F_e_u; // Velocity fluxes

    // Minmod
    Eigen::VectorXd minmod_i_n, minmod_e_n; // Minmod slopes for densities
    Eigen::VectorXd minmod_i_u, minmod_e_u; // Minmod slopes for velocities

public:
    /**
     * @brief Constructor
     */
    PlasmaSystem(int nx, double dx_val, const PlasmaParams& p, const double cfl = 0.5) 
        : Nx(nx), dx(dx_val), params(p), cfl_factor(cfl) {
        // Initialize state vectors
        n_i1.resize(Nx+1); n_i1.setZero();
        u_i1.resize(Nx+1); u_i1.setZero();
        n_e1.resize(Nx+1); n_e1.setZero();
        u_e1.resize(Nx+1); u_e1.setZero();
        E.resize(Nx+1); E.setZero();
        
        // Initialiser les vecteurs temporaires
        Phi.resize(Nx+1); Phi_star.resize(Nx+1);
        E_star.resize(Nx+1);

        n_iR.resize(Nx+1); n_iL.resize(Nx+1);
        u_iR.resize(Nx+1); u_iL.resize(Nx+1);
        n_eR.resize(Nx+1); n_eL.resize(Nx+1);
        u_eR.resize(Nx+1); u_eL.resize(Nx+1);
        n_e_star.resize(Nx+1); n_i_star.resize(Nx+1);
        u_e_star.resize(Nx+1); u_i_star.resize(Nx+1);

        F_e_n.resize(Nx+1); F_i_n.resize(Nx+1);
        F_i_u.resize(Nx+1); F_e_u.resize(Nx+1);

        minmod_i_n.resize(Nx+1); minmod_e_n.resize(Nx+1);
        minmod_i_u.resize(Nx+1); minmod_e_u.resize(Nx+1);

        // Initialize dt with CFL condition
        compute_time_step();
    }
    
    void compute_time_step() {
        double max_speed = 0.0;
        for (int i = 0; i < Nx+1; ++i) {
            double speed_i = std::abs(u_i1(i)) + params.cs_i;
            double speed_e = std::abs(u_e1(i)) + params.cs_e;
            max_speed = std::max({max_speed, speed_i, speed_e});
        }
        dt = cfl_factor * dx / max_speed;
    }

    void calc_potential(Eigen::VectorXd& n_i, Eigen::VectorXd& n_e, Eigen::VectorXd& Phi_out);

    void calc_electric_field(Eigen::VectorXd& Phi_in, Eigen::VectorXd& E_out);
    
    void minmod(Eigen::VectorXd& n_i, Eigen::VectorXd& n_e, 
                Eigen::VectorXd& u_i, Eigen::VectorXd& u_e);

    void MUSCL_reconstruction(int i, Eigen::VectorXd& n_i, Eigen::VectorXd& n_e,
                             Eigen::VectorXd& u_i, Eigen::VectorXd& u_e);
        
    void calc_fluxes(int i, Eigen::VectorXd& n_i, Eigen::VectorXd& n_e,
                     Eigen::VectorXd& u_i, Eigen::VectorXd& u_e);

    void update_star_densities(int i);

    double calc_pressure_gradient_i(int i, Eigen::VectorXd& n_i);
    
    double calc_pressure_gradient_e(int i, Eigen::VectorXd& n_e);

    void update_star_velocities(int i, double grad_P_i, double grad_P_e);

    void step_euler();

public:
    /**
     * @brief Initialize time evolution output files
     */
    void init_time_files(const std::vector<double>& x_grid, const std::string& prefix = "../data/") {
        // Sauvegarder la grille spatiale
        std::ofstream file_x(prefix + "x_grid.txt");
        if (!file_x.is_open()) {
            std::cerr << "Erreur: Impossible d'ouvrir " << prefix << "x_grid.txt" << std::endl;
            return;
        }
        for (int i = 0; i < Nx+1; ++i) {
            file_x << x_grid[i] << "\n";
        }
        file_x.close();
        
        // Ouvrir les fichiers et les garder ouverts
        file_t.open(prefix + "time.txt");
        file_n_i1.open(prefix + "n_i1_time.txt");
        file_u_i1.open(prefix + "u_i1_time.txt");
        file_n_e1.open(prefix + "n_e1_time.txt");
        file_u_e1.open(prefix + "u_e1_time.txt");
        file_E.open(prefix + "E_time.txt");
        
        if (!file_t.is_open() || !file_n_i1.is_open() || !file_u_i1.is_open() || 
            !file_n_e1.is_open() || !file_u_e1.is_open() || !file_E.is_open()) {
            std::cerr << "Erreur: Impossible d'ouvrir les fichiers de sortie" << std::endl;
            return;
        }
        
        std::cout << "Fichiers temporels initialisés dans " << prefix << std::endl;
    }
    
    /**
     * @brief Append current state to time evolution files
     */
    void append_time_data(double& t) {
        // Écrire le temps
        file_t << t << "\n";
        
        // Écrire les données spatiales
        for (int i = 0; i < Nx+1; ++i) {
            file_n_i1 << n_i1(i);
            file_u_i1 << u_i1(i);
            file_n_e1 << n_e1(i);
            file_u_e1 << u_e1(i);
            file_E << E(i);
            
            if (i < Nx) {
                file_n_i1 << " ";
                file_u_i1 << " ";
                file_n_e1 << " ";
                file_u_e1 << " ";
                file_E << " ";
            }
        }
        
        file_n_i1 << "\n";
        file_u_i1 << "\n";
        file_n_e1 << "\n";
        file_u_e1 << "\n";
        file_E << "\n";
        
        // Flush périodique (par exemple tous les 100 pas)
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
    * @brief Run time evolution for total time tf and save the datas at every step
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
        
        while (t < tf) {
            step_euler();
            if (t == 0.0){
                std::cout << "dt initial: " << dt << " s" << std::endl;
            }
            t += dt;
            step++;
            append_time_data(t);
            
            std::cout << "\rProgress: " << std::fixed << std::setprecision(2) 
                      << (t/tf)*100.0 << "% | Step: " << step 
                      << std::flush;
        }
        
        close_time_files();
        
        std::cout << std::endl << "Datas saved in data/ | " << step << " steps" << std::endl;
        std::cout << "=============================" << std::endl;
    }
    
    /**
     * @brief Set initial Gaussian perturbation
     */
    void set_gaussian_perturbation(const std::vector<double>& x_grid, 
                                   double x0, double sigma_x, double amplitude,
                                   double E_amplitude) {
        for (int i = 0; i < Nx; ++i) {
            const double x = x_grid[i];
            const double gauss = amplitude * std::exp(-0.5 * std::pow((x - x0) / sigma_x, 2));
            
            E(i) = E_amplitude * std::exp(-0.5 * std::pow((x - x0) / sigma_x, 2));
        }
    }
};
