#pragma once

#include "Eigen/Dense"
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
    
    // Ajoutez des vecteurs temporaires comme membres pour éviter les réallocations
    Eigen::VectorXd dn_i1_dt, du_i1_dt, dn_e1_dt, du_e1_dt, dE_dx;

public:
    /**
     * @brief Constructor
     */
    PlasmaSystem(int nx, double dx_val, const PlasmaParams& p, double cfl = 0.5) 
        : Nx(nx), dx(dx_val), params(p), cfl_factor(cfl) {
        // Initialize state vectors
        n_i1.resize(Nx); n_i1.setZero();
        u_i1.resize(Nx); u_i1.setZero();
        n_e1.resize(Nx); n_e1.setZero();
        u_e1.resize(Nx); u_e1.setZero();
        E.resize(Nx); E.setZero();
        
        // Initialiser les vecteurs temporaires
        dn_i1_dt.resize(Nx);
        du_i1_dt.resize(Nx);
        dn_e1_dt.resize(Nx);
        du_e1_dt.resize(Nx);
        dE_dx.resize(Nx);
        
        // Initialize dt with CFL condition
        dt = compute_cfl_timestep();
    }
    
    /**
     * @brief Compute timestep based on CFL condition
     * CFL condition: dt <= cfl_factor * dx / v_max
     * where v_max is the maximum wave speed in the system
     */
    double compute_cfl_timestep() const {
        // Compute sound speeds for ions and electrons
        const double c_s_i = std::sqrt(params.gamma_i * params.P_i0 / (params.n_i0 * params.m_i));
        const double c_s_e = std::sqrt(params.gamma_e * params.P_e0 / (params.n_e0 * params.m_e));
        
        // Find maximum velocities
        const double u_i_max = u_i1.cwiseAbs().maxCoeff();
        const double u_e_max = u_e1.cwiseAbs().maxCoeff();
        
        // Maximum wave speeds (velocity + sound speed)
        const double v_max_i = u_i_max + c_s_i;
        const double v_max_e = u_e_max + c_s_e;
        const double v_max = std::max(v_max_i, v_max_e);
        
        // CFL condition
        const double dt_cfl = cfl_factor * dx / v_max;
        return dt_cfl;
    }
    
    /**
     * @brief Compute spatial derivative using upwind scheme
     * Upwind: uses backward difference if u > 0, forward if u < 0
     */
    double derivative_x_upwind(const Eigen::VectorXd& f, int i, double u) const {
        if (u > 0.0) {
            // Backward difference (upwind for positive velocity)
            if (i == 0) {
                return (f(i) - f(Nx-1)) / dx;  // Periodic boundary
            } else {
                return (f(i) - f(i-1)) / dx;
            }
        } else {
            // Forward difference (upwind for negative velocity)
            if (i == Nx - 1) {
                return (f(0) - f(i)) / dx;  // Periodic boundary
            } else {
                return (f(i+1) - f(i)) / dx;
            }
        }
    }
    
    /**
     * @brief Compute spatial derivative using centered finite differences
     * Used for pressure gradient terms (non-advective)
     */
    double derivative_x_centered(const Eigen::VectorXd& f, int i) const {
        if (i == 0) {
            return (f(1) - f(Nx-1)) / (2.0 * dx);  // Periodic boundary
        } else if (i == Nx - 1) {
            return (f(0) - f(Nx-2)) / (2.0 * dx);  // Periodic boundary
        } else {
            return (f(i+1) - f(i-1)) / (2.0 * dx);
        }
    }
    
    /**
     * @brief Compute pressure gradient term for species s
     * dP/dx term from equation 2
     */
    double pressure_gradient(const Eigen::VectorXd& n_s1, int i, 
                            double n_s0, double P_s0, double gamma_s) const {
        const double dn_dx = derivative_x_centered(n_s1, i);
        const double n_s = n_s0 + n_s1(i);
        
        const double term1 = gamma_s / n_s0 * dn_dx * (1.0 + gamma_s * n_s1(i) / n_s0);
        const double term2 = 2.0 * gamma_s / (n_s0 * n_s0) * n_s1(i) * dn_dx;
        
        return (P_s0 / n_s0) * (term1 - term2);
    }
    
    /**
     * @brief Compute time derivatives for all state variables
     * E field must be computed before calling this
     */
    void compute_derivatives(Eigen::VectorXd& dn_i1_dt, Eigen::VectorXd& du_i1_dt,
                            Eigen::VectorXd& dn_e1_dt, Eigen::VectorXd& du_e1_dt) {
        
        // For each spatial point, compute time derivatives
        for (int i = 0; i < Nx; ++i) {
            // Ion continuity equation (equation 1 for ions) - schéma upwind
            const double du_i1_dx = derivative_x_upwind(u_i1, i, params.n_i0 + n_i1(i));
            const double dn_i1_dx = derivative_x_upwind(n_i1, i, u_i1(i));
            dn_i1_dt(i) = -(params.n_i0 + n_i1(i)) * du_i1_dx - u_i1(i) * dn_i1_dx;
            
            // Ion momentum equation (equation 2 for ions)
            const double du_i1_dx_u = derivative_x_upwind(u_i1, i, u_i1(i));
            const double dP_i_dx = pressure_gradient(n_i1, i, params.n_i0, params.P_i0, params.gamma_i);
            du_i1_dt(i) = -u_i1(i) * du_i1_dx_u - dP_i_dx / params.m_i 
                         + params.q_i * E(i) / params.m_i;
            
            // Electron continuity equation (equation 1 for electrons) - schéma upwind
            const double du_e1_dx = derivative_x_upwind(u_e1, i, params.n_e0 + n_e1(i));
            const double dn_e1_dx = derivative_x_upwind(n_e1, i, u_e1(i));
            dn_e1_dt(i) = -(params.n_e0 + n_e1(i)) * du_e1_dx - u_e1(i) * dn_e1_dx;
            
            // Electron momentum equation (equation 2 for electrons)
            const double du_e1_dx_u = derivative_x_upwind(u_e1, i, u_e1(i));
            const double dP_e_dx = pressure_gradient(n_e1, i, params.n_e0, params.P_e0, params.gamma_e);
            du_e1_dt(i) = -u_e1(i) * du_e1_dx_u - dP_e_dx / params.m_e 
                         + params.q_e * E(i) / params.m_e;
        }
    }
    
    /**
     * @brief Advance system by one time step using explicit Euler method
     */
    void step_euler() {
        // Update dt based on current state (CFL condition)
        dt = compute_cfl_timestep();
        
        // Now compute time derivatives using current E field
        compute_derivatives(dn_i1_dt, du_i1_dt, dn_e1_dt, du_e1_dt);
        
        // Update state variables
        n_i1.noalias() += dt * dn_i1_dt;
        u_i1.noalias() += dt * du_i1_dt;
        n_e1.noalias() += dt * dn_e1_dt;
        u_e1.noalias() += dt * du_e1_dt;

        // Update E field using Poisson's equation (equation 3)
        for (int i = 0; i < Nx; ++i) {
            const double charge_density = params.q_i * n_i1(i) + params.q_e * n_e1(i);
            dE_dx(i) = charge_density / params.epsilon_0;
        }
        
        // Integrate dE/dx to get E using periodic boundary condition
        E(0) = E(Nx-1);  // Periodic boundary condition
        for (int i = 1; i < Nx; ++i) {
            E(i) = E(i-1) + dE_dx(i-1) * dx;
        }
    }
    
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
        for (int i = 0; i < Nx; ++i) {
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
    void append_time_data(double t) {
        // Écrire le temps
        file_t << t << "\n";
        
        // Écrire les données spatiales
        for (int i = 0; i < Nx; ++i) {
            file_n_i1 << n_i1(i);
            file_u_i1 << u_i1(i);
            file_n_e1 << n_e1(i);
            file_u_e1 << u_e1(i);
            file_E << E(i);
            
            if (i < Nx - 1) {
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
        std::vector<double> x_grid(Nx);
        for (int i = 0; i < Nx; ++i) {
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
