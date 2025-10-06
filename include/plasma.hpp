#pragma once

#include "Eigen/Dense"
#include "Eigen/src/Core/Matrix.h"
#include <vector>

/**
 * @brief Class representing a plasma species in Fourier (kx) space for 1D bi-plasma simulations
 * 
 * This class encapsulates both equilibrium properties and perturbations of a plasma species.
 * It stores velocity perturbations (U1), density perturbations (n1), and fundamental
 * plasma parameters such as equilibrium density, charge, mass, and temperature.
 * All perturbation quantities are represented as complex matrices in Fourier space.
 */
class plasma{ 
    public:
        /// Vector of velocity perturbation matrices U1 (complex-valued, Fourier space)
        /// Index 0: x-component, Index 1: y-component of velocity perturbations
        std::vector<Eigen::MatrixXcd> m_U1;
        
        /// Density perturbation matrix n1 (complex-valued, Fourier space)
        Eigen::MatrixXcd  m_n1;
        
        /// Equilibrium density (particles per unit volume)
        double m_n0;
        
        /// Electric charge of the plasma species (in elementary charge units)
        double m_q;
        
        /// Mass of the plasma species (in atomic mass units or kg)
        double m_m;
        
        /// Temperature of the plasma species (in energy units)
        double m_T;

        // Setters and getters
        
        /**
         * @brief Set velocity perturbation at specific matrix position and component
         * @param i Row index (wave vector component)
         * @param j Column index (time step)
         * @param component Velocity component (0 for x, 1 for y)
         * @param value Complex velocity perturbation value
         */
        void set_U1_at(int i, int j, int component, std::complex<double> value){ m_U1[component](i,j) = value; }
        
        /**
         * @brief Set density perturbation at specific matrix position
         * @param i Row index (wave vector component)
         * @param j Column index (time step)
         * @param value Complex density perturbation value
         */
        void set_n1_at(int i, int j, std::complex<double> value){ m_n1(i,j) = value; }
        
        /**
         * @brief Set equilibrium density
         * @param n0 Equilibrium density value
         */
        void set_n0(const double& n0){ m_n0 = n0; }
        
        /**
         * @brief Set electric charge
         * @param q Electric charge value
         */
        void set_q(const double& q){ m_q = q; }
        
        /**
         * @brief Set particle mass
         * @param m Mass value
         */
        void set_m(const double& m){ m_m = m; }
        
        /**
         * @brief Set temperature
         * @param T Temperature value
         */
        void set_T(const double& T){ m_T = T; }
       
        /**
         * @brief Get the complete velocity perturbation matrices
         * @return Copy of the U1 vector containing velocity perturbation matrices
         */
        std::vector<Eigen::MatrixXcd> get_U1() const { return m_U1; }
        
        /**
         * @brief Get the complete density perturbation matrix
         * @return Copy of the n1 matrix
         */
        Eigen::MatrixXcd get_n1() const { return m_n1; }
        
        /**
         * @brief Get equilibrium density
         * @return Equilibrium density value
         */
        double get_n0() const { return m_n0; }
        
        /**
         * @brief Get electric charge
         * @return Electric charge value
         */
        double get_q() const { return m_q; }
        
        /**
         * @brief Get particle mass
         * @return Mass value
         */
        double get_m() const { return m_m; }
        
        /**
         * @brief Get temperature
         * @return Temperature value
         */
        double get_T() const { return m_T; }

        /**
         * @brief Initialize equilibrium plasma parameters
         * @param n0 Equilibrium density
         * @param q Electric charge
         * @param m Particle mass
         * @param T Temperature
         * 
         * Sets all fundamental plasma species parameters in one call
         */
        void initial_constant(double n0, double q, double m, double T){
            m_n0 = n0; m_q = q, m_m = m, m_T = T;
        }
        
        /**
         * @brief Initialize perturbation quantities with given matrices
         * @param n1 Initial density perturbation matrix
         * @param U1 Initial velocity perturbation matrices (vector with x and y components)
         * 
         * Uses move semantics for efficient transfer of matrix data to avoid unnecessary copies
         */
        void initial_perturbation(Eigen::MatrixXcd n1, std::vector<Eigen::MatrixXcd> U1){
            // Move to avoid extra copies when caller can provide rvalues
            m_n1 = std::move(n1);
            m_U1 = std::move(U1);
        }

        /**
         * @brief Resize perturbation matrices and initialize to zero
         * @param Nkx Number of wave vector components (rows)
         * @param step Number of time steps (columns)
         * 
         * Allocates memory for density and velocity perturbation matrices with specified dimensions.
         * Creates 2 velocity component matrices (x and y) and sets all elements to zero.
         */
        void resize_perturbation(int Nkx, int step){
            m_n1.resize(Nkx, step);
            m_U1.resize(2);
            m_n1.setZero();
            for(int i=0; i<2; i++){
                m_U1[i].resize(Nkx, step);
                m_U1[i].setZero();
            }
        }

        /**
         * @brief Clear perturbation matrices and free memory
         * 
         * Resizes all perturbation matrices to 0x0 dimensions to release allocated memory.
         * Clears the velocity component vector to ensure complete memory cleanup.
         */
        void clear_perturbation() {
            m_n1.resize(0,0);  // Redimensionner à 0 pour libérer la mémoire
            for(auto& U : m_U1) {
                U.resize(0,0);  // Redimensionner chaque matrice à 0 pour libérer la mémoire
            }
            m_U1.clear(); // Vider le vecteur pour libérer la mémoire
        }
};