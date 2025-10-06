#pragma once

#include "Eigen/Dense"
#include "Eigen/src/Core/Matrix.h"

/**
 * @brief Class representing electromagnetic field perturbations in a 1D bi-plasma system
 * 
 * This class stores and manages electric and magnetic field perturbations E1 and B1
 * as complex matrices using Eigen library. The fields are represented in Fourier space
 * with dimensions corresponding to wave vector components (kx) and time steps.
 */
class field{
    public:
        /// Electric field perturbation matrix E1 (complex-valued, Fourier space)
        Eigen::MatrixXcd m_E1;
        
        /// Magnetic field perturbation matrix B1 (complex-valued, Fourier space) 
        Eigen::MatrixXcd m_B1;

        // Setters and getters
        
        /**
         * @brief Set electric field perturbation at specific matrix position
         * @param i Row index (wave vector component)
         * @param j Column index (time step)
         * @param E1 Complex electric field value
         */
        void set_E1_at(int i, int j, std::complex<double> E1){ m_E1(i,j) = E1;}
        
        /**
         * @brief Set magnetic field perturbation at specific matrix position
         * @param i Row index (wave vector component)
         * @param j Column index (time step)
         * @param B1 Complex magnetic field value
         */
        void set_B1_at(int i, int j, std::complex<double> B1){ m_B1(i,j) = B1;}

        /**
         * @brief Get the complete electric field perturbation matrix
         * @return Copy of the E1 matrix
         */
        Eigen::MatrixXcd get_E1() const { return m_E1; }
        
        /**
         * @brief Get the complete magnetic field perturbation matrix
         * @return Copy of the B1 matrix
         */
        Eigen::MatrixXcd get_B1() const { return m_B1; }

        /**
         * @brief Initialize field perturbations with given matrices
         * @param E1 Initial electric field perturbation matrix
         * @param B1 Initial magnetic field perturbation matrix
         * 
         * Uses move semantics for efficient transfer of matrix data
         */
        void initial_perturbation(Eigen::MatrixXcd E1, Eigen::MatrixXcd B1){
            m_E1 = std::move(E1);
            m_B1 = std::move(B1);
        }

        /**
         * @brief Resize field perturbation matrices and initialize to zero
         * @param Nkx Number of wave vector components (rows)
         * @param step Number of time steps (columns)
         * 
         * Allocates memory for the specified dimensions and sets all elements to zero
         */
        void resize_perturbation(int Nkx, int step){
            m_E1.resize(Nkx, step);
            m_B1.resize(Nkx, step);
            m_E1.setZero();
            m_B1.setZero();
        }

        /**
         * @brief Clear field perturbation matrices and free memory
         * 
         * Resizes matrices to 0x0 dimensions to release allocated memory
         */
        void clear_perturbation() {
            m_E1.resize(0,0);  // Redimensionner à 0 pour libérer la mémoire
            m_B1.resize(0,0);  // Redimensionner à 0 pour libérer la mémoire
        }
};