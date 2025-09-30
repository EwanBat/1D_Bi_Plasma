#pragma once

#include "Eigen/Dense"
#include "Eigen/src/Core/Matrix.h"

class field{
    public:
        Eigen::MatrixXcd m_E1, m_B1;

        // Setters and getters
        void set_E1_at(int i, int j, std::complex<double> E1){ m_E1(i,j) = E1;}
        void set_B1_at(int i, int j, std::complex<double> B1){ m_B1(i,j) = B1;}

        Eigen::MatrixXcd get_E1() const { return m_E1; }
        Eigen::MatrixXcd get_B1() const { return m_B1; }

        void initial_perturbation(Eigen::MatrixXcd E1, Eigen::MatrixXcd B1){
            m_E1 = E1;
            m_B1 = B1;
        }

        void resize_perturbation(int Nkx, int step){
            m_E1.resize(Nkx, step);
            m_B1.resize(Nkx, step);
            m_E1.setZero();
            m_B1.setZero();
        }

        void clear_perturbation() {
            m_E1.resize(0,0);  // Redimensionner à 0 pour libérer la mémoire
            m_B1.resize(0,0);  // Redimensionner à 0 pour libérer la mémoire
        }
};