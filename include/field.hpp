#pragma once

#include "Eigen/Dense"
#include "Eigen/src/Core/Matrix.h"

class field{
    private:
        Eigen::MatrixXcd m_E1, m_B1;

    public:
        // Setters and getters
        Eigen::MatrixXcd set_E1(const Eigen::MatrixXcd E1){ m_E1 = E1; return m_E1; }
        Eigen::MatrixXcd set_B1(const Eigen::MatrixXcd B1){ m_B1 = B1; return m_B1; }

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
};