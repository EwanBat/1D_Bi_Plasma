#pragma once

#include "Eigen/Dense"
#include<complex>

class field{
    public:
        std::complex<double> m_E1, m_B1;

        // Setters and getters
        std::complex<double> set_E1(const std::complex<double> E1){ m_E1 = E1; return m_E1; }
        std::complex<double> set_B1(const std::complex<double> B1){ m_B1 = B1; return m_B1; }

        std::complex<double> get_E1() const { return m_E1; }
        std::complex<double> get_B1() const { return m_B1; }

        void initial_perturbation(std::complex<double> E1, std::complex<double> B1){
            m_E1 = E1;
            m_B1 = B1;
        }
};