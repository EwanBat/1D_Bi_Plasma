#pragma once

#include "Eigen/Dense"
#include "Eigen/src/Core/Matrix.h"
#include <complex>

// Define a plasma in kx-space 
class plasma{ 
    public:
        Eigen::Vector2cd m_U1;
        std::complex<double>  m_n1;
        double m_n0, m_q, m_m, m_T;

        // Setters and getters
        Eigen::Vector2cd& set_U1(const Eigen::Vector2cd& U1){ m_U1 = U1; return m_U1; }
        std::complex<double>& set_n1(const std::complex<double>& n1){ m_n1 = n1; return m_n1; }
        double& set_n0(const double& n0){ m_n0 = n0; return m_n0; }
        double& set_q(const double& q){ m_q = q; return m_q; }
        double& set_m(const double& m){ m_m = m; return m_m; }
        double& set_T(const double& T){ m_T = T; return m_T; }
       
        Eigen::Vector2cd get_U1() const { return m_U1; }
        std::complex<double> get_n1() const { return m_n1; }
        double get_n0() const { return m_n0; }
        double get_q() const { return m_q; }
        double get_m() const { return m_m; }
        double get_T() const { return m_T; }

        void initial_constant(double n0, double q, double m, double T){
            m_n0 = n0; m_q = q, m_m = m, m_T = T;
        }
        void initial_pertubation(double n0, std::complex<double> n1, Eigen::Vector2cd U1, double q){
            m_n1 = n1;
            m_U1 = U1;
        }
};