#pragma once

#include "Eigen/Dense"
#include "Eigen/src/Core/Matrix.h"
#include <complex>

// Define a plasma in kx-space 
class plasma{ 
    private:
        void initial_constant(double n0, double q, double m){
            m_n0 = n0; m_q = q, m_m = m;
        }
        void initial_pertubation(double n0, std::complex<double> n1, Eigen::Vector3d U1, double q){
            m_n1 = n1;
            m_U1 = U1;
        }

    public:
        Eigen::Vector3cd m_U1;
        std::complex<double>  m_n1;
        double m_n0, m_q, m_m;

};