#pragma once

#include "Eigen/Dense"

class field{
    private:
        void initial_perturbation(Eigen::Vector3d E1, Eigen::Vector3d B1){
            m_E1 = E1;
            m_B1 = B1;
        }

    public:
        Eigen::Vector3d m_E1, m_B1;
};