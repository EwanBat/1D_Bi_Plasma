#pragma once

#include "Eigen/Dense"
#include "Eigen/src/Core/Matrix.h"
#include <vector>

// Define a plasma in kx-space 
class plasma{ 
    public:
        std::vector<Eigen::MatrixXcd> m_U1;
        Eigen::MatrixXcd  m_n1;
        double m_n0, m_q, m_m, m_T;

        // Setters and getters
        std::vector<Eigen::MatrixXcd>& set_U1(const std::vector<Eigen::MatrixXcd>& U1){ m_U1 = U1; return m_U1; }
        Eigen::MatrixXcd& set_n1(const Eigen::MatrixXcd& n1){ m_n1 = n1; return m_n1; }

        double& set_n0(const double& n0){ m_n0 = n0; return m_n0; }
        double& set_q(const double& q){ m_q = q; return m_q; }
        double& set_m(const double& m){ m_m = m; return m_m; }
        double& set_T(const double& T){ m_T = T; return m_T; }
       
        std::vector<Eigen::MatrixXcd> get_U1() const { return m_U1; }
        Eigen::MatrixXcd get_n1() const { return m_n1; }
        double get_n0() const { return m_n0; }
        double get_q() const { return m_q; }
        double get_m() const { return m_m; }
        double get_T() const { return m_T; }

        void initial_constant(double n0, double q, double m, double T){
            m_n0 = n0; m_q = q, m_m = m, m_T = T;
        }
        void initial_pertubation(Eigen::MatrixXcd n1, std::vector<Eigen::MatrixXcd> U1){
            m_n1 = n1;
            m_U1 = U1;
        }

        void resize_perturbation(int Nkx, int step){
            m_n1.resize(Nkx, step);
            m_U1.resize(2);
            m_n1.setZero();
            for(int i=0; i<2; i++){
                m_U1[i].resize(Nkx, step);
                m_U1[i].setZero();
            }
        }

        void clear_perturbation() {
            m_n1.resize(0,0);  // Redimensionner à 0 pour libérer la mémoire
            for(auto& U : m_U1) {
                U.resize(0,0);  // Redimensionner chaque matrice à 0 pour libérer la mémoire
            }
            m_U1.clear(); // Vider le vecteur pour libérer la mémoire
        }
};