#pragma once

#include<fstream>
#include<iostream>
#include "Eigen/Dense"
#include "Eigen/src/Core/Matrix.h"

#include "field.hpp"
#include "plasma.hpp"
#include "constants.hpp"

// Define a bi-fluid plasma system in kx-space
class bi_system{
    private:
        void update_matrices(){
            // Longitudinal matrix
            longitudinal_matrix = Eigen::Matrix4cd::Zero();
            longitudinal_matrix(0,1) = -std::complex<double>(0, m_kx);

            longitudinal_matrix(1,0) = -std::complex<double>(0, omega_e/m_kx + cs_e*m_kx);
            longitudinal_matrix(1,2) = std::complex<double>(0, omega_e/m_kx);

            longitudinal_matrix(2,3) = -std::complex<double>(0, m_kx);

            longitudinal_matrix(3,1) = std::complex<double>(0, omega_i/m_kx);
            longitudinal_matrix(3,3) = -std::complex<double>(0, omega_i/m_kx + cs_i*m_kx);

            // Transverse matrix y
            transverse_matrix = Eigen::Matrix4cd::Zero();
            transverse_matrix(0,2) = std::complex<double>(m_electron.get_q()/m_electron.get_m(), 0);

            transverse_matrix(1,2) = std::complex<double>(m_ion.get_q()/m_ion.get_m(), 0);

            transverse_matrix(2,0) = -std::complex<double>(m_electron.get_q() * m_electron.get_n0(), 0)/consts::epsilon0;
            transverse_matrix(2,1) = -std::complex<double>(m_electron.get_q() * m_electron.get_n0(), 0)/consts::epsilon0;
            transverse_matrix(2,3) = std::complex<double>(0, consts::c*consts::c*m_kx);

            transverse_matrix(3,2) = -std::complex<double>(0, m_kx);
        }

        void update_vectors(){
            // Longitudinal vector
            longitudinal_vector = Eigen::Vector4cd::Zero();
            longitudinal_vector(0) = m_electron.get_n1();
            longitudinal_vector(1) = m_electron.get_U1()(0);
            longitudinal_vector(2) = m_ion.get_n1();
            longitudinal_vector(3) = m_ion.get_U1()(0);

            // Transverse vector y
            transverse_vector = Eigen::Vector4cd::Zero();
            transverse_vector(0) = m_electron.get_U1()(1);
            transverse_vector(1) = m_ion.get_U1()(1);
            transverse_vector(2) = m_field.get_E1();
            transverse_vector(3) = m_field.get_B1();
        }

        // Fonction qui implémente Runge-Kutta 4 pour un système d'équations
        Eigen::VectorXcd rk4_step(const Eigen::MatrixXcd& A, const Eigen::VectorXcd& y, double dt) {
            Eigen::VectorXcd k1 = A * y;
            Eigen::VectorXcd k2 = A * (y + 0.5 * dt * k1);
            Eigen::VectorXcd k3 = A * (y + 0.5 * dt * k2);
            Eigen::VectorXcd k4 = A * (y + dt * k3);
            return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        }

    public:
        double m_kx;
        field m_field;
        plasma m_ion, m_electron;

        double omega_e = m_electron.get_n0() * m_electron.get_q()*m_electron.get_q() / (m_electron.get_m()*consts::epsilon0), 
               omega_i = m_ion.get_n0() * m_ion.get_q()*m_ion.get_q() / (m_ion.get_m()*consts::epsilon0);
        double cs_e = consts::kB * m_electron.get_T() / m_electron.get_m(),
               cs_i = consts::kB * m_ion.get_T() / m_ion.get_m();

        Eigen::Vector4cd longitudinal_vector;
        Eigen::Vector4cd transverse_vector;
        
        Eigen::Matrix4cd longitudinal_matrix;
        Eigen::Matrix4cd transverse_matrix;
        
        // Setters and getters       
        double get_kx() const { return m_kx; }

        void initial_wavenumber(double kx){
            m_kx = kx;
        }

        void iteration_time(double t0, double tf, double dt, const std::string& filename){
            int steps = static_cast<int>((tf - t0) / dt);
            for (int i = 0; i <= steps; ++i) {
                double current_time = t0 + i * dt;

                // Update matrices and vectors
                update_matrices();
                update_vectors();

                // Perform RK4 step for longitudinal and transverse components
                longitudinal_vector = rk4_step(longitudinal_matrix, longitudinal_vector, dt);
                transverse_vector = rk4_step(transverse_matrix, transverse_vector, dt);

                // Update the state variables from the vectors
                m_electron.set_n1(longitudinal_vector(0));
                m_electron.set_U1(Eigen::Vector2cd(longitudinal_vector(1), transverse_vector(0)));

                m_ion.set_n1(longitudinal_vector(2));
                m_ion.set_U1(Eigen::Vector2cd(longitudinal_vector(3), transverse_vector(1)));

                m_field.set_E1(transverse_vector(2));
                m_field.set_B1(transverse_vector(3));

                // Save data to file
            }
        }
};