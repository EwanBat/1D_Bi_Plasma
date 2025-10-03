#pragma once

#include <fstream>
#include <iostream>
#include <ostream>
#include "Eigen/Dense"
#include "Eigen/src/Core/Matrix.h"

#include "field.hpp"
#include "plasma.hpp"
#include "constants.hpp"
#include "fourier.hpp"
#include "runge_kutta.hpp"

// Define a bi-fluid plasma system in kx-space
class bi_system{
    private:
        void calculate_eigenvalues(int idx){
            Eigen::ComplexEigenSolver<Eigen::Matrix4cd> solver_longitudinal(longitudinal_matrix);
            Eigen::ComplexEigenSolver<Eigen::Matrix4cd> solver_transverse(transverse_matrix);
            eigenvalues_longitudinal[idx][0] = -solver_longitudinal.eigenvalues()(0) / std::complex<double>(0,1);
            eigenvalues_longitudinal[idx][1] = -solver_longitudinal.eigenvalues()(1) / std::complex<double>(0,1);
            eigenvalues_longitudinal[idx][2] = -solver_longitudinal.eigenvalues()(2) / std::complex<double>(0,1);
            eigenvalues_longitudinal[idx][3] = -solver_longitudinal.eigenvalues()(3) / std::complex<double>(0,1);
            eigenvalues_transverse[idx][0] = -solver_transverse.eigenvalues()(0) / std::complex<double>(0,1);
            eigenvalues_transverse[idx][1] = -solver_transverse.eigenvalues()(1) / std::complex<double>(0,1);
            eigenvalues_transverse[idx][2] = -solver_transverse.eigenvalues()(2) / std::complex<double>(0,1);
            eigenvalues_transverse[idx][3] = -solver_transverse.eigenvalues()(3) / std::complex<double>(0,1);
        }

        void update_matrices(double kx){
            double ne0 = m_electron.get_n0(); double ni0 = m_ion.get_n0();
            
            // Longitudinal matrix
            longitudinal_matrix = Eigen::Matrix4cd::Zero();
            longitudinal_matrix(0,2) = -std::complex<double>(0, kx * ne0);

            longitudinal_matrix(1,3) = -std::complex<double>(0, kx * ni0);

            longitudinal_matrix(2,0) = -std::complex<double>(0, omega_e / (kx * ne0) + cs_e * kx / ne0);
            longitudinal_matrix(2,1) = std::complex<double>(0,omega_e / (kx * ne0));

            longitudinal_matrix(3,0) = std::complex<double>(0,omega_i / (kx * ni0));
            longitudinal_matrix(3,1) = -std::complex<double>(0, omega_i / (kx * ni0) + cs_i * kx / ni0);


            // Transverse matrix y
            transverse_matrix = Eigen::Matrix4cd::Zero();
            transverse_matrix(0,2) = std::complex<double>(m_electron.get_q()/m_electron.get_m(), 0);

            transverse_matrix(1,2) = std::complex<double>(m_ion.get_q()/m_ion.get_m(), 0);

            transverse_matrix(2,0) = -std::complex<double>(m_electron.get_q() * m_electron.get_n0(), 0)/consts::epsilon0;
            transverse_matrix(2,1) = -std::complex<double>(m_electron.get_q() * m_electron.get_n0(), 0)/consts::epsilon0;
            transverse_matrix(2,3) = std::complex<double>(0, consts::c*consts::c*kx);

            transverse_matrix(3,2) = -std::complex<double>(0, kx);
        }

        void update_vectors(int idx, int step){
            // Longitudinal vector
            longitudinal_vector = Eigen::Vector4cd::Zero();
            longitudinal_vector(0) = m_electron.get_n1()(idx,step);
            longitudinal_vector(1) = m_ion.get_n1()(idx,step);
            longitudinal_vector(2) = m_electron.get_U1()[0](idx,step);
            longitudinal_vector(3) = m_ion.get_U1()[0](idx,step);

            // Transverse vector y
            transverse_vector = Eigen::Vector4cd::Zero();
            transverse_vector(0) = m_electron.get_U1()[1](idx,step);
            transverse_vector(1) = m_ion.get_U1()[1](idx,step);
            transverse_vector(2) = m_field.get_E1()(idx,step);
            transverse_vector(3) = m_field.get_B1()(idx,step);
        }

        void save_eigevalues(const std::string& filename_longitudinal, const std::string& filename_transverse) {
            std::ofstream file_longitudinal(filename_longitudinal);
            std::ofstream file_transverse(filename_transverse);
            if (file_longitudinal.is_open() && file_transverse.is_open()) {
                for (size_t i = 0; i < eigenvalues_longitudinal.size(); ++i) {
                    for (size_t j = 0; j < eigenvalues_longitudinal[i].size(); ++j) {
                        file_longitudinal << eigenvalues_longitudinal[i][j];
                        if (j < eigenvalues_longitudinal[i].size() - 1) {
                            file_longitudinal << ","; // Separate values with commas
                        }
                    }
                    file_longitudinal << "\n"; // New line at the end of each row
                }
                for (size_t i = 0; i < eigenvalues_transverse.size(); ++i) {
                    for (size_t j = 0; j < eigenvalues_transverse[i].size(); ++j) {
                        file_transverse << eigenvalues_transverse[i][j];
                        if (j < eigenvalues_transverse[i].size() - 1) {
                            file_transverse << ","; // Separate values with commas
                        }
                    }
                    file_transverse << "\n"; // New line at the end of each row
                }
                file_longitudinal.close();
                file_transverse.close();
            } else {
                std::cerr << "Unable to open file " << filename_longitudinal << " or " << filename_transverse << std::endl;
            }
        }

        void save_vector_to_txt(std::vector<double>& vec, const std::string& filename) {
            std::ofstream file(filename);
            if (file.is_open()) {
                for (const auto& value : vec) {
                    file << value << "\n";
                }
                file.close();
                vec.clear(); // Utiliser clear() pour vider un std::vector
            } else {
                std::cerr << "Unable to open file " << filename << std::endl;
            }
        }

        void save_matrix_to_txt(Eigen::MatrixXd& matrix, const std::string& filename) {
            std::ofstream file(filename);
            if (file.is_open()) {
                for (int i = 0; i < matrix.rows(); ++i) {
                    for (int j = 0; j < matrix.cols(); ++j) {
                        file << matrix(i, j);
                        if (j < matrix.cols() - 1) {
                            file << ","; // Separate values with commas
                        }
                    }
                    file << "\n"; // New line at the end of each row
                }
                file.close();
                matrix.resize(0,0); // Ceci est OK pour les MatrixXd (taille dynamique)
            } else {
                std::cerr << "Unable to open file " << filename << std::endl;
            }
        }

    public:
        field m_field;
        plasma m_ion, m_electron;

        double omega_e, omega_i;
        double cs_e, cs_i;

        Eigen::Vector4cd longitudinal_vector;
        Eigen::Vector4cd transverse_vector;
        
        Eigen::Matrix4cd longitudinal_matrix;
        Eigen::Matrix4cd transverse_matrix;

        std::vector<std::vector<std::complex<double>>> eigenvalues_longitudinal;
        std::vector<std::vector<std::complex<double>>> eigenvalues_transverse;
        
        void setup_system(int Nkx, int Tstep){
            omega_e = m_electron.get_n0() * m_electron.get_q()*m_electron.get_q() / (m_electron.get_m()*consts::epsilon0);
            omega_i = m_ion.get_n0() * m_ion.get_q()*m_ion.get_q() / (m_ion.get_m()*consts::epsilon0);
            cs_e = consts::kB * m_electron.get_T() / m_electron.get_m();
            cs_i = consts::kB * m_ion.get_T() / m_ion.get_m();
            
            m_electron.resize_perturbation(Nkx, Tstep);
            m_ion.resize_perturbation(Nkx, Tstep);
            m_field.resize_perturbation(Nkx, Tstep);
            eigenvalues_longitudinal.resize(Nkx, std::vector<std::complex<double>>(4, 0));
            eigenvalues_transverse.resize(Nkx, std::vector<std::complex<double>>(4, 0));
        }

        void print_parameter(){
            std::cout << "Plasma frequency of electron: " << std::sqrt(omega_e) << " s-1" << std::endl;
            std::cout << "Plasma frequency of ion: " << std::sqrt(omega_i) << " s-1" << std::endl;
            std::cout << "Sound celerity of electron: " << std::sqrt(cs_e) << " m.s-1" << std::endl;
            std::cout << "Sound celerity of ion: " << std::sqrt(cs_i) << " m.s-1" << std::endl;
        }

        void iteration(double t0, double tf, double dt, double kxi, double kxf, double dkx){
            const int Tstep = ((tf - t0) / dt) + 1;
            const int Nkx = ((kxf - kxi) / dkx) + 1;

            Eigen::Vector4cd result_rk4;
            for (int idx = 0; idx < Nkx; idx++) {
                double kx = kxi + idx * dkx;
                update_matrices(kx);
                calculate_eigenvalues(idx);
                for (int step = 0; step < Tstep-1; step++) {
                    update_vectors(idx, step);

                    // Update longitudinal perturbations
                    result_rk4 = rk4_step(longitudinal_matrix, longitudinal_vector, dt);
                    m_electron.set_n1_at(idx, step+1, result_rk4(0));
                    m_ion.set_n1_at(idx, step+1, result_rk4(1));
                    m_electron.set_U1_at(idx, step+1, 0, result_rk4(2));
                    m_ion.set_U1_at(idx, step+1, 0, result_rk4(3));

                    // Update transverse perturbations
                    result_rk4 = rk4_step(transverse_matrix, transverse_vector, dt);
                    m_electron.set_U1_at(idx, step+1, 1, result_rk4(0));
                    m_ion.set_U1_at(idx, step+1, 1, result_rk4(1));
                    m_field.set_E1_at(idx, step + 1, result_rk4(2));
                    m_field.set_B1_at(idx, step + 1, result_rk4(3));
                }
            }
        }

        void save_datas(std::vector<double>& t_grid, std::vector<double>& kx_grid, std::vector<double>& x_grid){
            int Nt = t_grid.size(), Nx = x_grid.size();   
        
            // Eigen::MatrixXd n1e_real, n1i_real, E_real, B_real;
            // std::vector<Eigen::MatrixXd> U1e_real(2), U1i_real(2);
            // U1e_real[0].resize(Nx, Nt); U1e_real[0].setZero(); U1e_real[1].resize(Nx, Nt); U1e_real[1].setZero();
            // U1i_real[0].resize(Nx, Nt); U1i_real[0].setZero(); U1i_real[1].resize(Nx, Nt); U1i_real[1].setZero();
            // inverse_fourier_transform(m_electron.m_n1, n1e_real, t_grid, kx_grid, x_grid);
            // inverse_fourier_transform(m_ion.m_n1, n1i_real, t_grid, kx_grid, x_grid);
            // inverse_fourier_transform(m_electron.m_U1[0], U1e_real[0], t_grid, kx_grid, x_grid);
            // inverse_fourier_transform(m_electron.m_U1[1], U1e_real[1], t_grid, kx_grid, x_grid);
            // inverse_fourier_transform(m_ion.m_U1[0], U1i_real[0], t_grid, kx_grid, x_grid);
            // inverse_fourier_transform(m_ion.m_U1[1], U1i_real[1], t_grid, kx_grid, x_grid);
            // inverse_fourier_transform(m_field.m_E1, E_real, t_grid, kx_grid, x_grid);
            // inverse_fourier_transform(m_field.m_B1, B_real, t_grid, kx_grid, x_grid);

            save_vector_to_txt(t_grid, "../data/t_grid.txt");
            save_vector_to_txt(kx_grid, "../data/kx_grid.txt");
            save_vector_to_txt(x_grid, "../data/x_grid.txt");

            // save_matrix_to_txt(n1e_real, "../data/n1e_real.txt");
            // save_matrix_to_txt(n1i_real, "../data/n1i_real.txt");
            // save_matrix_to_txt(U1e_real[0], "../data/U1e_real_x.txt");
            // save_matrix_to_txt(U1e_real[1], "../data/U1e_real_y.txt");
            // save_matrix_to_txt(U1i_real[0], "../data/U1i_real_x.txt");
            // save_matrix_to_txt(U1i_real[1], "../data/U1i_real_y.txt");
            // save_matrix_to_txt(E_real, "../data/E_real.txt");
            // save_matrix_to_txt(B_real, "../data/B_real.txt");

            save_eigevalues("../data/eigenvalues_long.txt", "../data/eigenvalues_trans.txt");
        }

        void clear_system() {
            m_electron.clear_perturbation();
            m_ion.clear_perturbation();
            m_field.clear_perturbation();
            // Les matrices de taille fixe 4x4 ne peuvent pas être redimensionnées
            longitudinal_matrix.setZero();
            transverse_matrix.setZero();
            eigenvalues_longitudinal.clear();
            eigenvalues_transverse.clear();
        }        
};