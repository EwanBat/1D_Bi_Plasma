#include <complex>
#include <vector>
#include <iostream>

#include "../include/fourier.hpp"
#include "../include/Eigen/Dense"
#include "../include/Eigen/src/Core/Matrix.h"

#include <complex>


// input  : lignes = t, colonnes = k
// output : lignes = t, colonnes = x
void inverse_fourier_transform(Eigen::MatrixXcd& input,
                               Eigen::MatrixXd& output,
                               std::vector<double>& t_grid,
                               std::vector<double>& k_grid,
                               std::vector<double>& x_grid) {
    int Nt = t_grid.size();   // nombre de temps
    int Nk = k_grid.size();   // nombre de modes k
    int Nx = x_grid.size();  // nombre de points dans l'espace

    output.resize(Nx, Nt);

    for (int step = 0; step < Nt; ++step) {
        for (int idx = 0; idx < Nx; ++idx) {
            // std::cout << "Step: " << step << ", x index: " << idx << std::endl;
            double x = x_grid[idx];
            std::complex<double> sum = 0.0;

            for (int idk = 0; idk < Nk; ++idk) {
                // std::cout << "  k index: " << idk << std::endl;
                double k = k_grid[idk]; // assuming uniform spacing in k
                sum += input(idk, step) * std::exp(std::complex<double>(0, k * x));
            }

            double sum_real = (sum / static_cast<double>(Nk)).real(); // Normalization factor
            output(idx, step) = sum_real; // Normalization factor
        }
    }
}

