#pragma once

#include <vector>
#include <string>
#include "Eigen/Dense"
#include "Eigen/src/Core/Matrix.h"

void fourier_transform_initial(Eigen::MatrixXd&, Eigen::MatrixXcd&, std::vector<double>&, std::vector<double>&, std::vector<double>&);

// Stream the inverse Fourier transform directly to a CSV file to avoid allocating a full (Nx x Nt) matrix.
// Writes one line per x (Nx lines), with Nt comma-separated values per line.
void inverse_fourier_transform_to_csv(Eigen::MatrixXcd& input,
									  const std::vector<double>& t_grid,
									  const std::vector<double>& k_grid,
									  const std::vector<double>& x_grid,
									  const std::string& filename);