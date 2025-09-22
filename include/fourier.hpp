#pragma once

#include "Eigen/Dense"
#include "Eigen/src/Core/Matrix.h"

void inverse_fourier_transform(Eigen::MatrixXcd&, Eigen::MatrixXd&, std::vector<double>&, std::vector<double>&, std::vector<double>&);