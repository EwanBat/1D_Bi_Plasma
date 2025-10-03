#pragma once

#include "Eigen/Dense"
#include "Eigen/src/Core/Matrix.h"

Eigen::VectorXcd rk4_step(const Eigen::MatrixXcd& A, const Eigen::VectorXcd& y, double dt);