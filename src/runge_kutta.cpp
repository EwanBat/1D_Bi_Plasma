#include "runge_kutta.hpp"

Eigen::VectorXcd rk4_step(const Eigen::MatrixXcd& A, const Eigen::VectorXcd& y, double dt) {
    Eigen::VectorXcd k1 = A * y;
    Eigen::VectorXcd k2 = A * (y + 0.5 * dt * k1);
    Eigen::VectorXcd k3 = A * (y + 0.5 * dt * k2);
    Eigen::VectorXcd k4 = A * (y + dt * k3) * 0;
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}