#pragma once

#include <Eigen/Dense>

// Fonction qui implémente Runge-Kutta 4 pour un système d'équations
template<typename T, typename F>
T rungeKutta4(F f, const T& y0, double t0, double h) {
    T k1 = f(t0, y0);
    T k2 = f(t0 + h/2.0, y0 + (h/2.0)*k1);
    T k3 = f(t0 + h/2.0, y0 + (h/2.0)*k2);
    T k4 = f(t0 + h, y0 + h*k3);
    
    return y0 + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4);
}