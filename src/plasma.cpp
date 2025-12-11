#include "../include/plasma.hpp"

void PlasmaSystem::calc_potential(Eigen::VectorXd& n_i, Eigen::VectorXd& n_e, Eigen::VectorXd& Phi_out) {
    int M = Nx + 1;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(M, M);
    Eigen::VectorXd rho = (params.q_i * n_i + params.q_e * n_e) / params.epsilon_0;
    Eigen::VectorXd b = - rho * dx * dx;

    // Neumann mirror BC at i=0: Phi0 - Phi1 = 0  -> row: [1, -1, 0...], rhs=0
    A(0,0) = 1.0; A(0,1) = -1.0;
    b(0) = 0.0;

    // interior rows i=1..M-2
    for (int i = 1; i <= M-2; ++i) {
        A(i, i-1) = 1.0;
        A(i, i)   = -2.0;
        A(i, i+1) = 1.0;
        // b(i) already set as -rho*dx^2
    }

    // Neumann at i=M-1 (i = Nx): Phi_N - Phi_{N-1} = 0
    A(M-1, M-2) = -1.0; A(M-1, M-1) = 1.0;
    b(M-1) = 0.0;

    Phi_out = A.colPivHouseholderQr().solve(b);
}

void PlasmaSystem::calc_electric_field(Eigen::VectorXd& Phi_in, Eigen::VectorXd& E_out) {
    for (int i = 1; i < Nx; ++i) {
        E_out(i) = - (Phi_in(i+1) - Phi_in(i-1)) / (2.0 * dx);
    }
    // mirror BC
    E_out(0) = E_out(1);
    E_out(Nx) = E_out(Nx-1);
}

void PlasmaSystem::minmod(Eigen::VectorXd& n_i, Eigen::VectorXd& n_e, 
                          Eigen::VectorXd& u_i, Eigen::VectorXd& u_e){
    const double inv_dx = 1.0 / dx;  // Calcul une seule fois
    double a, b;
    for (int i = 1; i < Nx; ++i){
        a = (n_i(i) - n_i(i - 1)) * inv_dx;
        b = (n_i(i + 1) - n_i(i)) * inv_dx;
        minmod_i_n(i) = 0.5 * (std::copysign(1.0, a) + std::copysign(1.0, b)) * std::min(std::abs(a), std::abs(b));

        a = (n_e(i) - n_e(i - 1)) * inv_dx;
        b = (n_e(i + 1) - n_e(i)) * inv_dx;
        minmod_e_n(i) = 0.5 * (std::copysign(1.0, a) + std::copysign(1.0, b)) * std::min(std::abs(a), std::abs(b));

        a = (u_i(i) - u_i(i - 1)) * inv_dx;
        b = (u_i(i + 1) - u_i(i)) * inv_dx;
        minmod_i_u(i) = 0.5 * (std::copysign(1.0, a) + std::copysign(1.0, b)) * std::min(std::abs(a), std::abs(b));

        a = (u_e(i) - u_e(i - 1)) * inv_dx;
        b = (u_e(i + 1) - u_e(i)) * inv_dx;
        minmod_e_u(i) = 0.5 * (std::copysign(1.0, a) + std::copysign(1.0, b)) * std::min(std::abs(a), std::abs(b));
    }
}

void PlasmaSystem::MUSCL_reconstruction(int i, Eigen::VectorXd& n_i, Eigen::VectorXd& n_e,
                                        Eigen::VectorXd& u_i, Eigen::VectorXd& u_e) {
    
    // Ions density - Slope limiter (minmod)
    n_iL(i) = n_i(i) + 0.5 * minmod_i_n(i) * dx;
    n_iR(i) = n_i(i+1) - 0.5 * minmod_i_n(i+1) * dx;

    // Electrons density - Slope limiter (minmod)
    n_eL(i) = n_e(i) + 0.5 * minmod_e_n(i) * dx;
    n_eR(i) = n_e(i+1) - 0.5 * minmod_e_n(i+1) * dx;

    // Ions velocity - Slope limiter (minmod)
    u_iL(i) = u_i(i) + 0.5 * minmod_i_u(i) * dx;
    u_iR(i) = u_i(i+1) - 0.5 * minmod_i_u(i+1) * dx;

    // Electrons velocity - Slope limiter (minmod)
    u_eL(i) = u_e(i) + 0.5 * minmod_e_u(i) * dx;
    u_eR(i) = u_e(i+1) - 0.5 * minmod_e_u(i+1) * dx;
}

void PlasmaSystem::calc_fluxes(int i, Eigen::VectorXd& n_i, Eigen::VectorXd& n_e,
                                        Eigen::VectorXd& u_i, Eigen::VectorXd& u_e) { 
    double alpha_i= std::max(std::abs(u_iL(i)) + params.cs_i, std::abs(u_iR(i)) + params.cs_i);
    double alpha_e = std::max(std::abs(u_eL(i)) + params.cs_e, std::abs(u_eR(i)) + params.cs_e); // Maximum wave speeds for ions and electrons
    
    // Ion density fluxes
    double F_iL_n = (params.n_i0 + n_iL(i)) * u_iL(i);
    double F_iR_n = (params.n_i0 + n_iR(i)) * u_iR(i);
    F_i_n(i) = 0.5 * (F_iL_n + F_iR_n) - 0.5 * alpha_i * (n_iR(i) - n_iL(i));

    // Electron density fluxes
    double F_eL_n = (params.n_e0 + n_eL(i)) * u_eL(i);
    double F_eR_n = (params.n_e0 + n_eR(i)) * u_eR(i);
    F_e_n(i) = 0.5 * (F_eL_n + F_eR_n) - 0.5 * alpha_e * (n_eR(i) - n_eL(i));

    // Ion velocity fluxes
    double F_iL_u = 0.5 * u_iL(i) * u_iL(i);
    double F_iR_u = 0.5 * u_iR(i) * u_iR(i);
    F_i_u(i) = 0.5 * (F_iL_u + F_iR_u) - 0.5 * alpha_i * (u_iR(i) - u_iL(i));

    // Electron velocity fluxes
    double F_eL_u = 0.5 * u_eL(i) * u_eL(i);
    double F_eR_u = 0.5 * u_eR(i) * u_eR(i);
    F_e_u(i) = 0.5 * (F_eL_u + F_eR_u) - 0.5 * alpha_e * (u_eR(i) - u_eL(i));
}

double PlasmaSystem::calc_pressure_gradient_i(int i, Eigen::VectorXd& n_i) {
    const double dn_i1_dx = (n_i(i + 1) - n_i(i - 1)) / (2.0 * dx);
    const double n_ratio = n_i(i) / params.n_i0;
    
    // Factorisation : P_i0/n_i0 * gamma_i/n_i0 * dn/dx * (1 + gamma_i*n/n_i0 - 2*gamma_i*n/n_i0)
    const double factor = (params.P_i0 * params.gamma_i) / (params.n_i0 * params.n_i0);
    return factor * dn_i1_dx * (1.0 - params.gamma_i * n_ratio);
}

double PlasmaSystem::calc_pressure_gradient_e(int i, Eigen::VectorXd& n_e) {
    const double dn_e1_dx = (n_e(i + 1) - n_e(i - 1)) / (2.0 * dx);
    const double n_ratio = n_e(i) / params.n_e0;
    
    // Factorisation identique aux ions
    const double factor = (params.P_e0 * params.gamma_e) / (params.n_e0 * params.n_e0);
    return factor * dn_e1_dx * (1.0 - params.gamma_e * n_ratio);
}

void PlasmaSystem::step_euler() {
    // Factoriser les constantes utilisées dans les boucles
    const double dt_dx = dt / dx;
    const double dt_mi = dt / params.m_i;
    const double dt_me = dt / params.m_e;
    
    // === ÉTAPE 1 : Prédiction (Euler forward) ===
    minmod(n_i1, n_e1, u_i1, u_e1);
    
    // Calcul des flux
    for (int i = 0; i < Nx; ++i) {
        MUSCL_reconstruction(i, n_i1, n_e1, u_i1, u_e1);
        calc_fluxes(i, n_i1, n_e1, u_i1, u_e1);
    }
    
    // Mise à jour vers état star
    for (int i = 1; i < Nx; ++i) {
        n_i_star(i) = n_i1(i) - dt_dx * (F_i_n(i) - F_i_n(i-1));
        n_e_star(i) = n_e1(i) - dt_dx * (F_e_n(i) - F_e_n(i-1));
        
        const double grad_P_i = calc_pressure_gradient_i(i, n_i1);
        const double grad_P_e = calc_pressure_gradient_e(i, n_e1);
        
        u_i_star(i) = u_i1(i) - dt_dx * (F_i_u(i) - F_i_u(i-1)) 
                      + dt_mi * (params.q_i * E(i) - grad_P_i);
        u_e_star(i) = u_e1(i) - dt_dx * (F_e_u(i) - F_e_u(i-1)) 
                      + dt_me * (params.q_e * E(i) - grad_P_e);
    }
    
    // Conditions aux limites miroir
    n_i_star(0) = n_i_star(1); n_i_star(Nx) = n_i_star(Nx - 1);
    n_e_star(0) = n_e_star(1); n_e_star(Nx) = n_e_star(Nx - 1);
    u_i_star(0) = u_i_star(1); u_i_star(Nx) = u_i_star(Nx - 1);
    u_e_star(0) = u_e_star(1); u_e_star(Nx) = u_e_star(Nx - 1);

    // === ÉTAPE 2 : Correction avec état star ===
    calc_potential(n_i_star, n_e_star, Phi_star);
    calc_electric_field(Phi_star, E_star);
    minmod(n_i_star, n_e_star, u_i_star, u_e_star);
    
    // Calcul des flux avec état star
    for (int i = 0; i < Nx; ++i) {
        MUSCL_reconstruction(i, n_i_star, n_e_star, u_i_star, u_e_star);
        calc_fluxes(i, n_i_star, n_e_star, u_i_star, u_e_star);
    }
    
    // Moyenne de Heun - Utilisation de .noalias() pour éviter temporaires
    for (int i = 1; i < Nx; ++i) {
        const double n_i_new = n_i1(i) - dt_dx * (F_i_n(i) - F_i_n(i-1));
        const double n_e_new = n_e1(i) - dt_dx * (F_e_n(i) - F_e_n(i-1));
        
        const double grad_P_i = calc_pressure_gradient_i(i, n_i_star);
        const double grad_P_e = calc_pressure_gradient_e(i, n_e_star);
        
        const double u_i_new = u_i1(i) - dt_dx * (F_i_u(i) - F_i_u(i-1)) 
                       + dt_mi * (params.q_i * E_star(i) - grad_P_i);
        const double u_e_new = u_e1(i) - dt_dx * (F_e_u(i) - F_e_u(i-1)) 
                       + dt_me * (params.q_e * E_star(i) - grad_P_e);
        
        // Moyenne sans créer de temporaires
        n_i1(i) = 0.5 * (n_i1(i) + n_i_new);
        n_e1(i) = 0.5 * (n_e1(i) + n_e_new);
        u_i1(i) = 0.5 * (u_i1(i) + u_i_new);
        u_e1(i) = 0.5 * (u_e1(i) + u_e_new);
    }
    
    // Conditions aux limites finales
    n_i1(0) = n_i1(1); n_i1(Nx) = n_i1(Nx - 1);
    n_e1(0) = n_e1(1); n_e1(Nx) = n_e1(Nx - 1);
    u_i1(0) = u_i1(1); u_i1(Nx) = u_i1(Nx - 1);
    u_e1(0) = u_e1(1); u_e1(Nx) = u_e1(Nx - 1);

    calc_potential(n_i1, n_e1, Phi);
    calc_electric_field(Phi, E);

    compute_time_step();
}