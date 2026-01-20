#include "../include/plasma.hpp"

void PlasmaSystem::calc_potential(Eigen::VectorXd& n_i, Eigen::VectorXd& n_e, Eigen::VectorXd& Phi_out) {
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m_Nx + 1, m_Nx + 1);
    Eigen::VectorXd rho = (m_params.q_i * n_i + m_params.q_e * n_e) / m_params.epsilon_0;
    Eigen::VectorXd b = - rho * m_dx * m_dx;

    // Neumann mirror BC at i=0: dPhi/dx = 0 => Phi_0 - Phi_1 = 0
    A(0, 0) = 1.0; 
    A(0, 1) = -1.0;
    b(0) = 0.0;

    // Interior points: standard second-order finite difference stencil
    for (int i = 1; i <= m_Nx - 1; ++i) {
        A(i, i-1) = 1.0;
        A(i, i)   = -2.0;
        A(i, i+1) = 1.0;
        // b(i) already set as -rho*dx^2
    }

    // Neumann BC at i=Nx: dPhi/dx = 0 => Phi_Nx - Phi_{Nx-1} = 0
    A(m_Nx, m_Nx - 1) = -1.0; 
    A(m_Nx, m_Nx) = 1.0;
    b(m_Nx) = 0.0;

    // Solve linear system using QR decomposition
    Phi_out = A.colPivHouseholderQr().solve(b);
}

void PlasmaSystem::calc_electric_field(Eigen::VectorXd& Phi_in, Eigen::VectorXd& E_out) {
    // Interior points: centered difference
    for (int i = 1; i < m_Nx; ++i) {
        E_out(i) = - (Phi_in(i+1) - Phi_in(i-1)) / (2.0 * m_dx) + m_E0(i);
    }
    // Mirror boundary conditions
    E_out(0) = E_out(1);
    E_out(m_Nx) = E_out(m_Nx-1);
}

void PlasmaSystem::apply_boundary_conditions(Eigen::VectorXd& n_e, Eigen::VectorXd& n_i,
                                        Eigen::VectorXd& u_e, Eigen::VectorXd& u_i) {
    // Mirror boundary conditions at both ends
    n_i(0) = n_i(1);         n_i(m_Nx) = n_i(m_Nx-1);
    n_e(0) = n_e(1);         n_e(m_Nx) = n_e(m_Nx-1);
    u_i(0) = 0;         u_i(m_Nx) = 0;
    u_e(0) = 0;         u_e(m_Nx) = 0;
}

void PlasmaSystem::minmod(Eigen::VectorXd& n_i, Eigen::VectorXd& n_e, 
                          Eigen::VectorXd& u_i, Eigen::VectorXd& u_e) {
    const double inv_dx = 1.0 / m_dx;  // Compute once for efficiency
    double a, b;
    
    for (int i = 1; i < m_Nx; ++i) {
        // Ion density slope
        a = (n_i(i) - n_i(i - 1)) * inv_dx;
        b = (n_i(i + 1) - n_i(i)) * inv_dx;
        m_minmod_i_n(i) = 0.5 * (std::copysign(1.0, a) + std::copysign(1.0, b)) * std::min(std::abs(a), std::abs(b));

        // Electron density slope
        a = (n_e(i) - n_e(i - 1)) * inv_dx;
        b = (n_e(i + 1) - n_e(i)) * inv_dx;
        m_minmod_e_n(i) = 0.5 * (std::copysign(1.0, a) + std::copysign(1.0, b)) * std::min(std::abs(a), std::abs(b));

        // Ion velocity slope
        a = (u_i(i) - u_i(i - 1)) * inv_dx;
        b = (u_i(i + 1) - u_i(i)) * inv_dx;
        m_minmod_i_u(i) = 0.5 * (std::copysign(1.0, a) + std::copysign(1.0, b)) * std::min(std::abs(a), std::abs(b));

        // Electron velocity slope
        a = (u_e(i) - u_e(i - 1)) * inv_dx;
        b = (u_e(i + 1) - u_e(i)) * inv_dx;
        m_minmod_e_u(i) = 0.5 * (std::copysign(1.0, a) + std::copysign(1.0, b)) * std::min(std::abs(a), std::abs(b));
    }
}

void PlasmaSystem::MUSCL_reconstruction(int i, Eigen::VectorXd& n_i, Eigen::VectorXd& n_e,
                                        Eigen::VectorXd& u_i, Eigen::VectorXd& u_e) {
    // Ion density: left and right states at interface
    m_n_iL = n_i(i) + 0.5 * m_minmod_i_n(i) * m_dx;
    m_n_iR = n_i(i+1) - 0.5 * m_minmod_i_n(i+1) * m_dx;

    // Electron density: left and right states at interface
    m_n_eL = n_e(i) + 0.5 * m_minmod_e_n(i) * m_dx;
    m_n_eR = n_e(i+1) - 0.5 * m_minmod_e_n(i+1) * m_dx;

    // Ion velocity: left and right states at interface
    m_u_iL = u_i(i) + 0.5 * m_minmod_i_u(i) * m_dx;
    m_u_iR = u_i(i+1) - 0.5 * m_minmod_i_u(i+1) * m_dx;

    // Electron velocity: left and right states at interface
    m_u_eL = u_e(i) + 0.5 * m_minmod_e_u(i) * m_dx;
    m_u_eR = u_e(i+1) - 0.5 * m_minmod_e_u(i+1) * m_dx;
}

void PlasmaSystem::calc_fluxes(int i, Eigen::VectorXd& n_i, Eigen::VectorXd& n_e,
                                        Eigen::VectorXd& u_i, Eigen::VectorXd& u_e) { 
    // Maximum wave speeds: |u| + c_s (sound speed)
    double alpha_i = std::max(std::abs(m_u_iL) + m_params.cs_i, std::abs(m_u_iR) + m_params.cs_i);
    double alpha_e = std::max(std::abs(m_u_eL) + m_params.cs_e, std::abs(m_u_eR) + m_params.cs_e);
    
    // Ion density flux: F = n*u (mass conservation)
    double F_iL_n = (m_params.n_i0 + m_n_iL) * m_u_iL;
    double F_iR_n = (m_params.n_i0 + m_n_iR) * m_u_iR;
    m_F_i_n(i) = 0.5 * (F_iL_n + F_iR_n) - 0.5 * alpha_i * (m_n_iR - m_n_iL);

    // Electron density flux: F = n*u (mass conservation)
    double F_eL_n = (m_params.n_e0 + m_n_eL) * m_u_eL;
    double F_eR_n = (m_params.n_e0 + m_n_eR) * m_u_eR;
    m_F_e_n(i) = 0.5 * (F_eL_n + F_eR_n) - 0.5 * alpha_e * (m_n_eR - m_n_eL);

    // Ion velocity flux: F = u²/2 (momentum advection)
    double F_iL_u = 0.5 * m_u_iL * m_u_iL;
    double F_iR_u = 0.5 * m_u_iR * m_u_iR;
    m_F_i_u(i) = 0.5 * (F_iL_u + F_iR_u) - 0.5 * alpha_i * (m_u_iR - m_u_iL);

    // Electron velocity flux: F = u²/2 (momentum advection)
    double F_eL_u = 0.5 * m_u_eL * m_u_eL;
    double F_eR_u = 0.5 * m_u_eR * m_u_eR;
    m_F_e_u(i) = 0.5 * (F_eL_u + F_eR_u) - 0.5 * alpha_e * (m_u_eR - m_u_eL);
}

double PlasmaSystem::calc_pressure_gradient_i(int i, Eigen::VectorXd& n_i) {
    const double dn_i1_dx = (n_i(i + 1) - n_i(i - 1)) / (2.0 * m_dx);
    const double n_ratio = n_i(i) / m_params.n_i0;
    
    // Factorization: (P_i0 * gamma_i / n_i0²) * dn/dx * (1 - gamma_i * n/n_i0)
    const double factor = (m_params.P_i0 * m_params.gamma_i) / (m_params.n_i0 * m_params.n_i0);
    return factor * dn_i1_dx * (1.0 - m_params.gamma_i * n_ratio);
}

double PlasmaSystem::calc_pressure_gradient_e(int i, Eigen::VectorXd& n_e) {
    const double dn_e1_dx = (n_e(i + 1) - n_e(i - 1)) / (2.0 * m_dx);
    const double n_ratio = n_e(i) / m_params.n_e0;
    
    // Factorization: (P_e0 * gamma_e / n_e0²) * dn/dx * (1 - gamma_e * n/n_e0)
    const double factor = (m_params.P_e0 * m_params.gamma_e) / (m_params.n_e0 * m_params.n_e0);
    return factor * dn_e1_dx * (1.0 - m_params.gamma_e * n_ratio);
}

void PlasmaSystem::step_euler() {
    // Precompute constants used in loops for efficiency
    const double dt_dx = m_dt / m_dx;
    const double dt_mi = m_dt / m_params.m_i;
    const double dt_me = m_dt / m_params.m_e;
    
    // === STAGE 1: Predictor (forward Euler) ===
    minmod(m_n_i1, m_n_e1, m_u_i1, m_u_e1);
    
    // Compute numerical fluxes
    for (int i = 0; i < m_Nx; ++i) {
        MUSCL_reconstruction(i, m_n_i1, m_n_e1, m_u_i1, m_u_e1);
        calc_fluxes(i, m_n_i1, m_n_e1, m_u_i1, m_u_e1);
    }
    
    // Update to intermediate "star" state
    for (int i = 1; i < m_Nx; ++i) {
        // Density update: dn/dt = -d(nu)/dx
        m_n_i_star(i) = m_n_i1(i) - dt_dx * (m_F_i_n(i) - m_F_i_n(i-1));
        m_n_e_star(i) = m_n_e1(i) - dt_dx * (m_F_e_n(i) - m_F_e_n(i-1));
        
        const double grad_P_i = calc_pressure_gradient_i(i, m_n_i1);
        const double grad_P_e = calc_pressure_gradient_e(i, m_n_e1);
        
        // Velocity update: du/dt = -d(u²/2)/dx + (q*E - dP/dx)/m
        m_u_i_star(i) = m_u_i1(i) - dt_dx * (m_F_i_u(i) - m_F_i_u(i-1)) 
                      + dt_mi * (m_params.q_i * m_E(i) - grad_P_i);
        m_u_e_star(i) = m_u_e1(i) - dt_dx * (m_F_e_u(i) - m_F_e_u(i-1)) 
                      + dt_me * (m_params.q_e * m_E(i) - grad_P_e);
    }
    
    // Apply mirror boundary conditions to star state
    apply_boundary_conditions( m_n_e_star, m_n_i_star, m_u_e_star, m_u_i_star);

    // === STAGE 2: Corrector (Heun's averaging) ===
    calc_potential(m_n_i_star, m_n_e_star, m_Phi_star);
    calc_electric_field(m_Phi_star, m_E_star);
    minmod(m_n_i_star, m_n_e_star, m_u_i_star, m_u_e_star);
    
    // Recompute fluxes using star state
    for (int i = 0; i < m_Nx; ++i) {
        MUSCL_reconstruction(i, m_n_i_star, m_n_e_star, m_u_i_star, m_u_e_star);
        calc_fluxes(i, m_n_i_star, m_n_e_star, m_u_i_star, m_u_e_star);
    }
    
    // Heun averaging: y^(n+1) = y^n + 0.5*(k1 + k2)*dt
    for (int i = 1; i < m_Nx; ++i) {
        const double n_i_new = m_n_i1(i) - dt_dx * (m_F_i_n(i) - m_F_i_n(i-1));
        const double n_e_new = m_n_e1(i) - dt_dx * (m_F_e_n(i) - m_F_e_n(i-1));
        
        const double grad_P_i = calc_pressure_gradient_i(i, m_n_i_star);
        const double grad_P_e = calc_pressure_gradient_e(i, m_n_e_star);
        
        const double u_i_new = m_u_i1(i) - dt_dx * (m_F_i_u(i) - m_F_i_u(i-1)) 
                       + dt_mi * (m_params.q_i * m_E_star(i) - grad_P_i);
        const double u_e_new = m_u_e1(i) - dt_dx * (m_F_e_u(i) - m_F_e_u(i-1)) 
                       + dt_me * (m_params.q_e * m_E_star(i) - grad_P_e);
        
        // Average initial and corrected states
        m_n_i1(i) = 0.5 * (m_n_i1(i) + n_i_new);
        m_n_e1(i) = 0.5 * (m_n_e1(i) + n_e_new);
        m_u_i1(i) = 0.5 * (m_u_i1(i) + u_i_new);
        m_u_e1(i) = 0.5 * (m_u_e1(i) + u_e_new);
    }
    
    // Apply mirror boundary conditions to final state
    apply_boundary_conditions( m_n_e1, m_n_i1, m_u_e1, m_u_i1);

    // Update electric field for next time step
    calc_potential(m_n_i1, m_n_e1, m_Phi);
    calc_electric_field(m_Phi, m_E);

    // Recompute time step based on CFL condition
    compute_time_step();
}