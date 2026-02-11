#include "../include/plasma.hpp"
#include "Eigen/src/Core/Matrix.h"

void PlasmaSystem::calc_potential(Eigen::VectorXd& n_i, Eigen::VectorXd& n_e, Eigen::VectorXd& Phi_out) {
    m_rho = -(m_params.q_i * n_i + m_params.q_e * n_e) / m_params.epsilon_0 * m_dx * m_dx;
    Phi_out = m_poisson_solver.solve(m_rho);
}

void PlasmaSystem::calc_electric_field(Eigen::VectorXd& Phi_in, Eigen::VectorXd& E_out) {
    // E = -dPhi/dx using central differences
    for (int i = 1; i < m_Nx; ++i) {
        E_out(i) = -(Phi_in(i+1) - Phi_in(i-1)) / (2.0 * m_dx) + m_E0(i);
    }
    // Boundary conditions for E
    E_out(0) = m_E0(0) - (Phi_in(1) - Phi_in(m_Nx)) / (2.0 * m_dx);
    E_out(m_Nx) = m_E0(m_Nx) - (Phi_in(0) - Phi_in(m_Nx-1)) / (2.0 * m_dx);
}

void PlasmaSystem::apply_boundary_conditions(Eigen::VectorXd& n_e, Eigen::VectorXd& n_i,
                                        Eigen::VectorXd& u_e, Eigen::VectorXd& u_i) {
    const double half_dx = 0.5 * m_dx;
    const double inv_dx = 1.0 / m_dx;
    
    // Compute minmod slopes at boundaries using periodic neighbors
    // For i = 0: use values at Nx-1, Nx, and 1
    // For i = Nx: use values at Nx-1, 0, and 1
    
    // === Ion density at boundaries ===
    double a_n_i_0 = (n_i(m_Nx) - n_i(m_Nx - 1)) * inv_dx;  // backward diff at 0 (using periodicity)
    double b_n_i_0 = (n_i(0) - n_i(m_Nx)) * inv_dx;          // forward diff at 0
    double minmod_n_i_0 = 0.5 * (std::copysign(1.0, a_n_i_0) + std::copysign(1.0, b_n_i_0)) 
                        * std::min(std::abs(a_n_i_0), std::abs(b_n_i_0));
    
    double a_n_i_Nx = (n_i(0) - n_i(m_Nx)) * inv_dx;     // backward diff at Nx
    double b_n_i_Nx = (n_i(1) - n_i(0)) * inv_dx;            // forward diff at Nx
    double minmod_n_i_Nx = 0.5 * (std::copysign(1.0, a_n_i_Nx) + std::copysign(1.0, b_n_i_Nx)) 
                         * std::min(std::abs(a_n_i_Nx), std::abs(b_n_i_Nx));
    
    // MUSCL reconstruction: average of left and right states
    n_i(0) = 0.5 * ((n_i(m_Nx) + half_dx * minmod_n_i_0) + (n_i(1) - half_dx * m_minmod_i_n(1)));
    n_i(m_Nx) = 0.5 * ((n_i(m_Nx - 1) + half_dx * m_minmod_i_n(m_Nx - 1)) + (n_i(1) - half_dx * minmod_n_i_Nx));
    
    // === Electron density at boundaries ===
    double a_n_e_0 = (n_e(m_Nx) - n_e(m_Nx - 1)) * inv_dx;
    double b_n_e_0 = (n_e(0) - n_e(m_Nx)) * inv_dx;
    double minmod_n_e_0 = 0.5 * (std::copysign(1.0, a_n_e_0) + std::copysign(1.0, b_n_e_0)) 
                        * std::min(std::abs(a_n_e_0), std::abs(b_n_e_0));
    
    double a_n_e_Nx = (n_e(0) - n_e(m_Nx)) * inv_dx;
    double b_n_e_Nx = (n_e(1) - n_e(0)) * inv_dx;
    double minmod_n_e_Nx = 0.5 * (std::copysign(1.0, a_n_e_Nx) + std::copysign(1.0, b_n_e_Nx)) 
                         * std::min(std::abs(a_n_e_Nx), std::abs(b_n_e_Nx));
    
    n_e(0) = 0.5 * ((n_e(m_Nx) + half_dx * minmod_n_e_0) + (n_e(1) - half_dx * m_minmod_e_n(1)));
    n_e(m_Nx) = 0.5 * ((n_e(m_Nx - 1) + half_dx * m_minmod_e_n(m_Nx - 1)) + (n_e(1) - half_dx * minmod_n_e_Nx));
    
    // === Ion velocity at boundaries ===
    double a_u_i_0 = (u_i(m_Nx) - u_i(m_Nx - 1)) * inv_dx;
    double b_u_i_0 = (u_i(1) - u_i(m_Nx)) * inv_dx;
    double minmod_u_i_0 = 0.5 * (std::copysign(1.0, a_u_i_0) + std::copysign(1.0, b_u_i_0)) 
                        * std::min(std::abs(a_u_i_0), std::abs(b_u_i_0));
    
    double a_u_i_Nx = (u_i(0) - u_i(m_Nx - 1)) * inv_dx;
    double b_u_i_Nx = (u_i(1) - u_i(0)) * inv_dx;
    double minmod_u_i_Nx = 0.5 * (std::copysign(1.0, a_u_i_Nx) + std::copysign(1.0, b_u_i_Nx)) 
                         * std::min(std::abs(a_u_i_Nx), std::abs(b_u_i_Nx));
    
    u_i(0) = 0.5 * ((u_i(m_Nx) + half_dx * minmod_u_i_0) + (u_i(1) - half_dx * m_minmod_i_u(1)));
    u_i(m_Nx) = 0.5 * ((u_i(m_Nx - 1) + half_dx * m_minmod_i_u(m_Nx - 1)) + (u_i(1) - half_dx * minmod_u_i_Nx));
    
    // === Electron velocity at boundaries ===
    double a_u_e_0 = (u_e(m_Nx) - u_e(m_Nx - 1)) * inv_dx;
    double b_u_e_0 = (u_e(1) - u_e(m_Nx)) * inv_dx;
    double minmod_u_e_0 = 0.5 * (std::copysign(1.0, a_u_e_0) + std::copysign(1.0, b_u_e_0)) 
                        * std::min(std::abs(a_u_e_0), std::abs(b_u_e_0));
    
    double a_u_e_Nx = (u_e(0) - u_e(m_Nx - 1)) * inv_dx;
    double b_u_e_Nx = (u_e(1) - u_e(0)) * inv_dx;
    double minmod_u_e_Nx = 0.5 * (std::copysign(1.0, a_u_e_Nx) + std::copysign(1.0, b_u_e_Nx)) 
                         * std::min(std::abs(a_u_e_Nx), std::abs(b_u_e_Nx));
    
    u_e(0) = 0.5 * ((u_e(m_Nx) + half_dx * minmod_u_e_0) + (u_e(1) - half_dx * m_minmod_e_u(1)));
    u_e(m_Nx) = 0.5 * ((u_e(m_Nx - 1) + half_dx * m_minmod_e_u(m_Nx - 1)) + (u_e(1) - half_dx * minmod_u_e_Nx));
}

void PlasmaSystem::minmod(Eigen::VectorXd& n_i, Eigen::VectorXd& n_e, 
                          Eigen::VectorXd& u_i, Eigen::VectorXd& u_e) {
    const double inv_dx = 1.0 / m_dx;
    const int N = m_Nx - 1;  // Number of interior points
    
    // Vectorized backward and forward differences
    Eigen::VectorXd a_n_i = (n_i.segment(1, N) - n_i.segment(0, N)) * inv_dx;
    Eigen::VectorXd b_n_i = (n_i.segment(2, N) - n_i.segment(1, N)) * inv_dx;
    
    Eigen::VectorXd a_n_e = (n_e.segment(1, N) - n_e.segment(0, N)) * inv_dx;
    Eigen::VectorXd b_n_e = (n_e.segment(2, N) - n_e.segment(1, N)) * inv_dx;
    
    Eigen::VectorXd a_u_i = (u_i.segment(1, N) - u_i.segment(0, N)) * inv_dx;
    Eigen::VectorXd b_u_i = (u_i.segment(2, N) - u_i.segment(1, N)) * inv_dx;
    
    Eigen::VectorXd a_u_e = (u_e.segment(1, N) - u_e.segment(0, N)) * inv_dx;
    Eigen::VectorXd b_u_e = (u_e.segment(2, N) - u_e.segment(1, N)) * inv_dx;
    
    // Vectorized minmod: 0.5 * (sign(a) + sign(b)) * min(|a|, |b|)
    m_minmod_i_n.segment(1, N) = 0.5 * (a_n_i.array().sign() + b_n_i.array().sign()) * a_n_i.array().abs().min(b_n_i.array().abs());
    m_minmod_e_n.segment(1, N) = 0.5 * (a_n_e.array().sign() + b_n_e.array().sign()) * a_n_e.array().abs().min(b_n_e.array().abs());
    m_minmod_i_u.segment(1, N) = 0.5 * (a_u_i.array().sign() + b_u_i.array().sign()) * a_u_i.array().abs().min(b_u_i.array().abs());
    m_minmod_e_u.segment(1, N) = 0.5 * (a_u_e.array().sign() + b_u_e.array().sign()) * a_u_e.array().abs().min(b_u_e.array().abs());
}

void PlasmaSystem::MUSCL_reconstruction_all(Eigen::VectorXd& n_i, Eigen::VectorXd& n_e,
                                             Eigen::VectorXd& u_i, Eigen::VectorXd& u_e) {
    const double half_dx = 0.5 * m_dx;
    
    // Ion density: left and right states at all interfaces (vectorized)
    m_n_iL_vec = n_i.head(m_Nx).array() + half_dx * m_minmod_i_n.head(m_Nx).array();
    m_n_iR_vec = n_i.tail(m_Nx).array() - half_dx * m_minmod_i_n.tail(m_Nx).array();
    
    // Electron density: left and right states at all interfaces
    m_n_eL_vec = n_e.head(m_Nx).array() + half_dx * m_minmod_e_n.head(m_Nx).array();
    m_n_eR_vec = n_e.tail(m_Nx).array() - half_dx * m_minmod_e_n.tail(m_Nx).array();
    
    // Ion velocity: left and right states at all interfaces
    m_u_iL_vec = u_i.head(m_Nx).array() + half_dx * m_minmod_i_u.head(m_Nx).array();
    m_u_iR_vec = u_i.tail(m_Nx).array() - half_dx * m_minmod_i_u.tail(m_Nx).array();
    
    // Electron velocity: left and right states at all interfaces
    m_u_eL_vec = u_e.head(m_Nx).array() + half_dx * m_minmod_e_u.head(m_Nx).array();
    m_u_eR_vec = u_e.tail(m_Nx).array() - half_dx * m_minmod_e_u.tail(m_Nx).array();
}

void PlasmaSystem::calc_fluxes_all() {
    // Maximum wave speeds: |u| + c_s (sound speed) - vectorized
    Eigen::ArrayXd alpha_i = (m_u_iL_vec.array().abs() + m_params.cs_i).max(m_u_iR_vec.array().abs() + m_params.cs_i);
    Eigen::ArrayXd alpha_e = (m_u_eL_vec.array().abs() + m_params.cs_e).max(m_u_eR_vec.array().abs() + m_params.cs_e);
    
    // Ion density flux: F = n*u (mass conservation)
    Eigen::ArrayXd F_iL_n = (m_params.n_i0 + m_n_iL_vec.array()) * m_u_iL_vec.array();
    Eigen::ArrayXd F_iR_n = (m_params.n_i0 + m_n_iR_vec.array()) * m_u_iR_vec.array();
    m_F_i_n.head(m_Nx) = 0.5 * (F_iL_n + F_iR_n) - 0.5 * alpha_i * (m_n_iR_vec.array() - m_n_iL_vec.array());
    
    // Electron density flux: F = n*u (mass conservation)
    Eigen::ArrayXd F_eL_n = (m_params.n_e0 + m_n_eL_vec.array()) * m_u_eL_vec.array();
    Eigen::ArrayXd F_eR_n = (m_params.n_e0 + m_n_eR_vec.array()) * m_u_eR_vec.array();
    m_F_e_n.head(m_Nx) = 0.5 * (F_eL_n + F_eR_n) - 0.5 * alpha_e * (m_n_eR_vec.array() - m_n_eL_vec.array());
    
    // Ion velocity flux: F = u²/2 (momentum advection)
    Eigen::ArrayXd F_iL_u = 0.5 * m_u_iL_vec.array().square();
    Eigen::ArrayXd F_iR_u = 0.5 * m_u_iR_vec.array().square();
    m_F_i_u.head(m_Nx) = 0.5 * (F_iL_u + F_iR_u) - 0.5 * alpha_i * (m_u_iR_vec.array() - m_u_iL_vec.array());
    
    // Electron velocity flux: F = u²/2 (momentum advection)
    Eigen::ArrayXd F_eL_u = 0.5 * m_u_eL_vec.array().square();
    Eigen::ArrayXd F_eR_u = 0.5 * m_u_eR_vec.array().square();
    m_F_e_u.head(m_Nx) = 0.5 * (F_eL_u + F_eR_u) - 0.5 * alpha_e * (m_u_eR_vec.array() - m_u_eL_vec.array());
}

void PlasmaSystem::calc_pressure_gradients(Eigen::VectorXd& n_i, Eigen::VectorXd& n_e) {
    const double inv_2dx = 1.0 / (2.0 * m_dx);
    const int N = m_Nx - 1;  // Number of interior points
    
    // Vectorized density gradients for ions
    Eigen::ArrayXd dn_i_dx = (n_i.segment(2, N) - n_i.segment(0, N)).array() * inv_2dx;
    Eigen::ArrayXd left_term_ion = m_params.gamma_i / m_params.n_i0 * dn_i_dx * (1 + m_params.gamma_i * n_i.segment(1, N).array() / m_params.n_i0);
    Eigen::ArrayXd right_term_ion = - 2 * m_params.gamma_i * n_i.segment(1, N).array() * dn_i_dx / (m_params.n_i0 * m_params.n_i0);
    m_grad_P_i.segment(1, N) = m_params.P_i0 / m_params.n_i0 * (left_term_ion + right_term_ion);
    
    // Vectorized density gradients for electrons
    Eigen::ArrayXd dn_e_dx = (n_e.segment(2, N) - n_e.segment(0, N)).array() * inv_2dx;
    Eigen::ArrayXd left_term_electron = m_params.gamma_e / m_params.n_e0 * dn_e_dx * (1 + m_params.gamma_e * n_e.segment(1, N).array() / m_params.n_e0);
    Eigen::ArrayXd right_term_electron = - 2 * m_params.gamma_e * n_e.segment(1, N).array() * dn_e_dx / (m_params.n_e0 * m_params.n_e0);
    m_grad_P_e.segment(1, N) = m_params.P_e0 / m_params.n_e0 * (left_term_electron + right_term_electron);
    
    // === Boundary conditions at i = 0 (using periodic: n(Nx) and n(1)) ===
    double dn_i_dx_0 = (n_i(1) - n_i(m_Nx)) * inv_2dx;
    double left_term_ion_0 = m_params.gamma_i / m_params.n_i0 * dn_i_dx_0 * (1 + m_params.gamma_i * n_i(0) / m_params.n_i0);
    double right_term_ion_0 = -2 * m_params.gamma_i * n_i(0) * dn_i_dx_0 / (m_params.n_i0 * m_params.n_i0);
    m_grad_P_i(0) = m_params.P_i0 / m_params.n_i0 * (left_term_ion_0 + right_term_ion_0);
    
    double dn_e_dx_0 = (n_e(1) - n_e(m_Nx)) * inv_2dx;
    double left_term_electron_0 = m_params.gamma_e / m_params.n_e0 * dn_e_dx_0 * (1 + m_params.gamma_e * n_e(0) / m_params.n_e0);
    double right_term_electron_0 = -2 * m_params.gamma_e * n_e(0) * dn_e_dx_0 / (m_params.n_e0 * m_params.n_e0);
    m_grad_P_e(0) = m_params.P_e0 / m_params.n_e0 * (left_term_electron_0 + right_term_electron_0);
    
    // === Boundary conditions at i = m_Nx (using periodic: n(Nx-1) and n(0)) ===
    double dn_i_dx_Nx = (n_i(0) - n_i(m_Nx - 1)) * inv_2dx;
    double left_term_ion_Nx = m_params.gamma_i / m_params.n_i0 * dn_i_dx_Nx * (1 + m_params.gamma_i * n_i(m_Nx) / m_params.n_i0);
    double right_term_ion_Nx = -2 * m_params.gamma_i * n_i(m_Nx) * dn_i_dx_Nx / (m_params.n_i0 * m_params.n_i0);
    m_grad_P_i(m_Nx) = m_params.P_i0 / m_params.n_i0 * (left_term_ion_Nx + right_term_ion_Nx);
    
    double dn_e_dx_Nx = (n_e(0) - n_e(m_Nx - 1)) * inv_2dx;
    double left_term_electron_Nx = m_params.gamma_e / m_params.n_e0 * dn_e_dx_Nx * (1 + m_params.gamma_e * n_e(m_Nx) / m_params.n_e0);
    double right_term_electron_Nx = -2 * m_params.gamma_e * n_e(m_Nx) * dn_e_dx_Nx / (m_params.n_e0 * m_params.n_e0);
    m_grad_P_e(m_Nx) = m_params.P_e0 / m_params.n_e0 * (left_term_electron_Nx + right_term_electron_Nx);
}

void PlasmaSystem::step_euler() {
    // Precompute constants used in operations for efficiency
    const double dt_dx = m_dt / m_dx;
    const double dt_mi = m_dt / m_params.m_i;
    const double dt_me = m_dt / m_params.m_e;
    const int N = m_Nx - 1;  // Number of interior points
    
    // === STAGE 1: Predictor (forward Euler) ===
    minmod(m_n_i1, m_n_e1, m_u_i1, m_u_e1);
    MUSCL_reconstruction_all(m_n_i1, m_n_e1, m_u_i1, m_u_e1);
    calc_fluxes_all();

    // Compute pressure gradients (vectorized)
    calc_pressure_gradients(m_n_i1, m_n_e1);

    m_n_i_star.segment(1, N) = m_n_i1.segment(1, N).array() 
                              - dt_dx * (m_F_i_n.segment(1, N) - m_F_i_n.segment(0, N)).array();
    m_n_e_star.segment(1, N) = m_n_e1.segment(1, N).array() 
                              - dt_dx * (m_F_e_n.segment(1, N) - m_F_e_n.segment(0, N)).array();
    
    
    // Vectorized velocity update: du/dt = -d(u²/2)/dx - dP/dx/m
    m_u_i_star.segment(1, N) = m_u_i1.segment(1, N).array() 
                             - dt_dx * (m_F_i_u.segment(1, N) - m_F_i_u.segment(0, N)).array()
                             - dt_mi * m_grad_P_i.segment(1, N).array();
    m_u_e_star.segment(1, N) = m_u_e1.segment(1, N).array() 
                             - dt_dx * (m_F_e_u.segment(1, N) - m_F_e_u.segment(0, N)).array()
                             - dt_me * m_grad_P_e.segment(1, N).array();
    
    // Apply mirror boundary conditions to star state
    apply_boundary_conditions(m_n_e_star, m_n_i_star, m_u_e_star, m_u_i_star);

    // === STAGE 2: Corrector (Heun's averaging) ===
    calc_potential(m_n_i_star, m_n_e_star, m_Phi_star);
    calc_electric_field(m_Phi_star, m_E_star);
    minmod(m_n_i_star, m_n_e_star, m_u_i_star, m_u_e_star);
    
    // Compute MUSCL reconstruction and fluxes using star state (vectorized)
    MUSCL_reconstruction_all(m_n_i_star, m_n_e_star, m_u_i_star, m_u_e_star);
    calc_fluxes_all();
    
    // Compute pressure gradients for star state (vectorized)
    calc_pressure_gradients(m_n_i_star, m_n_e_star);
    
    // Vectorized Heun averaging: y^(n+1) = y^n + 0.5*(k1 + k2)*dt
    Eigen::ArrayXd n_i_new = m_n_i1.segment(1, N).array() 
                           - dt_dx * (m_F_i_n.segment(1, N) - m_F_i_n.segment(0, N)).array();
    Eigen::ArrayXd n_e_new = m_n_e1.segment(1, N).array() 
                           - dt_dx * (m_F_e_n.segment(1, N) - m_F_e_n.segment(0, N)).array();
    
    Eigen::ArrayXd u_i_new = m_u_i1.segment(1, N).array() 
                           - dt_dx * (m_F_i_u.segment(1, N) - m_F_i_u.segment(0, N)).array()
                           + dt_mi * (m_params.q_i * m_E_star.segment(1, N).array() - m_grad_P_i.segment(1, N).array());
    Eigen::ArrayXd u_e_new = m_u_e1.segment(1, N).array() 
                           - dt_dx * (m_F_e_u.segment(1, N) - m_F_e_u.segment(0, N)).array()
                           + dt_me * (m_params.q_e * m_E_star.segment(1, N).array() - m_grad_P_e.segment(1, N).array());
    
    // Average initial and corrected states
    m_n_i1.segment(1, N) = 0.5 * (m_n_i1.segment(1, N).array() + n_i_new);
    m_n_e1.segment(1, N) = 0.5 * (m_n_e1.segment(1, N).array() + n_e_new);
    m_u_i1.segment(1, N) = 0.5 * (m_u_i1.segment(1, N).array() + u_i_new);
    m_u_e1.segment(1, N) = 0.5 * (m_u_e1.segment(1, N).array() + u_e_new);
    
    // Apply mirror boundary conditions to final state
    apply_boundary_conditions(m_n_e1, m_n_i1, m_u_e1, m_u_i1);

    // Update electric field for next time step
    calc_potential(m_n_i1, m_n_e1, m_Phi);
    calc_electric_field(m_Phi, m_E);

    // Recompute time step based on CFL condition
    compute_time_step();
}