# 1D Two-Fluid + Maxwell Simulation (C++ / Eigen)

## 📖 Description

This project implements a **1D spectral simulation** of a collisionless plasma modeled by the **two-fluid equations (ions + electrons) coupled with Maxwell's equations**.  
The goal is to reproduce **longitudinal waves (Langmuir, ion-acoustic)** and **transverse electromagnetic waves** in the linear regime.

The code is written in **C++17**, uses the **[Eigen](https://eigen.tuxfamily.org/)** library for linear algebra (vectors, matrices, eigenvalues), and is compiled with a portable **Makefile**.

---

## ⚡ Main Features

- **Spectral (FFT)** representation in space (1D periodic).  
- **Linearization** of the two-fluid + Maxwell system.  
- Split into independent blocks:
  - Longitudinal block (electrostatic plasma).  
  - Transverse block (electromagnetic waves).  
- Time evolution by:
  - Diagonalization of matrices for each mode \(k\), or  
  - Application of an integration scheme (e.g., RK4).  
- Analysis of the **numerical dispersion relation** by computing eigenvalues.

---

## 📦 Dependencies

- **C++17** or higher  
- **Eigen 3** (header-only library)  
- **FFTW3** *(optional, if using fast FFT instead of a custom implementation)*  

On Debian/Ubuntu:

```bash
sudo apt-get install libeigen3-dev libfftw3-dev
```
