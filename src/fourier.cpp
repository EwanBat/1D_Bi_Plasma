#include <complex>
#include <vector>
#include <fstream>
#include <string>
#include <iostream>

#include "../include/fourier.hpp"
#include "../include/Eigen/Dense"
#include "../include/Eigen/src/Core/Matrix.h"

#include <complex>

/*
  NOTE ABOUT CONVENTIONS AND SHAPES

  - This file implements a forward transform (x -> k) for the initial condition
    and an inverse transform (k -> x).

  - Matrix storage convention used by the code:
      * Spectral domain (k, t): input(idk, step) => rows = k modes, cols = time steps.
      * Physical domain (x, t): output(idx, step) => rows = x points, cols = time steps.

  - fourier_transform_initial:
      * Currently computes the forward transform only for the first time index (step = 0).
      * It reads input(idx, 0) over x and writes output(idk, 0) over k.
      * t_grid is only used to size the output (Nt = t_grid.size()) but the function
        does not iterate over time yet.

  - inverse_fourier_transform:
      * Computes the inverse transform for all time steps, for each x, summing over k.
      * The function returns the real part only (imaginary part is discarded).

  - Normalization:
      * Forward: 1 / Nx
      * Inverse: 1 / Nk
    This is consistent for a unitary-like pair if you prefer symmetric factors (1/sqrt(Nx), 1/sqrt(Nk)),
    you will need to adjust both functions accordingly.

  - Assumptions:
      * x_grid and k_grid define the sampling points used in the Riemann sums.
      * No explicit requirement of uniform spacing is enforced, but the simple 1/N factor
        implicitly assumes uniform spacing. If spacing is non-uniform, replace the
        1/N normalization by a proper quadrature weight.

  - Limitations / TODOs:
      * fourier_transform_initial ignores all columns except column 0 (time step 0).
      * Only the real part is stored for the inverse transform result, losing any imaginary part.
      * Consider vectorizing with Eigen or using FFT for performance if grids are uniform and periodic.
*/

// input  : rows = x, cols = t (only column 0 is used here)
// output : rows = k, cols = t (only column 0 is written here)
/**
 * Compute the forward Fourier transform at t = t_grid[0] (first time step) from x-space to k-space.
 *
 * Parameters:
 * - input  (Eigen::MatrixXd&): Physical-space field sampled on x_grid. Expected shape: (Nx, Nt).
 *                              Only column 0 is read: input(idx, 0), idx in [0, Nx).
 * - output (Eigen::MatrixXcd&): Spectral-space output. Resized to (Nk, Nt); only column 0 is written.
 * - t_grid (std::vector<double>&): Time grid; used to infer Nt but not iterated over.
 * - k_grid (std::vector<double>&): Wavenumber grid; looped over to produce spectral coefficients.
 * - x_grid (std::vector<double>&): Space grid; looped over to accumulate spatial sum.
 *
 * Notes:
 * - Normalization is 1/Nx.
 * - The transform computed is: F(k) = (1/Nx) * sum_x f(x) * exp(-i k x)
 * - Only the first time slice is transformed; extend loops if you need all time steps.
 */
void fourier_transform_initial(Eigen::MatrixXd &input,
                                Eigen::MatrixXcd &output, 
                                std::vector<double>& t_grid,
                                std::vector<double>& k_grid,
                                std::vector<double>& x_grid)
{
    int Nt = static_cast<int>(t_grid.size());   // number of time steps (not used beyond sizing)
    int Nk = static_cast<int>(k_grid.size());   // number of k modes
    int Nx = static_cast<int>(x_grid.size());   // number of x points

    // Allocate output as (k, t). Only column 0 will be filled below.
    output.resize(Nk, Nt);

    // Compute spectrum at t = t_grid[0] only.
    for (int idk = 0; idk < Nk; ++idk) {
        double k = k_grid[idk];
        std::complex<double> sum = 0.0;

        // Accumulate spatial sum over all x points (reads input(:, 0)).
        for (int idx = 0; idx < Nx; ++idx) {
            double x = x_grid[idx];
            // exp(-i k x) kernel for the forward transform
            sum += input(idx, 0) * std::exp(std::complex<double>(0.0, k * x));
        }

        // Normalize by Nx and write the complex coefficient for (idk, time=0).
        // Only the real part is kept in original code; however assigning a double to a complex
        // implicitly sets the imaginary part to 0.0. If you want to keep the full complex sum,
        // assign 'sum / double(Nx)' directly.
        std::complex<double> sum_final = (sum / static_cast<double>(Nx));
        output(idk, 0) = sum_final;
    }
}

/**
 * Stream the inverse Fourier transform (k -> x) directly to a CSV file.
 *
 * Purpose:
 *  - Avoid allocating a large (Nx x Nt) dense matrix by computing each output row on the fly
 *    and writing it immediately to disk.
 *
 * Input:
 *  - input (Eigen::MatrixXcd&): spectral-domain data F(k, t), shape (Nk, Nt), accessed as input(idk, step)
 *  - t_grid (std::vector<double>&): time grid, length Nt (used to determine number of columns)
 *  - k_grid (std::vector<double>&): wavenumber grid, length Nk (summation index of the inverse transform)
 *  - x_grid (std::vector<double>&): spatial grid, length Nx (rows written to the CSV)
 *  - filename (std::string): path to the CSV file to write
 *
 * Output format:
 *  - CSV with Nx lines; each line corresponds to a spatial position x_grid[idx]
 *  - Each line contains Nt comma-separated real values: f(x_idx, t_step)
 *
 * Mathematical form and normalization:
 *  - For each (x, t): f(x, t) = (1/Nk) * sum_{k} F(k, t) * exp(+i k x)
 *  - Only the real part is written (imaginary part is discarded)
 *  - If you need symmetric normalization or to preserve complex values, adjust accordingly
 *
 * Error handling:
 *  - Throws std::runtime_error if the output file cannot be opened
 *
 * Notes:
 *  - Assumes a simple 1/Nk normalization; if k_grid spacing is non-uniform, consider a weighted sum
 *  - The function is CPU-bound (O(Nx * Nt * Nk)); consider vectorization or FFT-based methods if applicable
 */
void inverse_fourier_transform_to_csv(Eigen::MatrixXcd& input,
                                      const std::vector<double>& t_grid,
                                      const std::vector<double>& k_grid,
                                      const std::vector<double>& x_grid,
                                      const std::string& filename)
{
    const int Nt = static_cast<int>(t_grid.size());
    const int Nk = static_cast<int>(k_grid.size());
    const int Nx = static_cast<int>(x_grid.size());

    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    // For each spatial position x, compute a full line of Nt values and write it as comma-separated values.
    for (int idx = 0; idx < Nx; ++idx) {
        const double x = x_grid[idx];
        // Iterate over all time steps; this forms the columns of the CSV
        for (int step = 0; step < Nt; ++step) {
            std::complex<double> sum = 0.0;
            // Accumulate contribution of all k-modes for the current (x, t)
            for (int idk = 0; idk < Nk; ++idk) {
                const double k = k_grid[idk];
                sum += input(idk, step) * std::exp(std::complex<double>(0.0, -k * x));
            }
            // Normalize by Nk and store real part only
            const double value = (sum / static_cast<double>(Nk)).real();
            file << value;
            if (step < Nt - 1) file << ",";
        }
        file << "\n";
    }
}

