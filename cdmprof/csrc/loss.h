/*
 * Copyright (C) 2024 Richard Stiskalek
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation; either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
#ifndef CDMPROF_LOSS_H
#define CDMPROF_LOSS_H

#define SIMPSON_N 512
#define SIMPSON_RTOL 0.01  /* Relative tolerance for convergence check */
#define PI 3.14159265358979323846
#define MAX_PARAMS 6

/*
 * Density function pointer type.
 * Signature: rho(r, Rs, a0, a1, a2, a3)
 * Unused parameters should be ignored by the function.
 */
typedef double (*DensityFunc)(double r, double Rs,
                               double a0, double a1, double a2, double a3);

/*
 * Simpson's rule integration of 4*pi*r^2*rho(r) from rmin to rmax.
 *
 * Parameters:
 *   rho   - Density function pointer
 *   rmin  - Minimum radius
 *   rmax  - Maximum radius
 *   N     - Number of intervals (must be even)
 *   Rs    - Scale radius
 *   a0-a3 - Free parameters
 *
 * Returns:
 *   Enclosed mass (unnormalized), or -1.0 if invalid
 */
double simpson_mass(DensityFunc rho, double rmin, double rmax, int N,
                    double Rs, double a0, double a1, double a2, double a3);

/*
 * Compute the negative log-likelihood loss for fitting a density profile.
 *
 * Loss = -sum(bin_counts * (2*log(bin_positions) + log(rho)))
 *        + npart * log(M(rmin, rmax))
 *
 * Parameters:
 *   bin_counts    - Particle counts per bin (nbin,)
 *   bin_positions - Radial bin positions (nbin,)
 *   nbin          - Number of bins
 *   npart         - Total particle count
 *   rmin          - Minimum radius for mass integration
 *   rmax          - Maximum radius for mass integration
 *   rho           - Density function pointer
 *   Rs            - Scale radius
 *   a0-a3         - Free parameters
 *
 * Returns:
 *   Loss value (large positive value if invalid)
 */
double compute_loss(double* bin_counts, double* bin_positions, int nbin,
                    int npart, double rmin, double rmax,
                    DensityFunc rho,
                    double Rs, double a0, double a1, double a2, double a3);

#endif /* CDMPROF_LOSS_H */
