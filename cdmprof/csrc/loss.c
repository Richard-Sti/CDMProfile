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
#include <math.h>
#include <float.h>
#include "loss.h"


void simpson_grid_init(SimpsonGrid* grid, double rmin, double rmax) {
    /*
     * Precompute the Simpson integration grid.
     * Uses multiplicative stepping to compute r and r^3 efficiently.
     */
    grid->h = (log(rmax) - log(rmin)) / (double)SIMPSON_N;
    double ratio = exp(grid->h);
    double ratio3 = ratio * ratio * ratio;

    double r = rmin;
    double r3 = rmin * rmin * rmin;

    for (int i = 0; i <= SIMPSON_N; i++) {
        grid->r[i] = r;
        grid->r3[i] = r3;
        r *= ratio;
        r3 *= ratio3;
    }
}


double simpson_mass(DensityFunc rho, const SimpsonGrid* grid,
                    double Rs, double a0, double a1, double a2, double a3) {
    /*
     * Simpson's 1/3 rule for integrating 4*pi*r^2*rho(r) using precomputed grid.
     * Uses SIMPSON_N intervals (SIMPSON_N+1 points). SIMPSON_N must be even.
     * Also checks that the density profile is monotonically decreasing.
     */
    double sum = 0.0;
    double rho_prev;

    /* First point: weight 1 */
    double rho_val = rho(grid->r[0], Rs, a0, a1, a2, a3);
    if (!isfinite(rho_val) || rho_val <= 0.0) return -1.0;
    sum += grid->r3[0] * rho_val;
    rho_prev = rho_val;

    /* Interior points: unrolled loop processing pairs (odd, even) */
    for (int i = 1; i < SIMPSON_N; i += 2) {
        /* Odd index: weight 4 */
        rho_val = rho(grid->r[i], Rs, a0, a1, a2, a3);
        if (!isfinite(rho_val) || rho_val <= 0.0 || rho_val > rho_prev)
            return -1.0;
        sum += 4.0 * grid->r3[i] * rho_val;
        rho_prev = rho_val;

        /* Even index: weight 2 (skip if this is the last point) */
        if (i + 1 < SIMPSON_N) {
            rho_val = rho(grid->r[i + 1], Rs, a0, a1, a2, a3);
            if (!isfinite(rho_val) || rho_val <= 0.0 || rho_val > rho_prev)
                return -1.0;
            sum += 2.0 * grid->r3[i + 1] * rho_val;
            rho_prev = rho_val;
        }
    }

    /* Last point: weight 1 */
    rho_val = rho(grid->r[SIMPSON_N], Rs, a0, a1, a2, a3);
    if (!isfinite(rho_val) || rho_val <= 0.0 || rho_val > rho_prev)
        return -1.0;
    sum += grid->r3[SIMPSON_N] * rho_val;

    /* Simpson's factor and 4*pi for spherical integral */
    return (4.0 * PI * grid->h / 3.0) * sum;
}


double compute_loss(double* bin_counts, double* bin_positions, int nbin,
                    int npart, const SimpsonGrid* grid,
                    DensityFunc rho,
                    double Rs, double a0, double a1, double a2, double a3) {
    /*
     * Compute the negative log-likelihood loss.
     *
     * L = -sum(n_i * (2*log(r_i) + log(rho(r_i)))) + N * log(M)
     *
     * where n_i = bin counts, r_i = bin positions, N = total particles,
     * M = enclosed mass from Simpson integration.
     */

    /* Check density at r=0: if finite, must be positive */
    double rho_zero = rho(0.0, Rs, a0, a1, a2, a3);
    if (isfinite(rho_zero) && rho_zero <= 0.0) {
        return DBL_MAX;
    }

    /*
     * Check r^2 * rho(r) is non-increasing between rmax and 10*rmax.
     * By rmax, the profile should be in its asymptotic regime where
     * mass shells are constant or decreasing. Allow small tolerance
     * for numerical precision and profiles approaching constant.
     * Sample 10 points logarithmically spaced.
     */
    double rmax = grid->r[SIMPSON_N];
    double r_outer_max = 10.0 * rmax;
    double log_ratio = log(r_outer_max / rmax) / 10.0;
    double r_check = rmax;
    double rho_val = rho(r_check, Rs, a0, a1, a2, a3);
    double r2rho_prev = r_check * r_check * rho_val;
    double r2rho_tol = 1.1;  /* Allow up to 10% total increase */

    for (int i = 1; i <= 10; i++) {
        r_check = rmax * exp(i * log_ratio);
        rho_val = rho(r_check, Rs, a0, a1, a2, a3);
        double r2rho = r_check * r_check * rho_val;

        /* If both are finite, check approximately non-increasing */
        if (isfinite(r2rho) && isfinite(r2rho_prev)) {
            if (r2rho > r2rho_prev * r2rho_tol) {
                return DBL_MAX;  /* Mass shell increased - reject */
            }
        }
        r2rho_prev = r2rho;
    }

    double sum_log_term = 0.0;

    /* Compute sum of n_i * (2*log(r_i) + log(rho(r_i))) */
    for (int i = 0; i < nbin; i++) {
        double r_i = bin_positions[i];
        double n_i = bin_counts[i];

        /* Skip empty bins */
        if (n_i <= 0.0) continue;

        double rho_val = rho(r_i, Rs, a0, a1, a2, a3);

        /* Invalid density: return large loss */
        if (!isfinite(rho_val) || rho_val <= 0.0) {
            return DBL_MAX;
        }

        sum_log_term += n_i * (2.0 * log(r_i) + log(rho_val));
    }

    /* Compute enclosed mass via Simpson integration */
    double mass = simpson_mass(rho, grid, Rs, a0, a1, a2, a3);

    /* Invalid mass: return large loss */
    if (mass <= 0.0 || !isfinite(mass)) {
        return DBL_MAX;
    }

    /* Final loss: -sum_term + npart * log(M) */
    double loss = -sum_log_term + (double)npart * log(mass);

    /* Check for NaN/Inf */
    if (!isfinite(loss)) {
        return DBL_MAX;
    }

    return loss;
}
