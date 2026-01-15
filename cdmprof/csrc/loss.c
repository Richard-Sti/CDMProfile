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


double simpson_mass(DensityFunc rho, double rmin, double rmax, int N,
                    double Rs, double a0, double a1, double a2, double a3) {
    /*
     * Simpson's 1/3 rule for integrating 4*pi*r^2*rho(r) from rmin to rmax.
     * Uses N intervals (N+1 points). N must be even.
     * Also checks that the density profile is monotonically decreasing.
     *
     * Uses logarithmic spacing via change of variables u = log(r):
     *   dr = r * du
     *   integral of 4*pi*r^2*rho(r) dr = integral of 4*pi*r^3*rho(r) du
     */
    double log_rmin = log(rmin);
    double log_rmax = log(rmax);
    double h = (log_rmax - log_rmin) / (double)N;

    double sum = 0.0;
    double rho_prev;

    /* First point: weight 1 */
    double r0 = rmin;
    double rho0 = rho(r0, Rs, a0, a1, a2, a3);
    if (!isfinite(rho0) || rho0 <= 0.0) return -1.0;
    sum += r0 * r0 * r0 * rho0;  /* r^3 * rho for log spacing */
    rho_prev = rho0;

    /* Interior points */
    for (int i = 1; i < N; i++) {
        double u = log_rmin + i * h;
        double r = exp(u);
        double rho_val = rho(r, Rs, a0, a1, a2, a3);
        if (!isfinite(rho_val) || rho_val <= 0.0) return -1.0;

        /* Check monotonicity: density must decrease with radius */
        if (rho_val > rho_prev) return -1.0;
        rho_prev = rho_val;

        double f = r * r * r * rho_val;  /* r^3 * rho for log spacing */

        /* Odd indices: weight 4, Even indices: weight 2 */
        if (i % 2 == 1) {
            sum += 4.0 * f;
        } else {
            sum += 2.0 * f;
        }
    }

    /* Last point: weight 1 */
    double rn = rmax;
    double rhon = rho(rn, Rs, a0, a1, a2, a3);
    if (!isfinite(rhon) || rhon <= 0.0) return -1.0;
    /* Check monotonicity for last point */
    if (rhon > rho_prev) return -1.0;
    sum += rn * rn * rn * rhon;  /* r^3 * rho for log spacing */

    /* Simpson's factor and 4*pi for spherical integral */
    double integral = (h / 3.0) * sum;
    return 4.0 * PI * integral;
}


double compute_loss(double* bin_counts, double* bin_positions, int nbin,
                    int npart, double rmin, double rmax,
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
    double mass = simpson_mass(rho, rmin, rmax, SIMPSON_N,
                               Rs, a0, a1, a2, a3);

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
