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
#ifndef CDMPROF_OPTIMIZER_H
#define CDMPROF_OPTIMIZER_H

#include "loss.h"

/*
 * Optimizer type enumeration.
 */
typedef enum {
    OPT_NELDERMEAD = 0,
    OPT_BOBYQA = 1,
    OPT_SBPLX = 2
} OptimizerType;

/*
 * Data passed to the objective function.
 */
typedef struct {
    double* bin_counts;
    double* bin_positions;
    int nbin;
    int npart;
    SimpsonGrid grid;  /* Precomputed Simpson integration grid */
    DensityFunc rho;
    int nparams;
} ObjectiveData;

/*
 * Fit a density profile to halo data using the specified optimizer.
 * Rs is optimized in log-space to ensure positivity.
 *
 * Parameters:
 *   bin_counts     - Particle counts per bin (nbin,)
 *   bin_positions  - Radial bin positions (nbin,)
 *   nbin           - Number of bins
 *   npart          - Total particle count
 *   rmin           - Minimum radius for mass integration
 *   rmax           - Maximum radius for mass integration
 *   rho            - Density function pointer
 *   nparams        - Number of parameters (1 + number of free params)
 *   initial_params - Initial guess [Rs, a0, a1, ...] in physical space
 *   lower_bounds   - Lower bounds [Rs_min, a0_min, ...] in physical space
 *   upper_bounds   - Upper bounds [Rs_max, a0_max, ...] in physical space
 *   xtol           - Relative tolerance on parameters
 *   ftol           - Relative tolerance on function value
 *   maxeval        - Maximum function evaluations
 *   optimizer_type - Optimizer: OPT_NELDERMEAD, OPT_BOBYQA, or OPT_SBPLX
 *   out_params     - Output: best-fit parameters in physical space
 *   out_loss       - Output: best-fit loss value
 *   out_converged  - Output: 1 if converged, 0 otherwise
 *   out_neval      - Output: number of function evaluations
 */
void fit_profile(double* bin_counts, double* bin_positions, int nbin,
                 int npart, double rmin, double rmax,
                 DensityFunc rho, int nparams, double* initial_params,
                 double* lower_bounds, double* upper_bounds,
                 double xtol, double ftol, int maxeval,
                 int optimizer_type,
                 double* out_params, double* out_loss,
                 int* out_converged, int* out_neval);

#endif /* CDMPROF_OPTIMIZER_H */
