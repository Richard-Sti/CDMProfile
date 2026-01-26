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
#ifndef CDMPROF_NESTED_OPTIMIZER_H
#define CDMPROF_NESTED_OPTIMIZER_H

#include "loss.h"

/*
 * Maximum number of halos and parameters.
 */
#define MAX_HALOS 1000
#define MAX_GLOBAL_PARAMS 4
#define MAX_LOCAL_PARAMS 4

/*
 * Data for a single halo.
 */
typedef struct {
    double* bin_counts;
    double* bin_positions;
    int nbin;
    int npart;
    double rmin;
    double rmax;
    SimpsonGrid grid;  /* Precomputed per-halo */
} HaloData;

/*
 * Configuration for nested optimization.
 *
 * The density function has signature: rho(r, Rs, a0, a1, a2, a3)
 * We split these into global (shared) and local (per-halo) params.
 *
 * param_is_global[i] = 1 if parameter i is global, 0 if local
 * Index mapping: 0=Rs, 1=a0, 2=a1, 3=a2, 4=a3
 *
 * Example: If Rs and a1 are local, a0 and a2 are global:
 *   param_is_global = {0, 1, 0, 1, 0}  (Rs=local, a0=global, a1=local, ...)
 *   n_global = 2 (a0, a2)
 *   n_local = 2 (Rs, a1)
 */
typedef struct {
    /* Halo data */
    HaloData* halos;
    int nhalos;

    /* Density function */
    DensityFunc rho;
    int nparams_total;  /* Total params used by rho (1 to 5) */

    /* Parameter classification */
    int param_is_global[5];  /* 1 if global, 0 if local */
    int n_global;            /* Number of global params */
    int n_local;             /* Number of local params per halo */

    /* Bounds (in physical space) */
    double lower_bounds[5];
    double upper_bounds[5];

    /* Optimization settings */
    double xtol;
    double ftol;
    int inner_maxeval;   /* Max evals per restart for inner optimization */
    int outer_maxeval;   /* Max evals for outer (global) optimization */

    /* Inner restart settings (like fit_with_restarts) */
    int inner_max_restarts;   /* Max restarts per halo */
    int inner_nconv_required; /* Convergences to same min required */
    double inner_conv_rtol;   /* Relative tolerance for convergence */
    double inner_conv_atol;   /* Absolute tolerance for convergence */

    /* Outer restart settings */
    int outer_max_restarts;   /* Max restarts for outer optimization */
    int outer_nconv_required; /* Convergences to same min required */
    double outer_conv_rtol;   /* Relative tolerance for convergence */
    double outer_conv_atol;   /* Absolute tolerance for convergence */

    /* Warm start: previous local solutions (nhalos x n_local) */
    double* local_solutions;  /* NULL on first call, updated after */

    /* Parallelization */
    int nthreads;  /* Number of OpenMP threads (0 = use default) */

    /* Numerical stability */
    double min_density;  /* Minimum density threshold (underflow protection) */

    /* Verbosity */
    int verbose;
} NestedOptConfig;

/*
 * Results from nested optimization.
 */
typedef struct {
    /* Global parameters (n_global) */
    double global_params[MAX_GLOBAL_PARAMS];

    /* Local parameters per halo (nhalos x n_local) */
    double* local_params;

    /* Per-halo losses */
    double* halo_losses;

    /* Total loss (sum of halo losses) */
    double total_loss;

    /* Convergence info */
    int converged;
    int outer_neval;
    int total_inner_neval;
} NestedOptResult;

/*
 * Initialize nested optimization configuration.
 * Call this before fit_nested_profile.
 *
 * Parameters:
 *   config         - Config struct to initialize
 *   halos          - Array of halo data (nhalos)
 *   nhalos         - Number of halos
 *   rho            - Density function pointer
 *   nparams_total  - Total params used by rho (1 to 5: Rs + a0..a3)
 *   param_is_global - Array of length 5: 1 if param is global, 0 if local
 */
void nested_opt_init(NestedOptConfig* config,
                     HaloData* halos, int nhalos,
                     DensityFunc rho, int nparams_total,
                     int* param_is_global);

/*
 * Set bounds for all parameters.
 *
 * Parameters:
 *   config       - Config struct
 *   lower_bounds - Lower bounds array (length 5)
 *   upper_bounds - Upper bounds array (length 5)
 */
void nested_opt_set_bounds(NestedOptConfig* config,
                           double* lower_bounds, double* upper_bounds);

/*
 * Set optimization tolerances and limits.
 */
void nested_opt_set_tolerances(NestedOptConfig* config,
                               double xtol, double ftol,
                               int inner_maxeval, int outer_maxeval);

/*
 * Set inner restart settings.
 */
void nested_opt_set_inner_restarts(NestedOptConfig* config,
                                   int max_restarts, int nconv_required,
                                   double conv_rtol, double conv_atol);

/*
 * Set outer restart settings.
 */
void nested_opt_set_outer_restarts(NestedOptConfig* config,
                                   int max_restarts, int nconv_required,
                                   double conv_rtol, double conv_atol);

/*
 * Fit density profiles using nested optimization.
 *
 * Outer loop: Nelder-Mead on global parameters
 * Inner loop: Nelder-Mead on local parameters (per halo, independent)
 *
 * Parameters:
 *   config         - Configuration struct (must be initialized)
 *   initial_global - Initial guess for global params (n_global)
 *   initial_local  - Initial guess for local params (nhalos x n_local)
 *                    If NULL, uses midpoint of bounds
 *   result         - Output results struct
 *
 * Note: result->local_params and result->halo_losses are allocated
 *       internally and must be freed by caller.
 */
void fit_nested_profile(NestedOptConfig* config,
                        double* initial_global,
                        double* initial_local,
                        NestedOptResult* result);

/*
 * Free memory allocated by fit_nested_profile.
 */
void nested_opt_result_free(NestedOptResult* result);

#endif /* CDMPROF_NESTED_OPTIMIZER_H */
