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
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include <nlopt.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "nested_optimizer.h"
#include "loss.h"


/*
 * Internal: maps from split (global, local) params to full param array.
 */
typedef struct {
    int global_indices[MAX_GLOBAL_PARAMS];  /* Which full indices are global */
    int local_indices[MAX_LOCAL_PARAMS];    /* Which full indices are local */
    int n_global;
    int n_local;
    int nparams_total;
} ParamMapping;


/*
 * Internal: data passed to inner (local) objective function.
 */
typedef struct {
    HaloData* halo;
    DensityFunc rho;
    const double* global_params;  /* Fixed during inner optimization */
    const ParamMapping* mapping;
    double* lower_bounds;
    double* upper_bounds;
    int nparams_total;
    int* eval_count;  /* Pointer to eval counter (for tracking) */
    double min_density;  /* Minimum density threshold */
} InnerObjectiveData;


/*
 * Internal: data passed to outer (global) objective function.
 */
typedef struct {
    NestedOptConfig* config;
    const ParamMapping* mapping;
    double* local_solutions;  /* nhalos x n_local, warm start */
    int total_inner_neval;    /* Accumulated inner evaluations */
} OuterObjectiveData;


/* Counter for evaluations */
static int inner_eval_count = 0;
static int outer_eval_count = 0;


/*
 * Build the parameter mapping from config.
 */
static void build_param_mapping(const NestedOptConfig* config,
                                 ParamMapping* mapping) {
    mapping->n_global = 0;
    mapping->n_local = 0;
    mapping->nparams_total = config->nparams_total;

    for (int i = 0; i < config->nparams_total; i++) {
        if (config->param_is_global[i]) {
            mapping->global_indices[mapping->n_global++] = i;
        } else {
            mapping->local_indices[mapping->n_local++] = i;
        }
    }
}


/*
 * Assemble full parameter array from global and local params.
 * full_params[i] = global or local depending on mapping.
 */
static void assemble_params(const ParamMapping* mapping,
                            const double* global_params,
                            const double* local_params,
                            double* full_params) {
    /* Initialize to zero for unused params */
    for (int i = 0; i < 5; i++) {
        full_params[i] = 0.0;
    }

    /* Fill global params */
    for (int i = 0; i < mapping->n_global; i++) {
        int idx = mapping->global_indices[i];
        full_params[idx] = global_params[i];
    }

    /* Fill local params */
    for (int i = 0; i < mapping->n_local; i++) {
        int idx = mapping->local_indices[i];
        full_params[idx] = local_params[i];
    }
}


/*
 * Transform parameters to internal space (log for Rs if it's local).
 * Rs is always index 0.
 */
static void to_internal_space(const ParamMapping* mapping,
                              const double* physical, double* internal,
                              int is_local) {
    int n = is_local ? mapping->n_local : mapping->n_global;
    const int* indices = is_local ? mapping->local_indices
                                   : mapping->global_indices;

    for (int i = 0; i < n; i++) {
        if (indices[i] == 0) {
            /* Rs: use log space */
            internal[i] = log(physical[i]);
        } else {
            internal[i] = physical[i];
        }
    }
}


/*
 * Transform parameters from internal to physical space.
 */
static void to_physical_space(const ParamMapping* mapping,
                              const double* internal, double* physical,
                              int is_local) {
    int n = is_local ? mapping->n_local : mapping->n_global;
    const int* indices = is_local ? mapping->local_indices
                                   : mapping->global_indices;

    for (int i = 0; i < n; i++) {
        if (indices[i] == 0) {
            /* Rs: use exp to recover */
            physical[i] = exp(internal[i]);
        } else {
            physical[i] = internal[i];
        }
    }
}


/*
 * Transform bounds to internal space.
 */
static void bounds_to_internal(const ParamMapping* mapping,
                               const double* lb_phys, const double* ub_phys,
                               double* lb_int, double* ub_int,
                               int is_local) {
    int n = is_local ? mapping->n_local : mapping->n_global;
    const int* indices = is_local ? mapping->local_indices
                                   : mapping->global_indices;

    for (int i = 0; i < n; i++) {
        int full_idx = indices[i];
        if (full_idx == 0) {
            /* Rs: log space */
            lb_int[i] = log(lb_phys[full_idx]);
            ub_int[i] = log(ub_phys[full_idx]);
        } else {
            lb_int[i] = lb_phys[full_idx];
            ub_int[i] = ub_phys[full_idx];
        }
    }
}


/*
 * Inner objective: optimize local params for a single halo.
 * Global params are fixed.
 */
static double inner_objective(unsigned n, const double* x,
                               double* grad, void* data) {
    (void)grad;  /* Gradient-free */
    (void)n;
    InnerObjectiveData* obj = (InnerObjectiveData*)data;

    /* Increment eval counter if tracking */
    if (obj->eval_count != NULL) {
        (*obj->eval_count)++;
    }

    /* Transform from internal to physical space */
    double local_physical[MAX_LOCAL_PARAMS];
    to_physical_space(obj->mapping, x, local_physical, 1);

    /* Assemble full parameter array */
    double full_params[5];
    assemble_params(obj->mapping, obj->global_params, local_physical,
                    full_params);

    /* Compute loss */
    double loss = compute_loss(obj->halo->bin_counts, obj->halo->bin_positions,
                               obj->halo->nbin, obj->halo->npart,
                               &obj->halo->grid, obj->rho,
                               full_params[0], full_params[1], full_params[2],
                               full_params[3], full_params[4],
                               obj->min_density);

    return loss;
}


/*
 * Simple deterministic sampling for restarts.
 * Returns a value in [0, 1] based on restart index and dimension.
 */
static double restart_sample(int restart_idx, int dim, int max_restarts) {
    /* Use a simple low-discrepancy-like sequence */
    unsigned int seed = (unsigned int)(restart_idx * 7 + dim * 13 + 1);
    seed = seed * 1103515245 + 12345;
    return (double)((seed >> 16) & 0x7FFF) / 32767.0;
}


/*
 * Run a single optimization attempt for local parameters.
 * Returns the loss value. Best params stored in out_params.
 * out_neval is incremented with number of function evaluations.
 */
static double run_single_inner_opt(HaloData* halo, DensityFunc rho,
                                    const double* global_params,
                                    const ParamMapping* mapping,
                                    double* lb_int, double* ub_int,
                                    double* initial_internal,
                                    double xtol, double ftol, int maxeval,
                                    double min_density,
                                    double* out_params_internal,
                                    int* out_neval) {
    int n_local = mapping->n_local;

    /* Setup inner objective data */
    InnerObjectiveData obj_data;
    obj_data.halo = halo;
    obj_data.rho = rho;
    obj_data.global_params = global_params;
    obj_data.mapping = mapping;
    obj_data.nparams_total = mapping->nparams_total;
    obj_data.eval_count = out_neval;
    obj_data.min_density = min_density;

    /* Create optimizer */
    nlopt_opt opt = nlopt_create(NLOPT_LN_NELDERMEAD, n_local);
    if (opt == NULL) {
        memcpy(out_params_internal, initial_internal, n_local * sizeof(double));
        return DBL_MAX;
    }

    nlopt_set_lower_bounds(opt, lb_int);
    nlopt_set_upper_bounds(opt, ub_int);
    nlopt_set_min_objective(opt, inner_objective, &obj_data);
    nlopt_set_xtol_rel(opt, xtol);
    nlopt_set_ftol_rel(opt, ftol);
    nlopt_set_maxeval(opt, maxeval);

    /* Copy initial guess */
    double x[MAX_LOCAL_PARAMS];
    memcpy(x, initial_internal, n_local * sizeof(double));

    /* Clamp to bounds */
    for (int i = 0; i < n_local; i++) {
        if (x[i] < lb_int[i]) x[i] = lb_int[i];
        if (x[i] > ub_int[i]) x[i] = ub_int[i];
    }

    /* Optimize */
    double minf;
    nlopt_result ret = nlopt_optimize(opt, x, &minf);

    memcpy(out_params_internal, x, n_local * sizeof(double));
    nlopt_destroy(opt);

    if (ret < 0 && ret != NLOPT_ROUNDOFF_LIMITED) {
        return DBL_MAX;
    }

    return minf;
}


/*
 * Fit local parameters for a single halo with restarts.
 *
 * Returns the loss value. Best local params are stored in out_local_params.
 * out_neval is set to the number of inner function evaluations.
 */
static double fit_single_halo(HaloData* halo, DensityFunc rho,
                               const double* global_params,
                               const ParamMapping* mapping,
                               double* lower_bounds, double* upper_bounds,
                               double* initial_local,
                               double xtol, double ftol, int maxeval,
                               int max_restarts, int nconv_required,
                               double conv_rtol, double conv_atol,
                               double min_density,
                               double* out_local_params,
                               int* out_neval) {
    int n_local = mapping->n_local;

    if (n_local == 0) {
        /* No local params: just evaluate loss with global params */
        double full_params[5];
        assemble_params(mapping, global_params, NULL, full_params);
        if (out_neval != NULL) *out_neval = 1;
        return compute_loss(halo->bin_counts, halo->bin_positions,
                            halo->nbin, halo->npart, &halo->grid, rho,
                            full_params[0], full_params[1], full_params[2],
                            full_params[3], full_params[4], min_density);
    }

    /* Transform bounds to internal space */
    double lb_int[MAX_LOCAL_PARAMS], ub_int[MAX_LOCAL_PARAMS];
    bounds_to_internal(mapping, lower_bounds, upper_bounds, lb_int, ub_int, 1);

    /* Transform initial guess to internal space */
    double init_int[MAX_LOCAL_PARAMS];
    to_internal_space(mapping, initial_local, init_int, 1);

    /* Best result tracking */
    double best_loss = DBL_MAX;
    double best_params_int[MAX_LOCAL_PARAMS];
    int nconv = 0;

    /* Initialize eval counter */
    int total_evals = 0;

    for (int restart = 0; restart < max_restarts; restart++) {
        /* Generate initial guess for this restart */
        double x0[MAX_LOCAL_PARAMS];

        if (restart == 0) {
            /* First restart: use provided initial guess */
            memcpy(x0, init_int, n_local * sizeof(double));
        } else {
            /* Subsequent restarts: sample from bounds */
            for (int i = 0; i < n_local; i++) {
                double t = restart_sample(restart, i, max_restarts);
                x0[i] = lb_int[i] + t * (ub_int[i] - lb_int[i]);
            }
        }

        /* Run single optimization */
        double result_int[MAX_LOCAL_PARAMS];
        double loss = run_single_inner_opt(halo, rho, global_params, mapping,
                                            lb_int, ub_int, x0,
                                            xtol, ftol, maxeval,
                                            min_density,
                                            result_int,
                                            &total_evals);

        /* Skip failed optimizations */
        if (loss >= 1e29) {
            continue;
        }

        /* Convergence tolerance */
        double tol = conv_atol + conv_rtol * fabs(best_loss);

        /* Found significantly better minimum? Reset counter */
        if (loss < best_loss - tol) {
            nconv = 0;
        }

        /* Update best if better */
        if (loss < best_loss) {
            best_loss = loss;
            memcpy(best_params_int, result_int, n_local * sizeof(double));
        }

        /* Converged to same minimum? */
        if (fabs(loss - best_loss) <= tol) {
            nconv++;
        }

        /* Early stop if converged enough times */
        if (nconv >= nconv_required) {
            break;
        }
    }

    /* Transform best result to physical space */
    if (best_loss < DBL_MAX) {
        to_physical_space(mapping, best_params_int, out_local_params, 1);
    } else {
        /* All restarts failed: return initial guess */
        memcpy(out_local_params, initial_local, n_local * sizeof(double));
    }

    if (out_neval != NULL) {
        *out_neval = total_evals;
    }

    return best_loss;
}


/*
 * Outer objective: for given global params, optimize all local params
 * and return total loss.
 */
static double outer_objective(unsigned n, const double* x,
                               double* grad, void* data) {
    (void)grad;  /* Gradient-free */
    (void)n;
    OuterObjectiveData* obj = (OuterObjectiveData*)data;
    NestedOptConfig* config = obj->config;
    const ParamMapping* mapping = obj->mapping;
    outer_eval_count++;

    /* Transform global params to physical space */
    double global_physical[MAX_GLOBAL_PARAMS];
    to_physical_space(mapping, x, global_physical, 0);

    if (config->verbose) {
        printf("  Outer eval %d: global = [", outer_eval_count);
        for (int i = 0; i < mapping->n_global; i++) {
            printf("%.6f%s", global_physical[i],
                   i < mapping->n_global - 1 ? ", " : "");
        }
        printf("]\n");
        fflush(stdout);
    }

    /* Fit each halo independently (parallel) */
    int n_local = mapping->n_local;
    int nhalos = config->nhalos;

    /* Allocate per-halo loss array for parallel reduction */
    double* halo_losses = (double*)malloc(nhalos * sizeof(double));

    /* Inner eval counter (thread-safe accumulation) */
    int inner_evals_this_call = 0;

#ifdef _OPENMP
    if (config->nthreads > 0) {
        omp_set_num_threads(config->nthreads);
    }
#endif

    #pragma omp parallel for schedule(dynamic) reduction(+:inner_evals_this_call)
    for (int h = 0; h < nhalos; h++) {
        /* Get initial guess for this halo (warm start) */
        double initial_local[MAX_LOCAL_PARAMS];
        if (obj->local_solutions != NULL && n_local > 0) {
            memcpy(initial_local, &obj->local_solutions[h * n_local],
                   n_local * sizeof(double));
        } else {
            /* Use midpoint of bounds */
            for (int i = 0; i < n_local; i++) {
                int idx = mapping->local_indices[i];
                initial_local[i] = 0.5 * (config->lower_bounds[idx] +
                                          config->upper_bounds[idx]);
            }
        }

        /* Optimize local params for this halo */
        double best_local[MAX_LOCAL_PARAMS];
        int halo_inner_evals = 0;
        double loss = fit_single_halo(&config->halos[h], config->rho,
                                       global_physical, mapping,
                                       config->lower_bounds,
                                       config->upper_bounds,
                                       initial_local,
                                       config->xtol, config->ftol,
                                       config->inner_maxeval,
                                       config->inner_max_restarts,
                                       config->inner_nconv_required,
                                       config->inner_conv_rtol,
                                       config->inner_conv_atol,
                                       config->min_density,
                                       best_local,
                                       &halo_inner_evals);

        halo_losses[h] = loss;
        inner_evals_this_call += halo_inner_evals;

        /* Update warm start for next outer iteration */
        if (obj->local_solutions != NULL && n_local > 0) {
            memcpy(&obj->local_solutions[h * n_local], best_local,
                   n_local * sizeof(double));
        }
    }

    obj->total_inner_neval += inner_evals_this_call;

    /* Sum losses and check for failures */
    double total_loss = 0.0;
    for (int h = 0; h < nhalos; h++) {
        if (config->verbose && h < 3) {
            printf("    Halo %d: loss=%.6e, rmin=%.4f, rmax=%.4f\n",
                   h, halo_losses[h],
                   config->halos[h].rmin, config->halos[h].rmax);
            fflush(stdout);
        }
        total_loss += halo_losses[h];
        if (total_loss >= DBL_MAX / 2.0) {
            free(halo_losses);
            return DBL_MAX;
        }
    }

    free(halo_losses);

    if (config->verbose) {
        printf("    -> total_loss = %.6e (%d halos)\n", total_loss, nhalos);
        fflush(stdout);
    }

    return total_loss;
}


void nested_opt_init(NestedOptConfig* config,
                     HaloData* halos, int nhalos,
                     DensityFunc rho, int nparams_total,
                     int* param_is_global) {
    config->halos = halos;
    config->nhalos = nhalos;
    config->rho = rho;
    config->nparams_total = nparams_total;

    /* Copy param classification */
    config->n_global = 0;
    config->n_local = 0;
    for (int i = 0; i < 5; i++) {
        config->param_is_global[i] = (i < nparams_total) ? param_is_global[i]
                                                          : 0;
        if (i < nparams_total) {
            if (param_is_global[i]) {
                config->n_global++;
            } else {
                config->n_local++;
            }
        }
    }

    /* Initialize Simpson grids for each halo */
    for (int h = 0; h < nhalos; h++) {
        simpson_grid_init(&halos[h].grid, halos[h].rmin, halos[h].rmax);
    }

    /* Default values */
    config->xtol = 1e-6;
    config->ftol = 1e-6;
    config->inner_maxeval = 500;
    config->outer_maxeval = 200;

    /* Inner restart defaults */
    config->inner_max_restarts = 100;
    config->inner_nconv_required = 10;
    config->inner_conv_rtol = 1e-3;
    config->inner_conv_atol = 10.0;

    /* Outer restart defaults */
    config->outer_max_restarts = 5;
    config->outer_nconv_required = 2;
    config->outer_conv_rtol = 1e-3;
    config->outer_conv_atol = 10.0;

    config->local_solutions = NULL;
    config->nthreads = 0;  /* Use default (all available) */
    config->verbose = 0;
}


void nested_opt_set_bounds(NestedOptConfig* config,
                           double* lower_bounds, double* upper_bounds) {
    for (int i = 0; i < 5; i++) {
        config->lower_bounds[i] = lower_bounds[i];
        config->upper_bounds[i] = upper_bounds[i];
    }
}


void nested_opt_set_tolerances(NestedOptConfig* config,
                               double xtol, double ftol,
                               int inner_maxeval, int outer_maxeval) {
    config->xtol = xtol;
    config->ftol = ftol;
    config->inner_maxeval = inner_maxeval;
    config->outer_maxeval = outer_maxeval;
}


void nested_opt_set_inner_restarts(NestedOptConfig* config,
                                   int max_restarts, int nconv_required,
                                   double conv_rtol, double conv_atol) {
    config->inner_max_restarts = max_restarts;
    config->inner_nconv_required = nconv_required;
    config->inner_conv_rtol = conv_rtol;
    config->inner_conv_atol = conv_atol;
}


void nested_opt_set_outer_restarts(NestedOptConfig* config,
                                   int max_restarts, int nconv_required,
                                   double conv_rtol, double conv_atol) {
    config->outer_max_restarts = max_restarts;
    config->outer_nconv_required = nconv_required;
    config->outer_conv_rtol = conv_rtol;
    config->outer_conv_atol = conv_atol;
}


void fit_nested_profile(NestedOptConfig* config,
                        double* initial_global,
                        double* initial_local,
                        NestedOptResult* result) {
    /* Build parameter mapping */
    ParamMapping mapping;
    build_param_mapping(config, &mapping);

    int n_global = mapping.n_global;
    int n_local = mapping.n_local;
    int nhalos = config->nhalos;

    /* Allocate result arrays */
    result->local_params = (double*)malloc(nhalos * n_local * sizeof(double));
    result->halo_losses = (double*)malloc(nhalos * sizeof(double));

    /* Allocate warm start storage */
    double* local_solutions = NULL;
    if (n_local > 0) {
        local_solutions = (double*)malloc(nhalos * n_local * sizeof(double));

        /* Initialize from provided initial_local or midpoint */
        if (initial_local != NULL) {
            memcpy(local_solutions, initial_local,
                   nhalos * n_local * sizeof(double));
        } else {
            for (int h = 0; h < nhalos; h++) {
                for (int i = 0; i < n_local; i++) {
                    int idx = mapping.local_indices[i];
                    local_solutions[h * n_local + i] =
                        0.5 * (config->lower_bounds[idx] +
                               config->upper_bounds[idx]);
                }
            }
        }
    }

    /* Reset counters */
    inner_eval_count = 0;
    outer_eval_count = 0;

    /* Handle edge cases */
    if (n_global == 0) {
        /* No global params: just fit each halo independently */
        result->total_loss = 0.0;
        int total_inner_evals = 0;
        for (int h = 0; h < nhalos; h++) {
            double initial_h[MAX_LOCAL_PARAMS];
            if (initial_local != NULL) {
                memcpy(initial_h, &initial_local[h * n_local],
                       n_local * sizeof(double));
            } else {
                for (int i = 0; i < n_local; i++) {
                    int idx = mapping.local_indices[i];
                    initial_h[i] = 0.5 * (config->lower_bounds[idx] +
                                          config->upper_bounds[idx]);
                }
            }

            double best_local[MAX_LOCAL_PARAMS];
            int halo_evals = 0;
            double loss = fit_single_halo(&config->halos[h], config->rho,
                                           NULL, &mapping,
                                           config->lower_bounds,
                                           config->upper_bounds,
                                           initial_h,
                                           config->xtol, config->ftol,
                                           config->inner_maxeval,
                                           config->inner_max_restarts,
                                           config->inner_nconv_required,
                                           config->inner_conv_rtol,
                                           config->inner_conv_atol,
                                           config->min_density,
                                           best_local,
                                           &halo_evals);
            total_inner_evals += halo_evals;

            memcpy(&result->local_params[h * n_local], best_local,
                   n_local * sizeof(double));
            result->halo_losses[h] = loss;
            result->total_loss += loss;
        }

        result->converged = 1;
        result->outer_neval = 0;
        result->total_inner_neval = total_inner_evals;

        if (local_solutions != NULL) free(local_solutions);
        return;
    }

    /* Transform bounds for global params */
    double lb_int[MAX_GLOBAL_PARAMS], ub_int[MAX_GLOBAL_PARAMS];
    bounds_to_internal(&mapping, config->lower_bounds, config->upper_bounds,
                       lb_int, ub_int, 0);

    /* Best result tracking for outer restarts */
    double best_loss = DBL_MAX;
    double best_global_int[MAX_GLOBAL_PARAMS];
    double* best_local_solutions = NULL;
    if (n_local > 0) {
        best_local_solutions = (double*)malloc(nhalos * n_local * sizeof(double));
    }
    int nconv = 0;
    int total_outer_neval = 0;
    int total_inner_neval_all = 0;

    /* Outer restart loop */
    for (int restart = 0; restart < config->outer_max_restarts; restart++) {
        if (config->verbose) {
            printf("Outer restart %d/%d\n", restart + 1,
                   config->outer_max_restarts);
            fflush(stdout);
        }

        /* Reset local solutions for each outer restart */
        if (n_local > 0) {
            if (initial_local != NULL) {
                memcpy(local_solutions, initial_local,
                       nhalos * n_local * sizeof(double));
            } else {
                for (int h = 0; h < nhalos; h++) {
                    for (int i = 0; i < n_local; i++) {
                        int idx = mapping.local_indices[i];
                        local_solutions[h * n_local + i] =
                            0.5 * (config->lower_bounds[idx] +
                                   config->upper_bounds[idx]);
                    }
                }
            }
        }

        /* Setup outer objective data */
        OuterObjectiveData outer_data;
        outer_data.config = config;
        outer_data.mapping = &mapping;
        outer_data.local_solutions = local_solutions;
        outer_data.total_inner_neval = 0;

        /* Create outer optimizer */
        nlopt_opt opt = nlopt_create(NLOPT_LN_NELDERMEAD, n_global);
        if (opt == NULL) {
            continue;
        }

        nlopt_set_lower_bounds(opt, lb_int);
        nlopt_set_upper_bounds(opt, ub_int);
        nlopt_set_min_objective(opt, outer_objective, &outer_data);
        nlopt_set_xtol_rel(opt, config->xtol);
        nlopt_set_ftol_rel(opt, config->ftol);
        nlopt_set_maxeval(opt, config->outer_maxeval);

        /* Generate initial guess for this restart */
        double x[MAX_GLOBAL_PARAMS];
        if (restart == 0 && initial_global != NULL) {
            to_internal_space(&mapping, initial_global, x, 0);
        } else {
            /* Sample from bounds */
            for (int i = 0; i < n_global; i++) {
                double t = restart_sample(restart, i, config->outer_max_restarts);
                x[i] = lb_int[i] + t * (ub_int[i] - lb_int[i]);
            }
        }

        /* Clamp to bounds */
        for (int i = 0; i < n_global; i++) {
            if (x[i] < lb_int[i]) x[i] = lb_int[i];
            if (x[i] > ub_int[i]) x[i] = ub_int[i];
        }

        /* Run outer optimization */
        double minf;
        nlopt_result ret = nlopt_optimize(opt, x, &minf);

        total_outer_neval += outer_eval_count;
        total_inner_neval_all += outer_data.total_inner_neval;
        outer_eval_count = 0;  /* Reset for next restart */

        nlopt_destroy(opt);

        /* Skip failed optimizations */
        if (ret < 0 && ret != NLOPT_ROUNDOFF_LIMITED) {
            continue;
        }
        if (minf >= 1e29) {
            continue;
        }

        /* Convergence tolerance */
        double tol = config->outer_conv_atol +
                     config->outer_conv_rtol * fabs(best_loss);

        /* Found significantly better minimum? Reset counter */
        if (minf < best_loss - tol) {
            nconv = 0;
        }

        /* Update best if better */
        if (minf < best_loss) {
            best_loss = minf;
            memcpy(best_global_int, x, n_global * sizeof(double));
            if (n_local > 0) {
                memcpy(best_local_solutions, local_solutions,
                       nhalos * n_local * sizeof(double));
            }
        }

        /* Converged to same minimum? */
        if (fabs(minf - best_loss) <= tol) {
            nconv++;
        }

        if (config->verbose) {
            printf("  Restart %d: loss=%.6e, best=%.6e, nconv=%d\n",
                   restart + 1, minf, best_loss, nconv);
            fflush(stdout);
        }

        /* Early stop if converged enough times */
        if (nconv >= config->outer_nconv_required) {
            break;
        }
    }

    /* Store best global params in physical space */
    if (best_loss < DBL_MAX) {
        to_physical_space(&mapping, best_global_int, result->global_params, 0);
        if (n_local > 0) {
            memcpy(local_solutions, best_local_solutions,
                   nhalos * n_local * sizeof(double));
        }
    } else {
        /* All restarts failed */
        if (initial_global != NULL) {
            memcpy(result->global_params, initial_global,
                   n_global * sizeof(double));
        } else {
            for (int i = 0; i < n_global; i++) {
                result->global_params[i] = 0.5 * (lb_int[i] + ub_int[i]);
            }
            to_physical_space(&mapping, result->global_params,
                              result->global_params, 0);
        }
    }

    result->total_loss = best_loss;
    result->converged = (nconv >= config->outer_nconv_required) ? 1 : 0;
    result->outer_neval = total_outer_neval;
    result->total_inner_neval = total_inner_neval_all;

    /* Final pass: get per-halo losses with optimal global params */
    for (int h = 0; h < nhalos; h++) {
        if (n_local > 0) {
            memcpy(&result->local_params[h * n_local],
                   &local_solutions[h * n_local],
                   n_local * sizeof(double));
        }

        /* Compute final loss for this halo */
        double full_params[5];
        double* local_h = (n_local > 0) ? &result->local_params[h * n_local]
                                         : NULL;
        assemble_params(&mapping, result->global_params, local_h, full_params);

        result->halo_losses[h] = compute_loss(
            config->halos[h].bin_counts, config->halos[h].bin_positions,
            config->halos[h].nbin, config->halos[h].npart,
            &config->halos[h].grid, config->rho,
            full_params[0], full_params[1], full_params[2],
            full_params[3], full_params[4], config->min_density);
    }

    if (best_local_solutions != NULL) free(best_local_solutions);
    if (local_solutions != NULL) free(local_solutions);
}


void nested_opt_result_free(NestedOptResult* result) {
    if (result->local_params != NULL) {
        free(result->local_params);
        result->local_params = NULL;
    }
    if (result->halo_losses != NULL) {
        free(result->halo_losses);
        result->halo_losses = NULL;
    }
}
