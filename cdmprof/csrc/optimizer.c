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
#include <nlopt.h>
#include "optimizer.h"
#include "loss.h"


/* Counter for function evaluations */
static int eval_count = 0;


/*
 * NLopt objective function wrapper.
 * Works in transformed space: x[0] = log(Rs), x[1..] = a0, a1, ...
 */
static double nlopt_objective(unsigned n, const double* x,
                               double* grad, void* data) {
    (void)n;     /* Unused */
    (void)grad;  /* Gradient-free optimization */

    ObjectiveData* obj = (ObjectiveData*)data;
    eval_count++;

    /* Transform from internal to physical space: Rs = exp(x[0]) */
    double Rs = exp(x[0]);
    double a0 = (obj->nparams >= 2) ? x[1] : 0.0;
    double a1 = (obj->nparams >= 3) ? x[2] : 0.0;
    double a2 = (obj->nparams >= 4) ? x[3] : 0.0;
    double a3 = (obj->nparams >= 5) ? x[4] : 0.0;

    return compute_loss(obj->bin_counts, obj->bin_positions, obj->nbin,
                        obj->npart, &obj->grid,
                        obj->rho, Rs, a0, a1, a2, a3, obj->min_density);
}


void fit_profile(double* bin_counts, double* bin_positions, int nbin,
                 int npart, double rmin, double rmax,
                 DensityFunc rho, int nparams, double* initial_params,
                 double* lower_bounds, double* upper_bounds,
                 double xtol, double ftol, int maxeval,
                 int optimizer_type, double min_density,
                 double* out_params, double* out_loss,
                 int* out_converged, int* out_neval) {

    /* Setup objective data */
    ObjectiveData obj_data;
    obj_data.bin_counts = bin_counts;
    obj_data.bin_positions = bin_positions;
    obj_data.nbin = nbin;
    obj_data.npart = npart;
    obj_data.rho = rho;
    obj_data.nparams = nparams;
    obj_data.min_density = min_density;

    /* Precompute Simpson grid once for entire optimization */
    simpson_grid_init(&obj_data.grid, rmin, rmax);

    /* Select NLopt algorithm based on optimizer_type */
    nlopt_algorithm algo;
    if (optimizer_type == OPT_BOBYQA) {
        algo = NLOPT_LN_BOBYQA;
    } else if (optimizer_type == OPT_SBPLX) {
        algo = NLOPT_LN_SBPLX;
    } else {
        algo = NLOPT_LN_NELDERMEAD;  /* Default */
    }

    /* Create NLopt optimizer */
    nlopt_opt opt = nlopt_create(algo, nparams);
    if (opt == NULL) {
        *out_loss = 1e30;
        *out_converged = 0;
        *out_neval = 0;
        return;
    }

    /* Transform bounds to internal space: log(Rs) for first param */
    double* lb = (double*)malloc(nparams * sizeof(double));
    double* ub = (double*)malloc(nparams * sizeof(double));
    lb[0] = log(lower_bounds[0]);
    ub[0] = log(upper_bounds[0]);
    for (int i = 1; i < nparams; i++) {
        lb[i] = lower_bounds[i];
        ub[i] = upper_bounds[i];
    }
    nlopt_set_lower_bounds(opt, lb);
    nlopt_set_upper_bounds(opt, ub);

    /* Set objective function */
    nlopt_set_min_objective(opt, nlopt_objective, &obj_data);

    /* Set tolerances */
    nlopt_set_xtol_rel(opt, xtol);
    nlopt_set_ftol_rel(opt, ftol);
    nlopt_set_maxeval(opt, maxeval);

    /* Transform initial params to internal space: x[0] = log(Rs) */
    double* x = (double*)malloc(nparams * sizeof(double));
    x[0] = log(initial_params[0]);
    for (int i = 1; i < nparams; i++) {
        x[i] = initial_params[i];
    }

    /* Reset evaluation counter */
    eval_count = 0;

    /* Run optimization */
    double minf;
    nlopt_result nlopt_ret = nlopt_optimize(opt, x, &minf);

    /* Store results (transform Rs back to physical space) */
    *out_converged = (nlopt_ret > 0) ? 1 : 0;
    *out_loss = minf;
    *out_neval = eval_count;

    out_params[0] = exp(x[0]);  /* log(Rs) -> Rs */
    for (int i = 1; i < nparams; i++) {
        out_params[i] = x[i];
    }

    /* Cleanup */
    free(x);
    free(lb);
    free(ub);
    nlopt_destroy(opt);
}
