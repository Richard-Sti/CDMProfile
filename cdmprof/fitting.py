# Copyright (C) 2025 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
JIT-compiled fitting module for density profiles.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
from cffi import FFI
from scipy.stats import qmc
from sympy import Abs, ccode, simplify, symbols, sympify

from .symbolic import SympyParser

# Path to C source files
CSRC_DIR = Path(__file__).parent / "csrc"

# Optional per-job CFFI cache directory. Set this to a unique path before
# calling compile_fitter() to avoid .so file collisions between concurrent
# jobs. None uses CFFI's default (__pycache__ of this module).
CFFI_TMPDIR = None

# Constants for is_bad_function
_BAD_PREFIXES = ("0", "zoo", "<class")
_BAD_SUBSTRINGS = ("oo", "nan", "NaN")
_TRIG_FUNCTIONS = ("sin", "cos", "tan", "arcsin", "arccos", "atan")

# Optimizer type mapping
OPTIMIZER_MAP = {
    'neldermead': 0,
    'nelder-mead': 0,
    'bobyqa': 1,
    'sbplx': 2,
    'subplex': 2,
}


def _generate_lhs_samples(param_bounds, n_samples, seed=None):
    """
    Generate initial parameters using Latin Hypercube Sampling.

    Parameters
    ----------
    param_bounds : list of tuples
        List of (low, high) bounds for each parameter.
        First parameter (Rs) is sampled log-uniformly.
    n_samples : int
        Number of samples to generate.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, n_params) with initial parameter values.
    """
    n_params = len(param_bounds)
    sampler = qmc.LatinHypercube(d=n_params, seed=seed)
    # Generate samples in [0, 1]^d
    unit_samples = sampler.random(n=n_samples)

    # Scale to parameter bounds
    samples = np.zeros((n_samples, n_params), dtype=np.float64)
    for i, (low, high) in enumerate(param_bounds):
        if i == 0:
            # Rs: log-uniform sampling
            log_low, log_high = np.log(low), np.log(high)
            samples[:, i] = np.exp(
                unit_samples[:, i] * (log_high - log_low) + log_low)
        else:
            # Other parameters: uniform sampling
            samples[:, i] = unit_samples[:, i] * (high - low) + low

    return samples


def _has_normalization_only_param(expr_str):
    """Check if any parameter is just a normalization factor (f = a * g(x))."""
    x = symbols('x', positive=True)
    # Don't assume sign for parameters - fitter allows [-500, 500]
    params = [symbols(f'a{i}', real=True) for i in range(4)]

    try:
        # Parse the expression
        local_dict = {'x': x}
        local_dict.update({f'a{i}': params[i] for i in range(4)})
        expr = sympify(expr_str, locals=local_dict)

        # Check each parameter
        for a in params:
            if a not in expr.free_symbols:
                continue

            # Check if f = a * g(x) where g doesn't contain a
            quotient = simplify(expr / a)
            if a not in quotient.free_symbols:
                return True

            # Check if f = Abs(a) * g(x) where g doesn't contain a
            quotient_abs = simplify(expr / Abs(a))
            if a not in quotient_abs.free_symbols:
                return True

    except Exception:
        # If parsing fails, don't filter
        return False

    return False


def _detect_abs_wrapped_params(expr_str, parser=None):
    """Return set of param indices (0-3) exactly wrapped in Abs()."""
    if parser is None:
        parser = SympyParser()

    try:
        expr = parser.parse(expr_str)
    except Exception:
        return set()

    abs_wrapped = set()

    # Find all Abs() calls in the expression
    for atom in expr.atoms(Abs):
        arg = atom.args[0]  # The argument inside Abs()
        # Check if the argument is exactly one of the free parameters
        for i, param in enumerate(parser._free_params):
            if arg == param:
                abs_wrapped.add(i)

    return abs_wrapped


def is_bad_function(expr_str):
    """Check if expression is invalid (infinity, NaN, trig, etc.)."""
    if expr_str.startswith(_BAD_PREFIXES):
        return True
    if any(bs in expr_str for bs in _BAD_SUBSTRINGS):
        return True
    if any(trig in expr_str for trig in _TRIG_FUNCTIONS):
        return True
    if _has_normalization_only_param(expr_str):
        return True
    return False


def load_equations(filepath):
    """Load equations from a text file (one per line), skipping empty lines."""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]


###############################################################################
#                         C code compilation                                  #
###############################################################################


def _read_csrc(name):
    """Read a C source file, stripping local #include directives."""
    with open(CSRC_DIR / name, 'r') as f:
        lines = f.read().split('\n')
    return '\n'.join(
        ln for ln in lines if not ln.strip().startswith('#include "'))


def _get_nlopt_paths():
    """Get platform-specific include/library paths for nlopt."""
    if sys.platform == "darwin":
        try:
            brew_prefix = subprocess.check_output(
                ["brew", "--prefix"], text=True
            ).strip()
            return [f"{brew_prefix}/include"], [f"{brew_prefix}/lib"]
        except (subprocess.CalledProcessError, FileNotFoundError):
            return (
                ["/opt/homebrew/include", "/usr/local/include"],
                ["/opt/homebrew/lib", "/usr/local/lib"]
            )
    else:
        return [], []


def compile_fitter(expr_str, parser=None, simpson_n=512):
    """
    JIT compile a density function + loss + optimizer.

    Returns a fit() function with fit.fit_with_restarts() for multi-restart.
    """
    if parser is None:
        parser = SympyParser()

    # Validate simpson_n
    if simpson_n % 2 != 0:
        raise ValueError(f"simpson_n must be even, got {simpson_n}")

    # Parse expression and count parameters
    expr = parser.parse(expr_str)
    nfree = parser.count_free(expr)
    nparams = 1 + nfree  # Rs + free parameters

    # Detect parameters wrapped in Abs() for automatic positive bounds
    abs_wrapped_params = _detect_abs_wrapped_params(expr_str, parser)

    # Generate C code for density function
    expr_substituted = expr.subs(parser._x, parser._r / parser._Rs)
    c_expr = ccode(expr_substituted)

    # Read static C files
    loss_h = _read_csrc('loss.h')
    loss_c = _read_csrc('loss.c')
    optimizer_h = _read_csrc('optimizer.h')
    optimizer_c = _read_csrc('optimizer.c')

    # Generate density function C code (only dynamic part)
    # Use __attribute__((always_inline)) for GCC/Clang to ensure inlining
    rho_func_c = f"""
/* Auto-generated density function */
/* Expression: {expr_str} */
static inline __attribute__((always_inline))
double rho_func(double r, double Rs,
                double a0, double a1, double a2, double a3) {{
    (void)a0; (void)a1; (void)a2; (void)a3;  /* Suppress unused warnings */
    return {c_expr};
}}
"""

    # Wrapper that binds rho_func to the generic fit_profile
    wrapper_c = """
/* Wrapper that uses the auto-generated rho_func */
void fit_profile_wrapper(double* bin_counts, double* bin_positions, int nbin,
                         int npart, double rmin, double rmax,
                         int nparams, double* initial_params,
                         double* lower_bounds, double* upper_bounds,
                         double xtol, double ftol, int maxeval,
                         int optimizer_type, double min_density,
                         double* out_params, double* out_loss,
                         int* out_converged, int* out_neval) {
    fit_profile(bin_counts, bin_positions, nbin, npart, rmin, rmax,
                rho_func, nparams, initial_params,
                lower_bounds, upper_bounds,
                xtol, ftol, maxeval,
                optimizer_type, min_density,
                out_params, out_loss, out_converged, out_neval);
}
"""

    # Combine: headers -> rho_func -> implementations -> wrapper
    # Inject SIMPSON_N before loss.h to override the default
    full_c_source = f"""
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>
#include <nlopt.h>

/* Simpson integration grid size (injected from Python config) */
#define SIMPSON_N {simpson_n}

/* ===== loss.h ===== */
{loss_h}

/* ===== optimizer.h ===== */
{optimizer_h}

/* ===== Auto-generated rho_func ===== */
{rho_func_c}

/* ===== loss.c ===== */
{loss_c}

/* ===== optimizer.c ===== */
{optimizer_c}

/* ===== Wrapper ===== */
{wrapper_c}
"""

    # cffi definitions
    cdef = """
    void fit_profile_wrapper(double* bin_counts, double* bin_positions,
                             int nbin,
                             int npart, double rmin, double rmax,
                             int nparams, double* initial_params,
                             double* lower_bounds, double* upper_bounds,
                             double xtol, double ftol, int maxeval,
                             int optimizer_type, double min_density,
                             double* out_params, double* out_loss,
                             int* out_converged, int* out_neval);
    """

    # Compile with cffi
    ffi = FFI()
    ffi.cdef(cdef)

    include_dirs, library_dirs = _get_nlopt_paths()

    verify_kwargs = dict(
        libraries=["m", "nlopt"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        extra_compile_args=["-O3", "-ffast-math", "-march=native"],
    )
    if CFFI_TMPDIR is not None:
        verify_kwargs["tmpdir"] = str(CFFI_TMPDIR)
    lib = ffi.verify(full_c_source, **verify_kwargs)

    # Create Python wrapper
    def fit(bin_counts, bin_positions, rmin, rmax,
            initial_params=None, lower_bounds=None, upper_bounds=None,
            xtol=1e-6, ftol=1e-6, maxeval=1000, optimizer='neldermead',
            min_density=1e-100):
        """
        Fit density profile to binned data. Returns dict with loss, params,
        converged, neval.
        """
        bin_counts = np.ascontiguousarray(bin_counts, dtype=np.float64)
        bin_positions = np.ascontiguousarray(bin_positions, dtype=np.float64)
        nbin = len(bin_counts)
        npart = int(np.sum(bin_counts))

        if initial_params is None:
            initial_params = np.ones(nparams, dtype=np.float64)
            initial_params[0] = np.sqrt(bin_positions[0] * bin_positions[-1])
        else:
            initial_params = np.ascontiguousarray(
                initial_params, dtype=np.float64)

        if lower_bounds is None:
            lower_bounds = np.array(
                [rmin / 4] + [-100.0] * nfree, dtype=np.float64)
        else:
            lower_bounds = np.ascontiguousarray(
                lower_bounds, dtype=np.float64)

        if upper_bounds is None:
            upper_bounds = np.array(
                [10 * rmax] + [100.0] * nfree, dtype=np.float64)
        else:
            upper_bounds = np.ascontiguousarray(
                upper_bounds, dtype=np.float64)

        out_params = np.zeros(nparams, dtype=np.float64)
        out_loss = np.zeros(1, dtype=np.float64)
        out_converged = np.zeros(1, dtype=np.int32)
        out_neval = np.zeros(1, dtype=np.int32)

        # Convert optimizer string to int
        optimizer_type = OPTIMIZER_MAP.get(optimizer.lower(), 0)

        lib.fit_profile_wrapper(
            ffi.cast("double*", bin_counts.ctypes.data),
            ffi.cast("double*", bin_positions.ctypes.data),
            nbin, npart, rmin, rmax, nparams,
            ffi.cast("double*", initial_params.ctypes.data),
            ffi.cast("double*", lower_bounds.ctypes.data),
            ffi.cast("double*", upper_bounds.ctypes.data),
            xtol, ftol, maxeval,
            optimizer_type, min_density,
            ffi.cast("double*", out_params.ctypes.data),
            ffi.cast("double*", out_loss.ctypes.data),
            ffi.cast("int*", out_converged.ctypes.data),
            ffi.cast("int*", out_neval.ctypes.data),
        )

        return {
            'loss': out_loss[0],
            'params': out_params,
            'converged': bool(out_converged[0]),
            'neval': out_neval[0],
        }

    def fit_with_restarts(bin_counts, bin_positions, rmin, rmax,
                          max_restarts=50, nconv_required=5,
                          conv_rtol=1e-3, conv_atol=10,
                          param_bounds=None,
                          Rs_lower_factor=0.25, Rs_upper_factor=10.0,
                          a_lower=-500.0, a_upper=500.0,
                          xtol=1e-6, ftol=1e-6, maxeval=5000, seed=None,
                          optimizer='sbplx', min_density=1e-100):
        """
        Fit with multiple restarts using Latin Hypercube Sampling.

        Uses LHS for better coverage of parameter space. Stops early if
        optimizer converges to the same minimum `nconv_required` times.
        Returns dict with loss, loss_all, params, converged, confirmed, neval,
        nrestart_used, nconv, reject_reason.

        The `loss_all` field contains the best loss from all attempts,
        including those that failed (loss >= 1e29). This is useful for
        computing statistics across all halos, even those that didn't converge.
        """
        if param_bounds is None:
            param_bounds = [(rmin * Rs_lower_factor, rmax * Rs_upper_factor)]
            for i in range(nfree):
                # If parameter is wrapped in Abs(), only need positive values
                if i in abs_wrapped_params:
                    param_bounds.append((0.0, a_upper))
                else:
                    param_bounds.append((a_lower, a_upper))

        lower_bounds = np.array([b[0] for b in param_bounds], dtype=np.float64)
        upper_bounds = np.array([b[1] for b in param_bounds], dtype=np.float64)

        # Pre-generate all initial points using Latin Hypercube Sampling
        lhs_samples = _generate_lhs_samples(param_bounds, max_restarts, seed)

        best_loss = np.inf
        best_loss_all = np.inf  # Track best loss including failed attempts
        best_params = None
        best_converged = False
        total_neval = 0
        nconv = 0  # Number of times converged to current best
        nrestart_used = 0

        for restart_idx in range(max_restarts):
            nrestart_used += 1

            # Use pre-generated LHS sample
            initial_params = lhs_samples[restart_idx]

            result = fit(bin_counts, bin_positions, rmin, rmax,
                         initial_params=initial_params,
                         lower_bounds=lower_bounds,
                         upper_bounds=upper_bounds,
                         xtol=xtol, ftol=ftol, maxeval=maxeval,
                         optimizer=optimizer, min_density=min_density)

            total_neval += result['neval']

            # Skip if optimizer failed (loss is huge)
            if result['loss'] >= 1e29:
                continue

            # Track best loss from all attempts (only valid ones, loss < 1e29)
            if result['loss'] < best_loss_all:
                best_loss_all = result['loss']

            # Convergence tolerance: atol + rtol * |best_loss|
            tol = conv_atol + conv_rtol * abs(best_loss)

            # Found significantly better minimum? Reset convergence counter
            if result['loss'] < best_loss - tol:
                nconv = 0

            # Update best if this is better
            if result['loss'] < best_loss:
                best_loss = result['loss']
                best_params = result['params'].copy()
                best_converged = result['converged']

            # Converged to same minimum (within tolerance)?
            if abs(result['loss'] - best_loss) <= tol:
                nconv += 1

            # Early stopping: converged enough times to the same minimum
            if nconv >= nconv_required:
                break

        # Reject fits with negative loss (numerical issues)
        reject_reason = None
        if best_loss < 0:
            reject_reason = "negative_loss"
            best_params = None

        return {
            'loss': best_loss,
            'loss_all': best_loss_all,  # Best loss including failed attempts
            'params': best_params,
            'converged': best_converged,
            'confirmed': nconv >= nconv_required,
            'neval': total_neval,
            'nrestart_used': nrestart_used,
            'nconv': nconv,
            'reject_reason': reject_reason,
        }

    # Attach metadata
    fit.expr_str = expr_str
    fit.nparams = nparams
    fit.nfree = nfree
    fit.simpson_n = simpson_n
    fit.abs_wrapped_params = abs_wrapped_params
    fit.fit_with_restarts = fit_with_restarts

    return fit


###############################################################################
#                         Nested optimization                                 #
###############################################################################


def compile_nested_fitter(expr_str, parser=None, simpson_n=512):
    """
    JIT compile a nested fitter for global + local parameter optimization.

    The nested optimizer fits multiple halos simultaneously, where some
    parameters are shared (global) and others are per-halo (local).

    Parameters
    ----------
    expr_str : str
        Expression for the density profile (e.g., "1/(x*(1+x)**a0)").
    parser : SympyParser, optional
        Parser instance for expression parsing.
    simpson_n : int, optional
        Number of points for Simpson integration. Default: 512.

    Returns
    -------
    NestedFitter
        Object with fit() method for nested optimization.
    """
    if parser is None:
        parser = SympyParser()

    if simpson_n % 2 != 0:
        raise ValueError(f"simpson_n must be even, got {simpson_n}")

    # Parse expression and count parameters
    expr = parser.parse(expr_str)
    nfree = parser.count_free(expr)
    nparams = 1 + nfree  # Rs + free parameters

    # Generate C code for density function
    expr_substituted = expr.subs(parser._x, parser._r / parser._Rs)
    c_expr = ccode(expr_substituted)

    # Read static C files
    loss_h = _read_csrc('loss.h')
    loss_c = _read_csrc('loss.c')
    nested_h = _read_csrc('nested_optimizer.h')
    nested_c = _read_csrc('nested_optimizer.c')

    # Generate density function C code
    rho_func_c = f"""
/* Auto-generated density function */
/* Expression: {expr_str} */
static inline __attribute__((always_inline))
double rho_func(double r, double Rs,
                double a0, double a1, double a2, double a3) {{
    (void)a0; (void)a1; (void)a2; (void)a3;
    return {c_expr};
}}
"""

    # Combine all C source
    full_c_source = f"""
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <nlopt.h>

#define SIMPSON_N {simpson_n}

/* ===== loss.h ===== */
{loss_h}

/* ===== nested_optimizer.h ===== */
{nested_h}

/* ===== Auto-generated rho_func ===== */
{rho_func_c}

/* ===== loss.c ===== */
{loss_c}

/* ===== nested_optimizer.c ===== */
{nested_c}

/* ===== Python interface wrapper ===== */
void fit_nested_wrapper(
    /* Halo data arrays (nhalos x nbin) */
    double* all_bin_counts,
    double* all_bin_positions,
    int* all_nbin,
    int* all_npart,
    double* all_rmin,
    double* all_rmax,
    int nhalos,

    /* Parameter configuration */
    int nparams_total,
    int* param_is_global,

    /* Bounds */
    double* lower_bounds,
    double* upper_bounds,

    /* Initial guesses */
    double* initial_global,
    double* initial_local,  /* nhalos x n_local, can be NULL */

    /* Tolerances */
    double xtol,
    double ftol,
    int inner_maxeval,
    int outer_maxeval,

    /* Inner restart settings */
    int inner_max_restarts,
    int inner_nconv_required,
    double inner_conv_rtol,
    double inner_conv_atol,

    /* Outer restart settings */
    int outer_max_restarts,
    int outer_nconv_required,
    double outer_conv_rtol,
    double outer_conv_atol,

    /* Parallelization */
    int nthreads,

    /* Numerical stability */
    double min_density,

    int verbose,

    /* Outputs */
    double* out_global_params,
    double* out_local_params,   /* nhalos x n_local */
    double* out_halo_losses,    /* nhalos */
    double* out_total_loss,
    int* out_converged,
    int* out_outer_neval,
    int* out_total_inner_neval
) {{
    /* Setup halo data array */
    HaloData* halos = (HaloData*)malloc(nhalos * sizeof(HaloData));

    int max_nbin = 0;
    for (int h = 0; h < nhalos; h++) {{
        if (all_nbin[h] > max_nbin) max_nbin = all_nbin[h];
    }}

    /* Point each halo to its slice of the data arrays */
    for (int h = 0; h < nhalos; h++) {{
        halos[h].bin_counts = &all_bin_counts[h * max_nbin];
        halos[h].bin_positions = &all_bin_positions[h * max_nbin];
        halos[h].nbin = all_nbin[h];
        halos[h].npart = all_npart[h];
        halos[h].rmin = all_rmin[h];
        halos[h].rmax = all_rmax[h];
    }}

    /* Initialize config */
    NestedOptConfig config;
    nested_opt_init(&config, halos, nhalos, rho_func, nparams_total,
                    param_is_global);
    nested_opt_set_bounds(&config, lower_bounds, upper_bounds);
    nested_opt_set_tolerances(&config, xtol, ftol,
                              inner_maxeval, outer_maxeval);
    nested_opt_set_inner_restarts(&config, inner_max_restarts,
                                  inner_nconv_required,
                                  inner_conv_rtol, inner_conv_atol);
    nested_opt_set_outer_restarts(&config, outer_max_restarts,
                                  outer_nconv_required,
                                  outer_conv_rtol, outer_conv_atol);
    config.nthreads = nthreads;
    config.min_density = min_density;
    config.verbose = verbose;

    /* Run nested optimization */
    NestedOptResult result;
    fit_nested_profile(&config, initial_global, initial_local, &result);

    /* Copy results */
    for (int i = 0; i < config.n_global; i++) {{
        out_global_params[i] = result.global_params[i];
    }}

    int n_local = config.n_local;
    for (int h = 0; h < nhalos; h++) {{
        for (int i = 0; i < n_local; i++) {{
            int idx = h * n_local + i;
            out_local_params[idx] = result.local_params[idx];
        }}
        out_halo_losses[h] = result.halo_losses[h];
    }}

    *out_total_loss = result.total_loss;
    *out_converged = result.converged;
    *out_outer_neval = result.outer_neval;
    *out_total_inner_neval = result.total_inner_neval;

    /* Cleanup */
    nested_opt_result_free(&result);
    free(halos);
}}
"""

    # CFFI definitions
    cdef = """
    void fit_nested_wrapper(
        double* all_bin_counts,
        double* all_bin_positions,
        int* all_nbin,
        int* all_npart,
        double* all_rmin,
        double* all_rmax,
        int nhalos,
        int nparams_total,
        int* param_is_global,
        double* lower_bounds,
        double* upper_bounds,
        double* initial_global,
        double* initial_local,
        double xtol,
        double ftol,
        int inner_maxeval,
        int outer_maxeval,
        int inner_max_restarts,
        int inner_nconv_required,
        double inner_conv_rtol,
        double inner_conv_atol,
        int outer_max_restarts,
        int outer_nconv_required,
        double outer_conv_rtol,
        double outer_conv_atol,
        int nthreads,
        double min_density,
        int verbose,
        double* out_global_params,
        double* out_local_params,
        double* out_halo_losses,
        double* out_total_loss,
        int* out_converged,
        int* out_outer_neval,
        int* out_total_inner_neval
    );
    """

    # Compile
    ffi = FFI()
    ffi.cdef(cdef)

    include_dirs, library_dirs = _get_nlopt_paths()

    # OpenMP flags depend on platform
    if sys.platform == "darwin":
        # macOS with clang needs special flags and libomp paths
        # libomp is keg-only, so we need explicit paths
        omp_compile = ["-Xpreprocessor", "-fopenmp",
                       "-I/opt/homebrew/opt/libomp/include"]
        omp_link = ["-L/opt/homebrew/opt/libomp/lib", "-lomp"]
    else:
        # Linux with gcc
        omp_compile = ["-fopenmp"]
        omp_link = ["-fopenmp"]

    verify_kwargs = dict(
        libraries=["m", "nlopt"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        extra_compile_args=["-O3", "-ffast-math", "-march=native"]
        + omp_compile,
        extra_link_args=omp_link,
    )
    if CFFI_TMPDIR is not None:
        verify_kwargs["tmpdir"] = str(CFFI_TMPDIR)
    lib = ffi.verify(full_c_source, **verify_kwargs)

    class NestedFitter:
        """Nested optimizer for global + local parameter fitting."""

        def __init__(self):
            self.expr_str = expr_str
            self.nparams = nparams
            self.nfree = nfree
            self.simpson_n = simpson_n

        def fit(self, binned_data, param_is_global,
                Rs_lower_factor=0.25, Rs_upper_factor=10.0,
                a_lower=-500.0, a_upper=500.0,
                initial_global=None, initial_local=None,
                xtol=1e-6, ftol=1e-6,
                inner_maxeval=500, outer_maxeval=200,
                inner_max_restarts=100, inner_nconv_required=10,
                inner_conv_rtol=1e-3, inner_conv_atol=10.0,
                outer_max_restarts=5, outer_nconv_required=2,
                outer_conv_rtol=1e-3, outer_conv_atol=10.0,
                nthreads=0, min_density=1e-100, verbose=False):
            """
            Fit multiple halos with global + local parameters.

            Parameters
            ----------
            binned_data : dict
                Output from HaloCollection.bin() with keys:
                'bin_counts', 'bin_positions', 'rmin', 'rmax', 'npart'
            param_is_global : list of bool
                Length nparams. True if parameter is global (shared).
                Index 0 = Rs, 1 = a0, 2 = a1, etc.
            Rs_lower_factor : float
                Lower bound for Rs = factor * min(rmin). Default: 0.25.
            Rs_upper_factor : float
                Upper bound for Rs = factor * max(rmax). Default: 10.0.
            a_lower, a_upper : float
                Bounds for free parameters a0, a1, etc. Default: [-500, 500].
            initial_global : array-like, optional
                Initial guess for global params.
            initial_local : array-like, optional
                Initial guess for local params (nhalos x n_local).
            xtol, ftol : float
                Optimization tolerances.
            inner_maxeval, outer_maxeval : int
                Max evaluations for inner/outer optimization.
            inner_max_restarts : int
                Max restarts per halo for inner optimization. Default: 100.
            inner_nconv_required : int
                Inner convergences to same minimum required. Default: 10.
            inner_conv_rtol, inner_conv_atol : float
                Inner convergence tolerances (atol + rtol * |loss|).
            outer_max_restarts : int
                Max restarts for outer optimization. Default: 5.
            outer_nconv_required : int
                Outer convergences to same minimum required. Default: 2.
            outer_conv_rtol, outer_conv_atol : float
                Outer convergence tolerances (atol + rtol * |loss|).
            nthreads : int
                Number of OpenMP threads for parallel halo fitting.
                0 = use all available cores (default).
            min_density : float
                Minimum allowed density value (underflow protection).
                Rejects parameter combinations that produce densities below
                this threshold. Default: 1e-100.
            verbose : bool
                Print progress.

            Returns
            -------
            dict with keys:
                global_params : ndarray of global parameter values
                local_params : ndarray (nhalos x n_local) of local params
                halo_losses : ndarray of per-halo losses
                total_loss : float
                converged : bool
                outer_neval : int
                total_inner_neval : int
            """
            bin_counts = binned_data['bin_counts']
            bin_positions = binned_data['bin_positions']
            rmin = binned_data['rmin']
            rmax = binned_data['rmax']
            npart = binned_data['npart']

            nhalos = len(bin_counts)
            max_nbin = bin_counts.shape[1]

            # Convert param_is_global to int array
            param_is_global = np.asarray(param_is_global, dtype=np.int32)
            if len(param_is_global) < 5:
                param_is_global = np.pad(
                    param_is_global, (0, 5 - len(param_is_global)))

            n_global = int(param_is_global[:nparams].sum())
            n_local = nparams - n_global

            # Compute bounds same as fit_with_restarts
            Rs_lower = Rs_lower_factor * rmin.min()
            Rs_upper = Rs_upper_factor * rmax.max()
            lower_bounds = np.array(
                [Rs_lower, a_lower, a_lower, a_lower, a_lower],
                dtype=np.float64)
            upper_bounds = np.array(
                [Rs_upper, a_upper, a_upper, a_upper, a_upper],
                dtype=np.float64)

            # Default initial global
            if initial_global is None:
                initial_global = np.zeros(n_global, dtype=np.float64)
                # Set to midpoint of bounds for global params
                g_idx = 0
                for i in range(nparams):
                    if param_is_global[i]:
                        initial_global[g_idx] = 0.5 * (
                            lower_bounds[i] + upper_bounds[i])
                        g_idx += 1
            else:
                initial_global = np.ascontiguousarray(
                    initial_global, dtype=np.float64)

            # Prepare data arrays
            all_bin_counts = np.ascontiguousarray(
                bin_counts, dtype=np.float64)
            all_bin_positions = np.ascontiguousarray(
                bin_positions, dtype=np.float64)
            all_nbin = np.ascontiguousarray(
                [max_nbin] * nhalos, dtype=np.int32)
            all_npart = np.ascontiguousarray(npart, dtype=np.int32)
            all_rmin = np.ascontiguousarray(rmin, dtype=np.float64)
            all_rmax = np.ascontiguousarray(rmax, dtype=np.float64)

            # Prepare initial local (or NULL)
            if initial_local is not None:
                initial_local = np.ascontiguousarray(
                    initial_local, dtype=np.float64)
                initial_local_ptr = ffi.cast(
                    "double*", initial_local.ctypes.data)
            else:
                initial_local_ptr = ffi.NULL

            # Output arrays
            out_global = np.zeros(n_global, dtype=np.float64)
            out_local = np.zeros((nhalos, n_local), dtype=np.float64)
            out_halo_losses = np.zeros(nhalos, dtype=np.float64)
            out_total_loss = np.zeros(1, dtype=np.float64)
            out_converged = np.zeros(1, dtype=np.int32)
            out_outer_neval = np.zeros(1, dtype=np.int32)
            out_total_inner_neval = np.zeros(1, dtype=np.int32)

            # Call C function
            lib.fit_nested_wrapper(
                ffi.cast("double*", all_bin_counts.ctypes.data),
                ffi.cast("double*", all_bin_positions.ctypes.data),
                ffi.cast("int*", all_nbin.ctypes.data),
                ffi.cast("int*", all_npart.ctypes.data),
                ffi.cast("double*", all_rmin.ctypes.data),
                ffi.cast("double*", all_rmax.ctypes.data),
                nhalos,
                nparams,
                ffi.cast("int*", param_is_global.ctypes.data),
                ffi.cast("double*", lower_bounds.ctypes.data),
                ffi.cast("double*", upper_bounds.ctypes.data),
                ffi.cast("double*", initial_global.ctypes.data),
                initial_local_ptr,
                xtol, ftol,
                inner_maxeval, outer_maxeval,
                inner_max_restarts, inner_nconv_required,
                inner_conv_rtol, inner_conv_atol,
                outer_max_restarts, outer_nconv_required,
                outer_conv_rtol, outer_conv_atol,
                nthreads,
                min_density,
                1 if verbose else 0,
                ffi.cast("double*", out_global.ctypes.data),
                ffi.cast("double*", out_local.ctypes.data),
                ffi.cast("double*", out_halo_losses.ctypes.data),
                ffi.cast("double*", out_total_loss.ctypes.data),
                ffi.cast("int*", out_converged.ctypes.data),
                ffi.cast("int*", out_outer_neval.ctypes.data),
                ffi.cast("int*", out_total_inner_neval.ctypes.data),
            )

            # Build param names for output
            global_param_names = []
            local_param_names = []
            param_names = ['Rs'] + [f'a{i}' for i in range(nfree)]
            for i in range(nparams):
                if param_is_global[i]:
                    global_param_names.append(param_names[i])
                else:
                    local_param_names.append(param_names[i])

            return {
                'global_params': out_global,
                'global_param_names': global_param_names,
                'local_params': out_local,
                'local_param_names': local_param_names,
                'halo_losses': out_halo_losses,
                'total_loss': out_total_loss[0],
                'converged': bool(out_converged[0]),
                'outer_neval': out_outer_neval[0],
                'total_inner_neval': out_total_inner_neval[0],
            }

    return NestedFitter()


###############################################################################
#                      Convenience NFW fitting function                       #
###############################################################################


# Cache for compiled NFW fitter
_nfw_fitter_cache = {}


def fit_nfw(radii, nbin=50, rmin=None, rmax=None, log_bins=True,
            max_restarts=50, nconv_required=5, seed=None, simpson_n=512,
            **kwargs):
    """
    Fit NFW profile to particle radii.

    The NFW density profile is: rho(r) = rho_s / ((r/r_s) * (1 + r/r_s)^2)

    Parameters
    ----------
    radii : array-like
        Particle radii from halo center.
    nbin : int, optional
        Number of radial bins. Default: 50.
    rmin, rmax : float, optional
        Radial range for binning. Default: min/max of radii.
    log_bins : bool, optional
        Use logarithmic bins. Default: True.
    max_restarts : int, optional
        Maximum optimizer restarts. Default: 50.
    nconv_required : int, optional
        Number of convergences to same minimum required. Default: 5.
    seed : int, optional
        Random seed for reproducibility.
    simpson_n : int, optional
        Number of points for Simpson integration. Default: 512.
    **kwargs
        Additional arguments passed to fit_with_restarts().

    Returns
    -------
    dict with keys:
        rs : float
            Scale radius (same units as input radii).
        loss : float
            Best-fit loss value.
        converged : bool
            Whether optimizer converged.
        confirmed : bool
            Whether minimum was confirmed by multiple restarts.
        params : ndarray
            Raw parameters [Rs].
        neval : int
            Total number of function evaluations.
    """
    radii = np.asarray(radii)

    if rmin is None:
        rmin = radii.min()
    if rmax is None:
        rmax = radii.max()

    # Bin the radii
    if log_bins:
        bin_edges = np.logspace(np.log10(rmin), np.log10(rmax), nbin + 1)
    else:
        bin_edges = np.linspace(rmin, rmax, nbin + 1)

    bin_counts, _ = np.histogram(radii, bins=bin_edges)

    # Bin centers (geometric mean for log bins, arithmetic for linear)
    if log_bins:
        bin_positions = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    else:
        bin_positions = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Get or compile NFW fitter
    cache_key = simpson_n
    if cache_key not in _nfw_fitter_cache:
        # NFW: rho = 1 / (x * (1 + x)^2) where x = r/Rs
        _nfw_fitter_cache[cache_key] = compile_fitter(
            "1/(x*(1+x)**2)", simpson_n=simpson_n)

    fitter = _nfw_fitter_cache[cache_key]

    # Fit
    result = fitter.fit_with_restarts(
        bin_counts.astype(np.float64),
        bin_positions,
        rmin, rmax,
        max_restarts=max_restarts,
        nconv_required=nconv_required,
        seed=seed,
        **kwargs
    )

    # Extract scale radius
    rs = result['params'][0] if result['params'] is not None else np.nan

    return {
        'rs': rs,
        'loss': result['loss'],
        'converged': result['converged'],
        'confirmed': result['confirmed'],
        'params': result['params'],
        'neval': result['neval'],
        'nrestart_used': result['nrestart_used'],
    }
