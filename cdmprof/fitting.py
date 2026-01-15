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
                         int optimizer_type,
                         double* out_params, double* out_loss,
                         int* out_converged, int* out_neval) {
    fit_profile(bin_counts, bin_positions, nbin, npart, rmin, rmax,
                rho_func, nparams, initial_params,
                lower_bounds, upper_bounds,
                xtol, ftol, maxeval,
                optimizer_type,
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
                             int optimizer_type,
                             double* out_params, double* out_loss,
                             int* out_converged, int* out_neval);
    """

    # Compile with cffi
    ffi = FFI()
    ffi.cdef(cdef)

    include_dirs, library_dirs = _get_nlopt_paths()

    lib = ffi.verify(
        full_c_source,
        libraries=["m", "nlopt"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        extra_compile_args=["-O3", "-ffast-math", "-march=native"],
    )

    # Create Python wrapper
    def fit(bin_counts, bin_positions, rmin, rmax,
            initial_params=None, lower_bounds=None, upper_bounds=None,
            xtol=1e-6, ftol=1e-6, maxeval=1000, optimizer='neldermead'):
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
            optimizer_type,
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
                          optimizer='sbplx'):
        """
        Fit with multiple restarts using Latin Hypercube Sampling.

        Uses LHS for better coverage of parameter space. Stops early if
        optimizer converges to the same minimum `nconv_required` times.
        Returns dict with loss, params, converged, confirmed, neval,
        nrestart_used, nconv, reject_reason.
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
                         optimizer=optimizer)

            total_neval += result['neval']

            # Skip if optimizer failed (loss is huge)
            if result['loss'] >= 1e29:
                continue

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
