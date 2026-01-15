# Copyright (C) 2024 Richard Stiskalek
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
from sympy import Abs, ccode, simplify, symbols, sympify

from .symbolic import SympyParser

# Path to C source files
CSRC_DIR = Path(__file__).parent / "csrc"


def _has_normalization_only_param(expr_str):
    """
    Check if any parameter is just an overall normalization factor.

    A parameter `a` is normalization-only if f = a * g(x) or f = Abs(a) * g(x)
    where g doesn't contain `a`. We check this by seeing if f/a or f/Abs(a)
    is independent of a.

    Parameters
    ----------
    expr_str : str
        Expression string.

    Returns
    -------
    bool
        True if any parameter is normalization-only.
    """
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
    """
    Detect parameters that appear exactly wrapped in Abs().

    Only detects exact patterns like Abs(a0), NOT Abs(a0 + 1).

    Parameters
    ----------
    expr_str : str
        Expression string.
    parser : SympyParser, optional
        Parser instance. If None, creates a new one.

    Returns
    -------
    set
        Set of parameter indices (0-3) that are wrapped in Abs().
        E.g., "Abs(a0) + a1" returns {0}
             "Abs(a0 + 1)" returns {} (not exact)
    """
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
    """
    Check if expression is invalid (infinity, NaN, trig functions, etc.).

    Parameters
    ----------
    expr_str : str
        Expression string.

    Returns
    -------
    bool
    """
    # Bad prefixes
    bad_prefixes = ["0", "zoo", "<class"]
    if any(expr_str.startswith(bp) for bp in bad_prefixes):
        return True

    # Bad substrings (infinity, NaN)
    bad_substrings = ["oo", "nan", "NaN"]
    if any(bs in expr_str for bs in bad_substrings):
        return True

    # Trigonometric functions (not suitable for density profiles)
    trigs = ["sin", "cos", "tan", "arcsin", "arccos", "atan"]
    if any(trig in expr_str for trig in trigs):
        return True

    # Check for normalization-only parameters
    if _has_normalization_only_param(expr_str):
        return True

    return False


def load_equations(filepath):
    """
    Load equations from a text file (one per line), skipping empty lines.

    Parameters
    ----------
    filepath : str or Path
        Path to equations file.

    Returns
    -------
    list of str
        List of equation strings.
    """
    equations = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                equations.append(line)
    return equations


###############################################################################
#                         C code compilation                                  #
###############################################################################


def _read_csrc(name):
    """Read a C source file, stripping local #include directives."""
    with open(CSRC_DIR / name, 'r') as f:
        content = f.read()
    # Remove local includes (they'll be inlined)
    lines = []
    for line in content.split('\n'):
        if line.strip().startswith('#include "'):
            continue
        lines.append(line)
    return '\n'.join(lines)


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

    Parameters
    ----------
    expr_str : str
        Symbolic expression for the density profile
        (e.g., "1 / (x * (1 + x)**a0)")
    parser : SympyParser, optional
        Parser instance. If None, creates a new one.
    simpson_n : int, optional
        Number of intervals for Simpson integration. Must be even.
        Higher values give more accuracy but slower computation.
        Default: 512.

    Returns
    -------
    callable
        Function with signature:
        fit(bin_counts, bin_positions, rmin, rmax, ...) -> dict

        Also has fit.fit_with_restarts(...) for multi-restart optimization.
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

    # Optimizer type mapping
    OPTIMIZER_MAP = {
        'neldermead': 0,
        'nelder-mead': 0,
        'bobyqa': 1,
    }

    # Create Python wrapper
    def fit(bin_counts, bin_positions, rmin, rmax,
            initial_params=None, lower_bounds=None, upper_bounds=None,
            xtol=1e-6, ftol=1e-6, maxeval=1000, optimizer='neldermead'):
        """
        Fit the density profile to binned halo data.

        Parameters
        ----------
        bin_counts : array-like, shape (nbin,)
            Particle counts per bin.
        bin_positions : array-like, shape (nbin,)
            Radial bin positions.
        rmin : float
            Minimum radius for mass integration.
        rmax : float
            Maximum radius for mass integration.
        initial_params : array-like, optional
            Initial parameter guess [Rs, a0, a1, ...].
        lower_bounds : array-like, optional
            Lower bounds [Rs_min, a0_min, ...].
        upper_bounds : array-like, optional
            Upper bounds [Rs_max, a0_max, ...].
        xtol : float, optional
            Relative tolerance on parameters.
        ftol : float, optional
            Relative tolerance on function value.
        maxeval : int, optional
            Maximum function evaluations.
        optimizer : str, optional
            Optimizer to use: 'neldermead' or 'bobyqa'. Default: 'neldermead'.

        Returns
        -------
        dict
            {'loss': float, 'params': array, 'converged': bool, 'neval': int}
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
                          optimizer='neldermead'):
        """
        Fit with multiple random restarts and early stopping.

        Runs up to `max_restarts` optimization attempts. Stops early if the
        optimizer converges to the same minimum `nconv_required` times without
        finding anything better.

        Parameters
        ----------
        bin_counts : array-like, shape (nbin,)
            Particle counts per bin.
        bin_positions : array-like, shape (nbin,)
            Radial bin positions.
        rmin : float
            Minimum radius for mass integration.
        rmax : float
            Maximum radius for mass integration.
        max_restarts : int, optional
            Maximum number of restarts. Default: 50.
        nconv_required : int, optional
            Number of convergences to same minimum required to stop early.
            Default: 5.
        conv_rtol : float, optional
            Relative tolerance for considering two losses as converged to
            the same minimum. Default: 1e-3.
        conv_atol : float, optional
            Absolute tolerance for considering two losses as converged to
            the same minimum. Default: 10.
        param_bounds : list of tuples, optional
            Bounds [(low, high), ...] for sampling and optimization.
            If None, uses Rs_lower/upper_factor and a_lower/upper.
        Rs_lower_factor : float, optional
            Rs_min = rmin * Rs_lower_factor. Default: 0.25.
        Rs_upper_factor : float, optional
            Rs_max = rmax * Rs_upper_factor. Default: 10.0.
        a_lower : float, optional
            Lower bound for a0-a3 parameters. Default: -500.0.
        a_upper : float, optional
            Upper bound for a0-a3 parameters. Default: 500.0.
        xtol : float, optional
            Relative tolerance on parameters.
        ftol : float, optional
            Relative tolerance on function value.
        maxeval : int, optional
            Maximum function evaluations per restart.
        seed : int, optional
            Random seed for reproducibility.
        optimizer : str, optional
            Optimizer to use: 'neldermead' or 'bobyqa'. Default: 'neldermead'.

        Returns
        -------
        dict
            loss : float - best loss found
            params : array or None - best parameters (None if rejected)
            converged : bool - whether best run's optimizer converged
            confirmed : bool - converged to same min nconv_required times
            neval : int - total function evaluations across all restarts
            nrestart_used : int - number of restarts before stopping
            nconv : int - times converged to the best minimum
            reject_reason : str or None - reason for rejection
        """
        rng = np.random.default_rng(seed)

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

        best_loss = np.inf
        best_params = None
        best_converged = False
        total_neval = 0
        nconv = 0  # Number of times converged to current best
        nrestart_used = 0

        for restart_idx in range(max_restarts):
            nrestart_used += 1

            # Generate initial parameters (uniform sampling)
            initial_params = np.zeros(nparams, dtype=np.float64)
            for i, (low, high) in enumerate(param_bounds):
                if i == 0:
                    # Rs: log-uniform sampling
                    log_val = rng.uniform(np.log(low), np.log(high))
                    initial_params[i] = np.exp(log_val)
                else:
                    initial_params[i] = rng.uniform(low, high)

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
    fit.abs_wrapped_params = abs_wrapped_params
    fit.fit_with_restarts = fit_with_restarts

    return fit
