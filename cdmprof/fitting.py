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
import multiprocessing as mp
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from cffi import FFI
from sympy import Abs, N, ccode, limit, oo, simplify, symbols, sympify

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
#                         Asymptote handling                                  #
###############################################################################


def _parse_limit_value(limit_str):
    """Try to parse a limit string as a numerical value."""
    try:
        return float(limit_str)
    except ValueError:
        try:
            return float(N(sympify(limit_str)))
        except Exception:
            return None


def load_asymptotes(asymp_path, limit_type):
    """
    Load asymptotes file and categorize each function.

    File format: `idx limit equation`

    Parameters
    ----------
    asymp_path : str or Path
        Path to asymptotes file.
    limit_type : str
        Either "inf" (x->inf, need limit=0) or "zero" (x->0+, need limit>0)

    Returns
    -------
    dict
        Dictionary mapping func_idx -> (category, limit_expr_str)
    """
    asymp_path = Path(asymp_path)
    if not asymp_path.exists():
        return {}

    param_pattern = re.compile(r'\ba[0-3]\b')
    asymp_dict = {}

    with open(asymp_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split(None, 2)
            if len(parts) < 2:
                continue

            idx = int(parts[0])
            limit_str = parts[1]

            # Handle special cases first
            if limit_str == "unknown":
                asymp_dict[idx] = ("unknown", limit_str)
                continue

            if param_pattern.search(limit_str):
                asymp_dict[idx] = ("param_dep", limit_str)
                continue

            # Categorize based on limit type
            if limit_type == "inf":
                # For x->inf: need limit = 0
                if limit_str in ("oo", "-oo", "zoo", "inf", "-inf"):
                    asymp_dict[idx] = ("skip_inf", limit_str)
                elif limit_str == "0":
                    asymp_dict[idx] = ("ok", limit_str)
                else:
                    val = _parse_limit_value(limit_str)
                    if val is None:
                        asymp_dict[idx] = ("unknown", limit_str)
                    elif np.isclose(val, 0):
                        asymp_dict[idx] = ("ok", limit_str)
                    else:
                        asymp_dict[idx] = ("skip_const", limit_str)
            else:
                # For x->0+: need limit > 0
                if limit_str == "0":
                    asymp_dict[idx] = ("skip_zero", limit_str)
                elif limit_str in ("-oo", "-inf"):
                    asymp_dict[idx] = ("skip_neg", limit_str)
                elif limit_str in ("oo", "zoo", "inf"):
                    asymp_dict[idx] = ("ok", limit_str)
                else:
                    val = _parse_limit_value(limit_str)
                    if val is None:
                        asymp_dict[idx] = ("unknown", limit_str)
                    elif val <= 0:
                        asymp_dict[idx] = ("skip_neg", limit_str)
                    else:
                        asymp_dict[idx] = ("ok", limit_str)

    return asymp_dict


def load_asymptotes_inf(asymp_path):
    """Load x->inf asymptotes. Wrapper for backwards compatibility."""
    return load_asymptotes(asymp_path, "inf")


def load_asymptotes_zero(asymp_path):
    """Load x->0+ asymptotes. Wrapper for backwards compatibility."""
    return load_asymptotes(asymp_path, "zero")


def evaluate_asymptote(limit_expr_str, params):
    """
    Evaluate a parameter-dependent asymptotic limit with fitted parameters.

    Parameters
    ----------
    limit_expr_str : str
        Symbolic expression for the limit (e.g., "a0 - 1").
    params : array-like
        Fitted parameters [Rs, a0, a1, a2, a3].

    Returns
    -------
    float or None
        Evaluated limit value, or None if evaluation fails.
    """
    try:
        param_symbols = [symbols(f'a{i}', real=True) for i in range(4)]

        local_dict = {f'a{i}': param_symbols[i] for i in range(4)}
        expr = sympify(limit_expr_str, locals=local_dict)

        # Substitute fitted values (params[0] is Rs, params[1:] are a0-a3)
        subs_dict = {}
        for i in range(4):
            if i + 1 < len(params):
                subs_dict[param_symbols[i]] = params[i + 1]
            else:
                subs_dict[param_symbols[i]] = 1.0

        result = expr.subs(subs_dict)
        return float(N(result))

    except Exception:
        return None


def check_asymptote_inf(value, tol=1e-6):
    """
    Check if an evaluated x->inf asymptote is acceptable (effectively zero).

    Parameters
    ----------
    value : float or None
        Evaluated asymptote value.
    tol : float
        Tolerance for considering a value as zero.

    Returns
    -------
    bool
        True if value is effectively zero, False otherwise.
    """
    if value is None:
        return False
    return abs(value) < tol


def check_asymptote_zero(value):
    """
    Check if an evaluated x->0+ asymptote is acceptable (positive).

    Parameters
    ----------
    value : float or None
        Evaluated asymptote value.

    Returns
    -------
    bool
        True if value is positive, False otherwise.
    """
    if value is None:
        return False
    return value > 0


def _numerical_limit(expr, x, direction):
    """
    Compute limit using sympy's limit function.

    Parameters
    ----------
    expr : sympy expression
        Expression with numerical coefficients.
    x : sympy symbol
        The variable.
    direction : str
        '0+' for x->0+ or 'inf' for x->inf.

    Returns
    -------
    float or None
        The limit value, or None if computation fails.
    """
    # print("This is 1")
    try:
        if direction == '0+':
            result = limit(expr, x, 1e-16, '+')
        else:
            result = limit(expr, x, oo)
        return float(result.evalf())
    except Exception:
        return None


def _compute_limit_worker(expr_str, result_queue):
    """Worker function for subprocess-based limit computation."""
    # print("this is B")
    try:
        x = symbols('x', positive=True)
        expr = sympify(expr_str, locals={'x': x})

        try:
            lim_zero = float(limit(expr, x, 1e-16, '+').evalf())
        except Exception:
            lim_zero = None

        try:
            lim_inf = float(limit(expr, x, oo).evalf())
        except Exception:
            lim_inf = None

        result_queue.put(('success', lim_zero, lim_inf))
    except Exception:
        result_queue.put(('error', None, None))


def compute_asymptotes_with_params(expr_str, params, round_decimals=5,
                                   timeout=0):
    """
    Compute asymptotes of an expression with concrete parameter values.

    Substitutes numerical values first, then computes the limits.
    Sympy automatically simplifies Abs() when values are known.

    Parameters
    ----------
    expr_str : str
        The density profile expression (e.g., "pow(Abs(a0 - x), a1)").
    params : array-like
        Fitted parameters [Rs, a0, a1, a2, a3].
    round_decimals : int, optional
        Round parameters to this many decimal places to avoid sympy
        creating huge rational expressions. Default is 5.
    timeout : int, optional
        Timeout in seconds. If 0, no timeout. Uses subprocess for MPI safety.

    Returns
    -------
    tuple
        (lim_zero, lim_inf) where each is float or None if computation fails.
    """
    # print(f"Computing asymptotes for expression: {expr_str}")
    # print(f"With parameters: {params}")
    try:
        parser = SympyParser()
        expr = parser.parse(expr_str)
        x = parser._x

        # Substitute numerical values using N() to force numerical evaluation
        subs_dict = {
            parser._free_params[i]: N(round(params[i + 1], round_decimals))
            for i in range(min(len(params) - 1, 4))
        }
        expr_numerical = expr.subs(subs_dict)

        # Use subprocess timeout if requested (MPI-safe)
        if timeout > 0:
            ctx = mp.get_context('spawn')
            result_queue = ctx.Queue()

            proc = ctx.Process(
                target=_compute_limit_worker,
                args=(str(expr_numerical), result_queue)
            )
            proc.start()
            proc.join(timeout=timeout)

            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1)
                if proc.is_alive():
                    proc.kill()
                    proc.join()
                return None, None

            try:
                status, lim_zero, lim_inf = result_queue.get_nowait()
                if status == 'success':
                    return lim_zero, lim_inf
                return None, None
            except Exception:
                return None, None

        # Try numerical evaluation first (much faster)
        # print(expr_numerical)
        # print("A")
        lim_zero = _numerical_limit(expr_numerical, x, '0+')
        # print("B")
        lim_inf = _numerical_limit(expr_numerical, x, 'inf')
        # print("C")

        return lim_zero, lim_inf

    except Exception:
        return None, None


def process_asymptotes(asymp_inf_path, asymp_zero_path):
    """
    Load and process both asymptote files, returning skip sets and info.

    Parameters
    ----------
    asymp_inf_path : str or Path
        Path to asymptotes_inf_{comp}.txt file.
    asymp_zero_path : str or Path
        Path to asymptotes_zero_{comp}.txt file.

    Returns
    -------
    tuple
        (skip_dict, asymp_inf_param_dep, asymp_zero_param_dep, asymp_unknown)
        - skip_dict: dict mapping func_idx -> reason for pre-fit skipping
        - asymp_inf_param_dep: dict mapping func_idx -> limit_expr for x->inf
        - asymp_zero_param_dep: dict mapping func_idx -> limit_expr for x->0+
        - asymp_unknown: set of func_idx needing post-fit asymptote computation
    """
    skip_dict = {}
    asymp_inf_param_dep = {}
    asymp_zero_param_dep = {}
    asymp_unknown = set()

    # Process x->inf asymptotes
    asymp_inf = load_asymptotes_inf(asymp_inf_path)
    for idx, (category, limit_str) in asymp_inf.items():
        if category.startswith("skip_"):
            skip_dict[idx] = f"asymp_inf_{category}"
        elif category == "param_dep":
            asymp_inf_param_dep[idx] = limit_str
        elif category == "unknown":
            asymp_unknown.add(idx)

    # Process x->0+ asymptotes
    asymp_zero = load_asymptotes_zero(asymp_zero_path)
    for idx, (category, limit_str) in asymp_zero.items():
        if category.startswith("skip_"):
            # Don't override if already skipped
            if idx not in skip_dict:
                skip_dict[idx] = f"asymp_zero_{category}"
        elif category == "param_dep":
            asymp_zero_param_dep[idx] = limit_str
        elif category == "unknown":
            asymp_unknown.add(idx)

    return skip_dict, asymp_inf_param_dep, asymp_zero_param_dep, asymp_unknown


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


def compile_fitter(expr_str, parser=None):
    """
    JIT compile a density function + loss + optimizer.

    Parameters
    ----------
    expr_str : str
        Symbolic expression for the density profile
        (e.g., "1 / (x * (1 + x)**a0)")
    parser : SympyParser, optional
        Parser instance. If None, creates a new one.

    Returns
    -------
    callable
        Function with signature:
        fit(bin_counts, bin_positions, rmin, rmax, ...) -> dict

        Also has fit.fit_with_restarts(...) for multi-restart optimization.
    """
    if parser is None:
        parser = SympyParser()

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
    optimizer_h = _read_csrc('optimizer.h')
    optimizer_c = _read_csrc('optimizer.c')

    # Generate density function C code (only dynamic part)
    rho_func_c = f"""
/* Auto-generated density function */
/* Expression: {expr_str} */
static double rho_func(double r, double Rs,
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
                         double* out_params, double* out_loss,
                         int* out_converged, int* out_neval) {
    fit_profile(bin_counts, bin_positions, nbin, npart, rmin, rmax,
                rho_func, nparams, initial_params,
                lower_bounds, upper_bounds,
                xtol, ftol, maxeval,
                out_params, out_loss, out_converged, out_neval);
}
"""

    # Combine: headers -> rho_func -> implementations -> wrapper
    full_c_source = f"""
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>
#include <nlopt.h>

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
    )

    # Create Python wrapper
    def fit(bin_counts, bin_positions, rmin, rmax,
            initial_params=None, lower_bounds=None, upper_bounds=None,
            xtol=1e-6, ftol=1e-6, maxeval=1000):
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

        lib.fit_profile_wrapper(
            ffi.cast("double*", bin_counts.ctypes.data),
            ffi.cast("double*", bin_positions.ctypes.data),
            nbin, npart, rmin, rmax, nparams,
            ffi.cast("double*", initial_params.ctypes.data),
            ffi.cast("double*", lower_bounds.ctypes.data),
            ffi.cast("double*", upper_bounds.ctypes.data),
            xtol, ftol, maxeval,
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
                          max_restarts=50, nconv_required=5, conv_rtol=1e-4,
                          param_bounds=None,
                          Rs_lower_factor=0.25, Rs_upper_factor=10.0,
                          a_lower=-500.0, a_upper=500.0,
                          xtol=1e-6, ftol=1e-6, maxeval=5000, seed=None):
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
            the same minimum. Default: 1e-4.
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
            for _ in range(nfree):
                param_bounds.append((a_lower, a_upper))

        lower_bounds = np.array([b[0] for b in param_bounds], dtype=np.float64)
        upper_bounds = np.array([b[1] for b in param_bounds], dtype=np.float64)

        best_loss = np.inf
        best_params = None
        best_converged = False
        total_neval = 0
        nconv = 0  # Number of times converged to current best
        nrestart_used = 0

        for _ in range(max_restarts):
            nrestart_used += 1

            # Generate random initial parameters
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
                         xtol=xtol, ftol=ftol, maxeval=maxeval)

            total_neval += result['neval']

            # Skip if optimizer failed (loss is huge)
            if result['loss'] >= 1e29:
                continue

            # Found significantly better minimum? Reset convergence counter
            if result['loss'] < best_loss * (1 - conv_rtol):
                nconv = 0

            # Update best if this is better
            if result['loss'] < best_loss:
                best_loss = result['loss']
                best_params = result['params'].copy()
                best_converged = result['converged']

            # Converged to same minimum (within relative tolerance)?
            if abs(result['loss'] - best_loss) <= conv_rtol * abs(best_loss):
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
    fit.fit_with_restarts = fit_with_restarts

    return fit
