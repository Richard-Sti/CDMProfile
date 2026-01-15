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
Asymptote handling for density profiles.
"""
import multiprocessing as mp
import re
from pathlib import Path

import numpy as np
from sympy import N, limit, oo, symbols, sympify

from .symbolic import SympyParser


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

        lim_zero = _numerical_limit(expr_numerical, x, '0+')
        lim_inf = _numerical_limit(expr_numerical, x, 'inf')

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
