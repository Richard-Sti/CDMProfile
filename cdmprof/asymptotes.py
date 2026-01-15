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

# Small positive value used to approximate x -> 0+ limit numerically
LIMIT_ZERO_EPSILON = 1e-16

# Pattern to detect parameter-dependent limits (a0, a1, a2, a3)
_PARAM_PATTERN = re.compile(r'\ba[0-3]\b')

# Infinity string representations
_POS_INFINITY = {"oo", "zoo", "inf"}
_NEG_INFINITY = {"-oo", "-inf"}
_ALL_INFINITY = _POS_INFINITY | _NEG_INFINITY


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
    Load asymptotes file (format: `idx limit equation`) and categorize each.
    Returns dict mapping func_idx -> (category, limit_expr_str).
    """
    asymp_path = Path(asymp_path)
    if not asymp_path.exists():
        return {}

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

            if _PARAM_PATTERN.search(limit_str):
                asymp_dict[idx] = ("param_dep", limit_str)
                continue

            # Categorize based on limit type
            if limit_type == "inf":
                # For x->inf: need limit = 0
                if limit_str in _ALL_INFINITY:
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
                elif limit_str in _NEG_INFINITY:
                    asymp_dict[idx] = ("skip_neg", limit_str)
                elif limit_str in _POS_INFINITY:
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


def check_asymptote(value, limit_type, tol=1e-6):
    """
    Check if asymptote is acceptable based on `limit_type`: ~0 for x->inf, >0
    for x->0+.
    """
    if value is None:
        return False
    if limit_type == "inf":
        return abs(value) < tol
    else:
        return value > 0


def _numerical_limit(expr, x, direction):
    """Compute limit of `expr` w.r.t. `x`: '0+' for x->0+ or 'inf' for x->inf."""
    try:
        if direction == '0+':
            result = limit(expr, x, LIMIT_ZERO_EPSILON, '+')
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
        lim_zero = _numerical_limit(expr, x, '0+')
        lim_inf = _numerical_limit(expr, x, 'inf')
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
    Load both asymptote files and categorize functions.
    Returns (skip_dict, asymp_inf_param_dep, asymp_zero_param_dep, asymp_unknown).
    """
    skip_dict = {}
    param_dep = {"inf": {}, "zero": {}}
    asymp_unknown = set()

    for limit_type, path in [("inf", asymp_inf_path), ("zero", asymp_zero_path)]:
        asymp_data = load_asymptotes(path, limit_type)
        for idx, (category, limit_str) in asymp_data.items():
            if category.startswith("skip_"):
                if idx not in skip_dict:  # Don't override if already skipped
                    skip_dict[idx] = f"asymp_{limit_type}_{category}"
            elif category == "param_dep":
                param_dep[limit_type][idx] = limit_str
            elif category == "unknown":
                asymp_unknown.add(idx)

    return skip_dict, param_dep["inf"], param_dep["zero"], asymp_unknown
