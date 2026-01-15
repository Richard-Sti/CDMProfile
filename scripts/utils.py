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
Utility functions for the CDM profile analysis.
"""
import tomllib
from os.path import dirname, exists, join
from pathlib import Path

import h5py
import numpy as np

PROJECT_DIR = join(dirname(__file__), "..")
CONFIG_PATH = join(PROJECT_DIR, "config.toml")
LOCAL_CONFIG_PATH = join(PROJECT_DIR, "local_config.toml")

LOCAL_CONFIG_TEMPLATE = """
[path]
results = "/path/to/results"
data = "/path/to/data"
venv = "/path/to/venv"
""".strip()


def read_config():
    """
    Read the configuration file, merging with local_config.toml.

    Returns
    -------
    dict
    """
    if not exists(LOCAL_CONFIG_PATH):
        raise FileNotFoundError(
            f"Local config file not found: {LOCAL_CONFIG_PATH}\n"
            f"Please create it with the following contents:\n\n"
            f"{LOCAL_CONFIG_TEMPLATE}"
        )

    with open(CONFIG_PATH, "rb") as f:
        config = tomllib.load(f)

    with open(LOCAL_CONFIG_PATH, "rb") as f:
        local_config = tomllib.load(f)

    # Merge local_config into config (local takes precedence)
    for key, value in local_config.items():
        if key in config and isinstance(config[key], dict):
            config[key].update(value)
        else:
            config[key] = value

    return config


###############################################################################
#                      Display and summary functions                          #
###############################################################################


def categorize_failed_functions(output_path, equations, skip_dict=None):
    """
    Categorize functions that failed to produce results.

    Parameters
    ----------
    output_path : Path
        Path to merged HDF5 results file.
    equations : list of str
        List of equation strings.
    skip_dict : dict, optional
        Pre-computed skip reasons from generate_functions.py.

    Returns
    -------
    dict
        Keys: 'negative_loss', 'asymptote', 'normalization_only',
        'bad_function', 'asymp_prefit', 'other'.
    """
    output_path = Path(output_path)
    if skip_dict is None:
        skip_dict = {}

    categories = {
        'negative_loss': [],
        'asymptote': [],
        'normalization_only': [],
        'bad_function': [],
        'asymp_prefit': [],
        'other': [],
    }

    if not output_path.exists():
        return categories

    with h5py.File(output_path, 'r') as f:
        if 'func_idx' not in f:
            successful = set()
        else:
            successful = set(f['func_idx'][:].tolist())

        if 'negative_loss_func_idx' in f:
            negative_loss_funcs = set(f['negative_loss_func_idx'][:].tolist())
        else:
            negative_loss_funcs = set()

        if 'asymp_reject_func_idx' in f:
            asymp_reject_funcs = set(f['asymp_reject_func_idx'][:].tolist())
        else:
            asymp_reject_funcs = set()

    all_funcs = set(range(len(equations)))
    failed = all_funcs - successful

    for fidx in sorted(failed):
        eq = equations[fidx]
        if fidx in negative_loss_funcs:
            categories['negative_loss'].append((fidx, eq))
        elif fidx in asymp_reject_funcs:
            categories['asymptote'].append((fidx, eq))
        elif fidx in skip_dict:
            reason = skip_dict[fidx]
            if reason == 'normalization_only':
                categories['normalization_only'].append((fidx, eq))
            elif reason == 'bad_function':
                categories['bad_function'].append((fidx, eq))
            elif reason.startswith('asymp_'):
                categories['asymp_prefit'].append((fidx, eq))
            else:
                categories['other'].append((fidx, eq))
        else:
            categories['other'].append((fidx, eq))

    return categories


def _print_category(name, items, max_show=20):
    """Helper to print a category of failed functions."""
    if len(items) == 0:
        return
    print(f"\n{name}: {len(items)}")
    show_items = items[:max_show] if len(items) > max_show else items
    for fidx, eq in show_items:
        eq_short = eq[:50] + "..." if len(eq) > 50 else eq
        print(f"  {fidx}: {eq_short}")
    if len(items) > max_show:
        print(f"  ... and {len(items) - max_show} more")


def print_failed_functions(output_path, equations, skip_dict=None,
                           categories=None):
    """
    Print functions that failed to produce any results.

    Returns the categories dict for reuse by write_failed_to_files.
    """
    if categories is None:
        categories = categorize_failed_functions(output_path, equations,
                                                 skip_dict)

    n_failed = sum(len(v) for v in categories.values())
    n_total = len(equations)

    if n_failed == 0:
        print(f"\nAll {n_total} functions produced results.")
        return categories

    print(f"\n{'=' * 80}")
    print(f"SKIPPED/FAILED FUNCTIONS ({n_failed}/{n_total})")
    print("=" * 80)

    _print_category("Negative loss (numerical issues/singularities)",
                    categories['negative_loss'])
    _print_category("Asymptote validation failed", categories['asymptote'])
    _print_category("Normalization-only parameter (skipped)",
                    categories['normalization_only'])
    _print_category("Bad functions (nan/inf/trig, skipped)",
                    categories['bad_function'])
    _print_category("Pre-fit asymptote check failed (skipped)",
                    categories['asymp_prefit'])
    _print_category("Compilation/fit failures", categories['other'])

    print("=" * 80)
    print("")

    return categories


def print_asymptote_summary(n_param_dep, n_unknown, n_rejected, threshold_inf,
                            threshold_zero):
    """Print summary of post-fit asymptote validation."""
    n_total = n_param_dep + n_unknown
    if n_total == 0:
        return

    n_passed = n_total - n_rejected
    pct_passed = 100 * n_passed / n_total if n_total > 0 else 0
    pct_rejected = 100 * n_rejected / n_total if n_total > 0 else 0

    print(f"\n{'=' * 60}")
    print("POST-FIT ASYMPTOTE VALIDATION")
    print("=" * 60)
    print(f"Thresholds: lim(x->inf)~0 >= {100*threshold_inf:.0f}%, "
          f"lim(x->0+)>0 >= {100*threshold_zero:.0f}%")
    print("\nFunctions requiring post-fit check:")
    print(f"  Parameter-dependent:  {n_param_dep:>5}")
    print(f"  Unknown (recomputed): {n_unknown:>5}")
    print(f"  Total:                {n_total:>5}")
    print("\nResults:")
    print(f"  Passed:   {n_passed:>5} ({pct_passed:.1f}%)")
    print(f"  Rejected: {n_rejected:>5} ({pct_rejected:.1f}%)")
    print("=" * 60)


def compute_function_scores(output_path, npart_per_halo,
                            min_success_fraction=0.0):
    """
    Compute function scores from HDF5 results file.

    Score = mean of (loss / npart) over successful fits only.

    Returns
    -------
    scores : list of tuples
        (func_idx, avg_score, n_halos) sorted by score ascending.
    asymp_pass_dict : dict
        func_idx -> (n_pass_inf, n_pass_zero, n_total).
    n_halos_total : int
    n_filtered : int
        Functions filtered due to insufficient halo coverage.
    """
    output_path = Path(output_path)
    asymp_pass_dict = {}

    with h5py.File(output_path, 'r') as f:
        func_idx = f['func_idx'][:]
        halo_idx = f['halo_idx'][:]
        loss = f['loss'][:]

        if 'asymp_pass_func_idx' in f:
            apf_idx = f['asymp_pass_func_idx'][:]
            apf_inf = f['asymp_pass_inf'][:]
            apf_zero = f['asymp_pass_zero'][:]
            apf_total = f['asymp_pass_total'][:]
            for i, fidx in enumerate(apf_idx):
                asymp_pass_dict[int(fidx)] = (
                    int(apf_inf[i]), int(apf_zero[i]), int(apf_total[i]))

    npart_per_halo = np.asarray(npart_per_halo)
    n_halos_total = len(npart_per_halo)
    normalized_loss = loss / npart_per_halo[halo_idx]

    unique_funcs, inverse_idx, counts = np.unique(
        func_idx, return_inverse=True, return_counts=True)
    sum_per_func = np.bincount(inverse_idx, weights=normalized_loss)
    avg_scores = sum_per_func / counts

    min_halos = int(min_success_fraction * n_halos_total)
    mask = counts >= min_halos
    n_filtered = np.sum(~mask)

    scores = [(f, s, c) for f, s, c, m
              in zip(unique_funcs, avg_scores, counts, mask) if m]
    scores.sort(key=lambda x: x[1])

    return scores, asymp_pass_dict, n_halos_total, n_filtered


def print_best_results(output_path, equations, npart_per_halo,
                       asymp_postfit_funcs=None, nfw_score=None, n_top=100,
                       min_success_fraction=0.0):
    """Print a table of the best functions ranked by normalized loss."""
    output_path = Path(output_path)
    if not output_path.exists():
        print("No results file found for ranking.")
        return

    if asymp_postfit_funcs is None:
        asymp_postfit_funcs = set()

    result = compute_function_scores(
        output_path, npart_per_halo, min_success_fraction)
    scores, asymp_pass_dict, n_halos_total, n_filtered = result

    if nfw_score is not None:
        print("\n" + "-" * 79)
        print(f"NFW REFERENCE: AvgScore = {nfw_score:.4f}  "
              f"(rho = 1 / (x * (1 + x)^2))")
        print("-" * 79)

    print("\n" + "=" * 79)
    print("TOP FUNCTIONS (ranked by avg loss/npart per halo, lower is better)")
    print("=" * 79)
    header = f"{'Rank':<6} {'Func#':<8} {'AvgScore':<12} {'#Halos':<8} "
    header += f"{'Asymp':<8} Equation"
    print(header)
    print("-" * 79)

    for rank, (fidx, score, n_halos) in enumerate(scores[:n_top], 1):
        eq = equations[fidx]
        if len(eq) > 35:
            eq = eq[:32] + "..."
        if fidx in asymp_pass_dict:
            n_inf, n_zero, n_total = asymp_pass_dict[fidx]
            if n_total > 0:
                frac_inf = n_inf / n_total
                frac_zero = n_zero / n_total
                asymp_status = f"{100*frac_inf:.0f}/{100*frac_zero:.0f}"
            else:
                asymp_status = "-"
        elif fidx in asymp_postfit_funcs:
            asymp_status = "?"
        else:
            asymp_status = "-"
        row = f"{rank:<6} {fidx:<8} {score:<12.4f} {n_halos:<8} "
        row += f"{asymp_status:<8} {eq}"
        print(row)

    print("=" * 79)
    print(f"Showing top {min(n_top, len(scores))} of {len(scores)} functions")
    if n_filtered > 0:
        min_halos = int(min_success_fraction * n_halos_total)
        print(f"(Filtered out {n_filtered} functions with < {min_halos} "
              f"successful fits)")
    print("(AvgScore = mean of loss/npart over successful fits only)")
    print("(Asymp: inf%/zero% pass fractions for x->inf and x->0+ checks)")
    print("")
