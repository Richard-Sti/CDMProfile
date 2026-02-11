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
                            min_success_fraction=0.0,
                            failure_loss_percentile=0):
    """
    Compute function scores from HDF5 results file.

    Score = mean of (loss / npart) over all halos. For halos where a function
    failed, the loss is imputed using the specified percentile of successful
    fits for that function.

    Also computes BIC = nparams * ln(npart) + 2 * loss for model comparison.

    Parameters
    ----------
    output_path : Path
        Path to HDF5 results file.
    npart_per_halo : array-like
        Number of particles per halo.
    min_success_fraction : float
        Minimum fraction of halos that must be successfully fit.
    failure_loss_percentile : float
        Percentile (0-100) of successful fits to use for imputing failed halos.
        If 0, no imputation is done and score is averaged over successful only.

    Returns
    -------
    scores : list of tuples
        (func_idx, avg_score, avg_bic, nparams, n_halos) sorted by score.
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

        # Read nparams directly if available, otherwise compute from params
        if 'nparams' in f:
            nparams_per_result = f['nparams'][:]
        else:
            params = f['params'][:]
            nparams_per_result = np.sum(~np.isnan(params), axis=1)

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

    # Compute BIC per result: BIC = k * ln(n) + 2 * loss
    npart_per_result = npart_per_halo[halo_idx]
    bic_per_result = (nparams_per_result * np.log(npart_per_result)
                      + 2 * loss)

    unique_funcs, first_idx, inverse_idx, counts = np.unique(
        func_idx, return_index=True, return_inverse=True,
        return_counts=True)
    sum_per_func = np.bincount(inverse_idx, weights=normalized_loss)
    sum_bic_per_func = np.bincount(inverse_idx, weights=bic_per_result)

    # Get nparams per function (same for all halos, take first occurrence)
    nparams_per_func = nparams_per_result[first_idx]

    # Compute scores with optional imputation for failed halos
    if failure_loss_percentile > 0:
        # Sort results by function group for efficient per-function access
        sort_idx = np.argsort(inverse_idx, kind='mergesort')
        sorted_losses = normalized_loss[sort_idx]
        sorted_bics = bic_per_result[sort_idx]
        split_points = np.cumsum(counts[:-1])
        loss_groups = np.split(sorted_losses, split_points)
        bic_groups = np.split(sorted_bics, split_points)

        avg_scores = np.zeros(len(unique_funcs))
        avg_bic = np.zeros(len(unique_funcs))
        for i, n_success in enumerate(counts):
            n_failed = n_halos_total - n_success
            if n_failed > 0 and n_success > 0:
                # Impute using percentile of successful fits
                imputed_loss = np.percentile(
                    loss_groups[i], failure_loss_percentile)
                imputed_bic = np.percentile(
                    bic_groups[i], failure_loss_percentile)
                total_loss = sum_per_func[i] + n_failed * imputed_loss
                total_bic = sum_bic_per_func[i] + n_failed * imputed_bic
                avg_scores[i] = total_loss / n_halos_total
                avg_bic[i] = total_bic / n_halos_total
            else:
                # No failed halos or no successful fits
                avg_scores[i] = (
                    sum_per_func[i] / n_success if n_success > 0 else np.inf)
                avg_bic[i] = (
                    sum_bic_per_func[i] / n_success
                    if n_success > 0 else np.inf)
    else:
        # Original behavior: average over successful fits only
        avg_scores = sum_per_func / counts
        avg_bic = sum_bic_per_func / counts

    min_halos = int(min_success_fraction * n_halos_total)
    mask = counts >= min_halos
    n_filtered = np.sum(~mask)

    scores = [(f, s, b, k, c) for f, s, b, k, c, m
              in zip(unique_funcs, avg_scores, avg_bic, nparams_per_func,
                     counts, mask) if m]
    scores.sort(key=lambda x: x[1])

    return scores, asymp_pass_dict, n_halos_total, n_filtered


def print_best_results(output_path, equations, npart_per_halo,
                       asymp_postfit_funcs=None, nfw_score=None, nfw_bic=None,
                       n_top=100, min_success_fraction=0.0,
                       failure_loss_percentile=0):
    """Print a table of the best functions ranked by normalized loss."""
    output_path = Path(output_path)
    if not output_path.exists():
        print("No results file found for ranking.")
        return

    # Check if there are any results
    with h5py.File(output_path, 'r') as f:
        if 'func_idx' not in f:
            print("\nNo successful fits to rank.")
            return

    if asymp_postfit_funcs is None:
        asymp_postfit_funcs = set()

    result = compute_function_scores(
        output_path, npart_per_halo, min_success_fraction,
        failure_loss_percentile)
    scores, asymp_pass_dict, n_halos_total, n_filtered = result

    # Estimate σ_CE (cross-entropy uncertainty) for the best function
    if scores:
        best_fidx = scores[0][0]
        npart_arr = np.asarray(npart_per_halo)
        n_total_particles = int(np.sum(npart_arr))

        with h5py.File(output_path, 'r') as f:
            all_func_idx = f['func_idx'][:]
            all_halo_idx = f['halo_idx'][:]
            all_loss = f['loss'][:]

        # Per-halo CE for the best function
        best_mask = all_func_idx == best_fidx
        best_halo_idx = all_halo_idx[best_mask]
        best_ce = all_loss[best_mask] / npart_arr[best_halo_idx]
        n_halos_best = len(best_ce)

        if n_halos_best > 1:
            sigma_ce_halos = np.std(best_ce, ddof=1)
        else:
            sigma_ce_halos = 0.0
        sigma_ce = (sigma_ce_halos / np.sqrt(n_halos_best)
                    if n_halos_best > 0 else 0.0)

        print(f"\nCROSS-ENTROPY UNCERTAINTY (from best function, "
              f"func #{best_fidx})")
        print("-" * 70)
        print(f"  N_halos          = {n_halos_best}")
        print(f"  N_particles      = {n_total_particles:,}")
        print(f"  Best CE (mean)   = {scores[0][1]:.6f}")
        print(f"  sigma_CE (halo)  = {sigma_ce_halos:.6f}  "
              f"(std of per-halo CE)")
        print(f"  sigma_CE (mean)  = {sigma_ce:.6f}  "
              f"(std error of mean CE)")
        print("\n  Interpretation: when is Delta_CE meaningful?")
        print(f"  {'Delta_CE':<15} {'Significance':<20} {'Meaning'}")
        print(f"  {'-' * 60}")

        thresholds = [
            (0.5 * sigma_ce, "< 0.5 sigma_CE", "Indistinguishable"),
            (1.0 * sigma_ce, "~ 1 sigma_CE", "Marginal"),
            (2.0 * sigma_ce, "~ 2 sigma_CE", "Likely meaningful"),
            (3.0 * sigma_ce, "~ 3 sigma_CE", "Significant"),
        ]
        for thresh, label, meaning in thresholds:
            print(f"  {thresh:<15.6f} {label:<20} {meaning}")

        print(f"\n  Delta_CE < {sigma_ce:.6f} is within noise "
              f"for this dataset.")
        print("")

    # Find best BIC for dBIC computation
    if scores:
        best_bic = min(s[2] for s in scores)
    else:
        best_bic = 0

    # Best CE for Delta_CE computation
    best_ce_val = scores[0][1] if scores else 0.0

    if nfw_score is not None:
        print("\n" + "-" * 109)
        nfw_dce = nfw_score - best_ce_val
        nfw_str = f"NFW REFERENCE: CE = {nfw_score:.6f}"
        nfw_str += f", Delta_CE = {nfw_dce:.6f}"
        if nfw_bic is not None:
            nfw_dbic = nfw_bic - best_bic
            nfw_str += f", dBIC = {nfw_dbic:.1f}"
        nfw_str += "  (rho = 1 / (x * (1 + x)^2))"
        print(nfw_str)
        print("-" * 109)

    print("\n" + "=" * 109)
    print("TOP FUNCTIONS (ranked by CE = avg loss/npart, "
          "lower is better)")
    print("=" * 109)
    header = (f"{'Rank':<6} {'Func#':<7} {'CE':<12} "
              f"{'Delta_CE':<12} {'dBIC':<10} "
              f"{'k':<3} {'#Halo':<6} {'Asymp':<7} Equation")
    print(header)
    print("-" * 109)

    for rank, (fidx, score, bic, nparams, n_halos) in enumerate(
            scores[:n_top], 1):
        eq = equations[fidx]
        if len(eq) > 30:
            eq = eq[:27] + "..."
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
        dce = score - best_ce_val
        dbic = bic - best_bic
        row = (f"{rank:<6} {fidx:<7} {score:<12.6f} "
               f"{dce:<12.6f} {dbic:<10.1f} "
               f"{nparams:<3} {n_halos:<6} {asymp_status:<7} {eq}")
        print(row)

    print("=" * 109)
    print(f"Showing top {min(n_top, len(scores))} of "
          f"{len(scores)} functions")
    if n_filtered > 0:
        min_halos = int(min_success_fraction * n_halos_total)
        print(f"(Filtered out {n_filtered} functions with "
              f"< {min_halos} successful fits)")
    if failure_loss_percentile > 0:
        print(f"(CE = mean of loss/npart; failed halos imputed "
              f"with p{failure_loss_percentile})")
    else:
        print("(CE = mean of loss/npart over successful "
              "fits only)")
    print("(Delta_CE = CE - CE_best)")
    print("(dBIC = BIC - BIC_best; BIC = k*ln(n) + 2*loss)")
    print("(Asymp: inf%/zero% pass fractions for "
          "x->inf and x->0+ checks)")
    print("")
