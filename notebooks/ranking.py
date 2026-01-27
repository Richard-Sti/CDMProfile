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
"""Ranking and Pareto front utilities for density profile model selection."""
from pathlib import Path

import h5py
import numpy as np


###############################################################################
#                             Data loading                                    #
###############################################################################


def load_all(particle_file, results_dir, result_pattern, complexities):
    """
    Load particle metadata and all result files into a single structure.

    Parameters
    ----------
    particle_file : str or Path
        HDF5 file with M200c, R200c, halo_id, offsets.
    results_dir : str or Path
        Directory containing result HDF5 files.
    result_pattern : str
        Filename pattern with ``{comp}`` placeholder.
    complexities : list of int
        Complexity levels to load.

    Returns
    -------
    dict
        Consolidated data with precomputed per-result CE and BIC.
    """
    particle_file = Path(particle_file)
    results_dir = Path(results_dir)

    # --- Particle metadata ---
    with h5py.File(particle_file, "r") as f:
        M200c = f["M200c"][:]
        R200c = f["R200c"][:]
        halo_ids = f["halo_id"][:]
        offsets = f["offsets"][:]
        units_M200c = str(f.attrs["units_M200c"])

    npart_per_halo = np.diff(offsets).astype(np.int64)
    n_halos = len(M200c)

    print(f"Loaded {n_halos} halos from {particle_file.name}")
    print(f"  M200c range: {M200c.min():.2e} - {M200c.max():.2e} "
          f"{units_M200c}")

    # --- Result files ---
    # Per-result arrays (concatenated across complexities).
    r_comp = []       # complexity level
    r_func_idx = []   # func index within its complexity
    r_halo_idx = []   # halo index into particle arrays
    r_loss = []
    r_nparams = []

    # Per-complexity metadata.
    equations = {}     # comp -> list of str
    nfw_scores = {}    # comp -> float or None

    for comp in complexities:
        fname = result_pattern.format(comp=comp)
        fpath = results_dir / fname
        if not fpath.exists():
            print(f"  [SKIP] {fname} not found")
            continue

        with h5py.File(fpath, "r") as f:
            eqs = f["equations"][:]
            fi = f["func_idx"][:]
            hi = f["halo_idx"][:]
            lo = f["loss"][:]
            if "nparams" in f:
                np_ = f["nparams"][:]
            else:
                params = f["params"][:]
                np_ = np.sum(~np.isnan(params), axis=1)
            nfw = f.attrs.get("nfw_score", None)

        equations[comp] = [e.decode() if isinstance(e, bytes) else str(e)
                           for e in eqs]
        nfw_scores[comp] = float(nfw) if nfw is not None else None

        n_rows = len(fi)
        r_comp.append(np.full(n_rows, comp, dtype=np.int32))
        r_func_idx.append(fi.astype(np.int32))
        r_halo_idx.append(hi.astype(np.int32))
        r_loss.append(lo.astype(np.float64))
        r_nparams.append(np_.astype(np.int32))

        n_funcs = len(set(fi))
        print(f"  compl{comp}: {n_rows} results, {n_funcs} functions")

    if not r_comp:
        raise RuntimeError("No result files found.  Check results_dir, "
                           "result_pattern and complexities.")

    # Concatenate into flat arrays.
    comp_arr = np.concatenate(r_comp)
    func_idx = np.concatenate(r_func_idx)
    halo_idx = np.concatenate(r_halo_idx)
    loss = np.concatenate(r_loss)
    nparams = np.concatenate(r_nparams)

    # Precompute per-result CE and BIC.
    npart_result = npart_per_halo[halo_idx]
    ce = loss / npart_result
    bic = nparams * np.log(npart_result) + 2 * loss

    # Unique global function key: pack (comp, func_idx) into one int.
    # comp occupies high bits, func_idx low bits.
    max_fidx = int(func_idx.max()) + 1 if len(func_idx) > 0 else 1
    global_fid = comp_arr.astype(np.int64) * max_fidx + func_idx

    return {
        # Particle metadata.
        "M200c": M200c,
        "R200c": R200c,
        "halo_ids": halo_ids,
        "npart_per_halo": npart_per_halo,
        "n_halos": n_halos,
        "units_M200c": units_M200c,
        # Per-result arrays.
        "comp": comp_arr,
        "func_idx": func_idx,
        "halo_idx": halo_idx,
        "loss": loss,
        "nparams": nparams,
        "ce": ce,
        "bic": bic,
        "global_fid": global_fid,
        "_max_fidx": max_fidx,
        # Per-complexity metadata.
        "equations": equations,
        "nfw_scores": nfw_scores,
        "complexities": sorted(equations.keys()),
    }


###############################################################################
#                             Scoring                                         #
###############################################################################


def compute_scores(data, halo_mask=None, min_success_fraction=0.5,
                   failure_loss_percentile=95):
    """
    Compute per-function CE and BIC scores.

    Parameters
    ----------
    data : dict
        Output of :func:`load_all`.
    halo_mask : ndarray of bool or None
        Boolean mask of shape ``(n_halos,)`` selecting which halos to
        include. ``None`` means all halos.
    min_success_fraction : float
        Discard functions that fit fewer than this fraction of halos.
    failure_loss_percentile : float
        Percentile (0-100) used to impute missing halos.  0 disables
        imputation.

    Returns
    -------
    scores : dict
        ``global_fid -> (comp, func_idx, CE, BIC, nparams, n_halos)``.
    ranked : list
        Same tuples sorted by CE (ascending).
    """
    comp = data["comp"]
    func_idx = data["func_idx"]
    halo_idx = data["halo_idx"]
    ce = data["ce"]
    bic = data["bic"]
    nparams = data["nparams"]
    global_fid = data["global_fid"]

    # Filter results to selected halos (vectorised).
    if halo_mask is not None:
        keep = halo_mask[halo_idx]
        comp = comp[keep]
        func_idx = func_idx[keep]
        halo_idx = halo_idx[keep]
        ce = ce[keep]
        bic = bic[keep]
        nparams = nparams[keep]
        global_fid = global_fid[keep]
        n_halos_total = int(halo_mask.sum())
    else:
        n_halos_total = data["n_halos"]

    if len(global_fid) == 0:
        return {}, []

    unique_gfid, first_idx, inv, counts = np.unique(
        global_fid, return_index=True, return_inverse=True,
        return_counts=True)
    sum_ce = np.bincount(inv, weights=ce)
    sum_bic = np.bincount(inv, weights=bic)

    # Per-function metadata (first occurrence).
    k_arr = nparams[first_idx]
    comp_arr = comp[first_idx]
    fidx_arr = func_idx[first_idx]

    # Imputation: for functions that did not fit all selected halos,
    # impute missing halos with the given percentile of successful fits.
    if failure_loss_percentile > 0 and n_halos_total > 0:
        avg_ce = np.empty(len(unique_gfid))
        avg_bic = np.empty(len(unique_gfid))

        needs_impute = counts < n_halos_total
        if needs_impute.any():
            # Sort once by global_fid so per-function slicing is O(1).
            sort_idx = np.argsort(global_fid)
            sorted_ce = ce[sort_idx]
            sorted_bic = bic[sort_idx]
            # Offsets into sorted arrays for each unique function.
            offsets = np.zeros(len(unique_gfid) + 1, dtype=np.int64)
            np.cumsum(counts, out=offsets[1:])

        for i in range(len(unique_gfid)):
            n_ok = int(counts[i])
            n_fail = n_halos_total - n_ok
            if n_fail > 0 and n_ok > 0:
                s, e = int(offsets[i]), int(offsets[i + 1])
                imp_ce = np.percentile(
                    sorted_ce[s:e], failure_loss_percentile)
                imp_bic = np.percentile(
                    sorted_bic[s:e], failure_loss_percentile)
                avg_ce[i] = (sum_ce[i] + n_fail * imp_ce) / n_halos_total
                avg_bic[i] = (
                    sum_bic[i] + n_fail * imp_bic) / n_halos_total
            else:
                avg_ce[i] = sum_ce[i] / n_ok if n_ok > 0 else np.inf
                avg_bic[i] = sum_bic[i] / n_ok if n_ok > 0 else np.inf
    else:
        avg_ce = sum_ce / counts
        avg_bic = sum_bic / counts

    min_halos = int(min_success_fraction * n_halos_total)
    keep_func = counts >= min_halos

    scores = {}
    for i in range(len(unique_gfid)):
        if not keep_func[i]:
            continue
        gfid = int(unique_gfid[i])
        scores[gfid] = (int(comp_arr[i]), int(fidx_arr[i]),
                        float(avg_ce[i]), float(avg_bic[i]),
                        int(k_arr[i]), int(counts[i]))

    ranked = sorted(scores.values(), key=lambda x: x[2])
    return scores, ranked


###############################################################################
#                             Display                                         #
###############################################################################


def _eq_str(equations, comp, fidx, maxlen=40):
    eq = equations[comp][fidx]
    if len(eq) > maxlen:
        eq = eq[:maxlen - 3] + "..."
    return eq


def print_ranking_table(ranked, equations, title, n_top=30,
                        nfw_score=None):
    """Print a ranking table."""
    if not ranked:
        print(f"\n{title}: no results.")
        return

    best_ce = ranked[0][2]
    best_bic = min(r[3] for r in ranked)

    if nfw_score is not None:
        print(f"NFW REFERENCE: CE = {nfw_score:.6f}, "
              f"Delta_CE = {nfw_score - best_ce:.6f}")
        print("-" * 100)

    print(f"\n{'=' * 100}")
    print(title)
    print(f"{'=' * 100}")
    print(f"{'Rank':<6} {'Comp':<6} {'Func#':<7} {'CE':<12} "
          f"{'Delta_CE':<12} {'dBIC':<10} "
          f"{'k':<4} {'#Halo':<7} Equation")
    print("-" * 100)

    for rank, (comp, fidx, ce, bic_, k, n_halos) in enumerate(
            ranked[:n_top], 1):
        eq = _eq_str(equations, comp, fidx)
        print(f"{rank:<6} {comp:<6} {fidx:<7} {ce:<12.6f} "
              f"{ce - best_ce:<12.6f} {bic_ - best_bic:<10.1f} "
              f"{k:<4} {n_halos:<7} {eq}")

    print(f"{'=' * 100}")
    print(f"Showing top {min(n_top, len(ranked))} of "
          f"{len(ranked)} functions")


###############################################################################
#                             Pareto front                                    #
###############################################################################


def extract_ranked_arrays(ranked):
    """
    Extract per-function arrays from the ranked list.

    Parameters
    ----------
    ranked : list of tuples
        ``(comp, func_idx, CE, BIC, k, n_halos)`` sorted by CE.

    Returns
    -------
    dict with keys ``ks``, ``ces``, ``bics``, ``comps``.
    """
    return {
        "ks": np.array([r[4] for r in ranked]),
        "ces": np.array([r[2] for r in ranked]),
        "bics": np.array([r[3] for r in ranked]),
        "comps": np.array([r[0] for r in ranked]),
    }


def pareto_front(x, y):
    """
    Compute the Pareto front minimising both *x* and *y*.

    Parameters
    ----------
    x, y : ndarray
        Objective arrays of equal length.

    Returns
    -------
    pareto_mask : ndarray of bool
        True for Pareto-optimal points.
    """
    n = len(x)
    pareto_mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not pareto_mask[i]:
            continue
        for j in range(n):
            if i == j or not pareto_mask[j]:
                continue
            if (x[j] <= x[i] and y[j] <= y[i]
                    and (x[j] < x[i] or y[j] < y[i])):
                pareto_mask[i] = False
                break

    return pareto_mask
