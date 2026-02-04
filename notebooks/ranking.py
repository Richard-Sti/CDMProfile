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

    # Uniform halo weights (single snapshot).
    halo_weight = np.ones(n_halos, dtype=np.float64)

    return {
        # Particle metadata.
        "M200c": M200c,
        "R200c": R200c,
        "halo_ids": halo_ids,
        "npart_per_halo": npart_per_halo,
        "n_halos": n_halos,
        "units_M200c": units_M200c,
        "halo_weight": halo_weight,
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


def load_all_multi(particle_files, results_dir, result_patterns,
                    complexities, reweight_snapshots=False):
    """
    Load and concatenate halos from multiple particle/result file pairs.

    Parameters
    ----------
    particle_files : list of str or Path
        HDF5 files with M200c, R200c, halo_id, offsets.
    results_dir : str or Path
        Directory containing result HDF5 files.
    result_patterns : list of str
        Filename patterns with ``{comp}`` placeholder, parallel to
        `particle_files`.
    complexities : list of int
        Complexity levels to load.
    reweight_snapshots : bool, optional
        If True, assign per-halo weights so that each snapshot contributes
        equally to the average scores, regardless of how many halos it has.
        Default is False (uniform weights).

    Returns
    -------
    dict
        Same structure as :func:`load_all`, with data concatenated across
        all particle files and halo indices offset accordingly.
    """
    if len(particle_files) != len(result_patterns):
        raise ValueError(
            f"particle_files ({len(particle_files)}) and result_patterns "
            f"({len(result_patterns)}) must have the same length.")

    parts = [load_all(pf, results_dir, rp, complexities)
             for pf, rp in zip(particle_files, result_patterns)]

    # --- Verify equations are identical across all files ---
    ref_eqs = parts[0]["equations"]
    for i, p in enumerate(parts[1:], 1):
        if p["equations"] != ref_eqs:
            raise RuntimeError(
                f"Equations from file {i} differ from file 0. All files "
                "must use the same symbolic regression library.")

    # --- Concatenate halo metadata ---
    M200c = np.concatenate([p["M200c"] for p in parts])
    R200c = np.concatenate([p["R200c"] for p in parts])
    halo_ids = np.concatenate([p["halo_ids"] for p in parts])
    npart_per_halo = np.concatenate([p["npart_per_halo"] for p in parts])
    n_halos = sum(p["n_halos"] for p in parts)

    # --- Concatenate result arrays, offsetting halo_idx ---
    comp_arr = np.concatenate([p["comp"] for p in parts])
    func_idx = np.concatenate([p["func_idx"] for p in parts])
    loss = np.concatenate([p["loss"] for p in parts])
    nparams = np.concatenate([p["nparams"] for p in parts])

    halo_offset = 0
    halo_idx_parts = []
    for p in parts:
        halo_idx_parts.append(p["halo_idx"] + halo_offset)
        halo_offset += p["n_halos"]
    halo_idx = np.concatenate(halo_idx_parts)

    # --- Recompute derived arrays ---
    npart_result = npart_per_halo[halo_idx]
    ce = loss / npart_result
    bic = nparams * np.log(npart_result) + 2 * loss

    max_fidx = int(func_idx.max()) + 1 if len(func_idx) > 0 else 1
    global_fid = comp_arr.astype(np.int64) * max_fidx + func_idx

    # --- Merge nfw_scores (average across files) ---
    nfw_scores = {}
    for comp in ref_eqs:
        vals = [p["nfw_scores"][comp] for p in parts
                if p["nfw_scores"].get(comp) is not None]
        nfw_scores[comp] = float(np.mean(vals)) if vals else None

    # --- Per-halo weights ---
    if reweight_snapshots:
        # Each snapshot contributes equally: weight = 1 / n_halos_in_snapshot,
        # then normalised so weights sum to n_halos (like uniform weights).
        n_snaps = len(parts)
        halo_weight = np.empty(n_halos, dtype=np.float64)
        offset = 0
        for p in parts:
            n_h = p["n_halos"]
            # w_i = (n_halos / n_snaps) / n_h  so sum(w) = n_halos
            halo_weight[offset:offset + n_h] = n_halos / (n_snaps * n_h)
            offset += n_h
        print(f"  Snapshot reweighting enabled "
              f"({', '.join(str(p['n_halos']) for p in parts)} halos)")
    else:
        halo_weight = np.ones(n_halos, dtype=np.float64)

    # --- Per-halo snapshot index ---
    snap_idx = np.empty(n_halos, dtype=np.int32)
    offset = 0
    for si, p in enumerate(parts):
        n_h = p["n_halos"]
        snap_idx[offset:offset + n_h] = si
        offset += n_h

    print(f"\nCombined: {n_halos} halos from {len(parts)} files")

    return {
        "M200c": M200c,
        "R200c": R200c,
        "halo_ids": halo_ids,
        "npart_per_halo": npart_per_halo,
        "n_halos": n_halos,
        "units_M200c": parts[0]["units_M200c"],
        "halo_weight": halo_weight,
        "snap_idx": snap_idx,
        "n_snaps": len(parts),
        "comp": comp_arr,
        "func_idx": func_idx,
        "halo_idx": halo_idx,
        "loss": loss,
        "nparams": nparams,
        "ce": ce,
        "bic": bic,
        "global_fid": global_fid,
        "_max_fidx": max_fidx,
        "equations": ref_eqs,
        "nfw_scores": nfw_scores,
        "complexities": sorted(ref_eqs.keys()),
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
    halo_weight = data.get("halo_weight",
                           np.ones(data["n_halos"], dtype=np.float64))

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
        total_weight = float(halo_weight[halo_mask].sum())
    else:
        n_halos_total = data["n_halos"]
        total_weight = float(halo_weight.sum())

    # Per-result weights (looked up from halo weights).
    result_weight = halo_weight[halo_idx]

    if len(global_fid) == 0:
        return {}, []

    unique_gfid, first_idx, inv, counts = np.unique(
        global_fid, return_index=True, return_inverse=True,
        return_counts=True)
    wsum_ce = np.bincount(inv, weights=ce * result_weight)
    wsum_bic = np.bincount(inv, weights=bic * result_weight)
    wsum = np.bincount(inv, weights=result_weight)

    # Per-function metadata (first occurrence).
    k_arr = nparams[first_idx]
    comp_arr = comp[first_idx]
    fidx_arr = func_idx[first_idx]

    # Imputation: for functions that did not fit all selected halos,
    # impute missing halos with the given percentile of successful fits.
    if failure_loss_percentile > 0 and n_halos_total > 0:
        avg_ce = np.empty(len(unique_gfid))
        avg_bic = np.empty(len(unique_gfid))

        needs_impute = wsum < total_weight - 1e-10
        if needs_impute.any():
            # Sort once by global_fid so per-function slicing is O(1).
            sort_idx = np.argsort(global_fid)
            sorted_ce = ce[sort_idx]
            sorted_bic = bic[sort_idx]
            # Offsets into sorted arrays for each unique function.
            offsets = np.zeros(len(unique_gfid) + 1, dtype=np.int64)
            np.cumsum(counts, out=offsets[1:])

        for i in range(len(unique_gfid)):
            w_ok = wsum[i]
            w_fail = total_weight - w_ok
            if w_fail > 1e-10 and w_ok > 1e-10:
                s, e = int(offsets[i]), int(offsets[i + 1])
                imp_ce = np.percentile(
                    sorted_ce[s:e], failure_loss_percentile)
                imp_bic = np.percentile(
                    sorted_bic[s:e], failure_loss_percentile)
                avg_ce[i] = (wsum_ce[i] + w_fail * imp_ce) / total_weight
                avg_bic[i] = (wsum_bic[i] + w_fail * imp_bic) / total_weight
            else:
                avg_ce[i] = wsum_ce[i] / w_ok if w_ok > 1e-10 else np.inf
                avg_bic[i] = wsum_bic[i] / w_ok if w_ok > 1e-10 else np.inf
    else:
        avg_ce = wsum_ce / wsum
        avg_bic = wsum_bic / wsum

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


###############################################################################
#                        Per-subset scoring                                   #
###############################################################################


def _compute_subset_rankings(data, halo_mask, **kwargs):
    """Compute ranked list for a halo subset, returning full ranked list."""
    if halo_mask.sum() == 0:
        return []
    _, ranked = compute_scores(data, halo_mask=halo_mask, **kwargs)
    return ranked


def _ranked_to_lookup(ranked):
    """Convert ranked list to {(comp, fidx) -> (rank, CE, dCE)}."""
    if not ranked:
        return {}
    best_ce = ranked[0][2]
    return {(r[0], r[1]): (ri, r[2], r[2] - best_ce)
            for ri, r in enumerate(ranked, 1)}


def compute_subset_rankings(data, mass_edges, halo_mask=None, **kwargs):
    """
    Compute ranked lists for all subsets at once.

    Returns
    -------
    dict with keys:
        "per_snap" : {snap_idx -> lookup}
        "per_mass" : {bin_idx -> lookup}
        "per_snap_mass" : {(snap_idx, bin_idx) -> lookup}

    Each lookup maps ``(comp, fidx) -> (rank, CE, dCE)``.
    """
    snap_idx = data.get("snap_idx")
    n_snaps = data.get("n_snaps", 0)
    M200c = data["M200c"]
    n_bins = len(mass_edges) - 1

    base_mask = halo_mask if halo_mask is not None else np.ones(
        data["n_halos"], dtype=bool)

    per_snap = {}
    per_mass = {}
    per_snap_mass = {}

    # Per-snapshot
    if snap_idx is not None:
        for si in range(n_snaps):
            m = base_mask & (snap_idx == si)
            per_snap[si] = _ranked_to_lookup(
                _compute_subset_rankings(data, m, **kwargs))

    # Per-mass bin
    for bi in range(n_bins):
        lo, hi = mass_edges[bi], mass_edges[bi + 1]
        m = base_mask & (M200c >= lo) & (M200c < hi)
        per_mass[bi] = _ranked_to_lookup(
            _compute_subset_rankings(data, m, **kwargs))

    # Per-(snapshot, mass bin)
    if snap_idx is not None:
        for si in range(n_snaps):
            for bi in range(n_bins):
                lo, hi = mass_edges[bi], mass_edges[bi + 1]
                m = base_mask & (snap_idx == si) & (M200c >= lo) & (M200c < hi)
                per_snap_mass[(si, bi)] = _ranked_to_lookup(
                    _compute_subset_rankings(data, m, **kwargs))

    return {
        "per_snap": per_snap,
        "per_mass": per_mass,
        "per_snap_mass": per_snap_mass,
    }



def get_function_results(data, comp, func_idx):
    """
    Extract all per-halo results for a single function.

    Returns
    -------
    dict with keys: halo_idx, loss, ce, nparams, snap_idx (if available)
    """
    mask = (data["comp"] == comp) & (data["func_idx"] == func_idx)
    out = {
        "halo_idx": data["halo_idx"][mask],
        "loss": data["loss"][mask],
        "ce": data["ce"][mask],
        "nparams": data["nparams"][mask][0] if mask.any() else 0,
    }
    if "snap_idx" in data:
        out["snap_idx"] = data["snap_idx"][out["halo_idx"]]
    out["M200c"] = data["M200c"][out["halo_idx"]]
    return out


def load_function_params(results_dir, result_patterns, comp, func_idx):
    """
    Load fitted parameters for a single function from HDF5 result files.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing result HDF5 files.
    result_patterns : list of str
        Filename patterns with ``{comp}`` placeholder, one per snapshot.
    comp : int
        Complexity level.
    func_idx : int
        Function index within the complexity.

    Returns
    -------
    dict
        ``snap_index -> params array (n_halos_fitted, nparams)``
    """
    results_dir = Path(results_dir)
    out = {}
    for si, pattern in enumerate(result_patterns):
        fpath = results_dir / pattern.format(comp=comp)
        if not fpath.exists():
            continue
        with h5py.File(fpath, "r") as f:
            fi = f["func_idx"][:]
            mask = fi == func_idx
            if not mask.any():
                continue
            params = f["params"][mask]
            # Strip NaN padding columns
            valid_cols = ~np.all(np.isnan(params), axis=0)
            out[si] = params[:, valid_cols]
    return out


def _fmt_rank_dce(lookup, key, col_w=14):
    """Format rank+dCE for a subset lookup, or '---'."""
    info = lookup.get(key)
    if info is None:
        return f"{'---':<{col_w}}"
    rank, ce, dce = info
    return f"#{rank:<5} +{dce:<6.4f}" + " "


def print_cross_snapshot_table(ranked, data, subset, n_top=20,
                               snap_labels=None):
    """
    Print top functions with per-snapshot rank and dCE.

    Parameters
    ----------
    subset : dict
        Output of compute_subset_rankings.
    """
    per_snap = subset["per_snap"]
    n_snaps = len(per_snap)
    if snap_labels is None:
        snap_labels = [f"S{i}" for i in range(n_snaps)]
    eqs = data["equations"]

    col_w = 14
    snap_cols = "".join(f"{lbl:<{col_w}}" for lbl in snap_labels)
    w = 60 + col_w * n_snaps
    print(f"\n{'=' * w}")
    print("RANKING BY SNAPSHOT (rank within snapshot, dCE to best)")
    print(f"{'=' * w}")
    print(f"{'Rank':<6} {'CE':<12} {snap_cols}{'k':<4} Equation")
    print("-" * w)

    for rank, entry in enumerate(ranked[:n_top], 1):
        comp, fidx, ce, bic_, k, n_halos = entry
        key = (comp, fidx)
        cols = "".join(_fmt_rank_dce(per_snap[si], key, col_w)
                       for si in range(n_snaps))
        eq = _eq_str(eqs, comp, fidx, maxlen=40)
        print(f"{rank:<6} {ce:<12.6f} {cols}{k:<4} {eq}")

    print(f"{'=' * w}")


def print_cross_mass_table(ranked, data, subset, mass_edges, n_top=20):
    """
    Print top functions with per-mass-bin rank and dCE.
    """
    per_mass = subset["per_mass"]
    n_bins = len(mass_edges) - 1
    eqs = data["equations"]

    bin_labels = []
    for bi in range(n_bins):
        lo, hi = mass_edges[bi], mass_edges[bi + 1]
        hi_s = f"{hi:.1e}" if np.isfinite(hi) else "inf"
        bin_labels.append(f"[{lo:.1e},{hi_s})")

    col_w = max(len(bl) + 2 for bl in bin_labels)
    col_w = max(col_w, 14)
    mass_cols = "".join(f"{lbl:<{col_w}}" for lbl in bin_labels)
    w = 60 + col_w * n_bins
    print(f"\n{'=' * w}")
    print("RANKING BY MASS BIN (rank within bin, dCE to best)")
    print(f"{'=' * w}")
    print(f"{'Rank':<6} {'CE':<12} {mass_cols}{'k':<4} Equation")
    print("-" * w)

    for rank, entry in enumerate(ranked[:n_top], 1):
        comp, fidx, ce, bic_, k, n_halos = entry
        key = (comp, fidx)
        cols = "".join(_fmt_rank_dce(per_mass[bi], key, col_w)
                       for bi in range(n_bins))
        eq = _eq_str(eqs, comp, fidx, maxlen=40)
        print(f"{rank:<6} {ce:<12.6f} {cols}{k:<4} {eq}")

    print(f"{'=' * w}")


def print_cross_snap_mass_table(ranked, data, subset, mass_edges, n_top=20,
                                snap_labels=None):
    """
    Print top functions with rank and dCE in each (snapshot x mass bin) cell.
    """
    per_snap_mass = subset["per_snap_mass"]
    n_snaps = data.get("n_snaps", 0)
    n_bins = len(mass_edges) - 1
    eqs = data["equations"]
    if snap_labels is None:
        snap_labels = [f"S{i}" for i in range(n_snaps)]

    bin_labels = []
    for bi in range(n_bins):
        lo, hi = mass_edges[bi], mass_edges[bi + 1]
        hi_s = f"{hi:.1e}" if np.isfinite(hi) else "inf"
        bin_labels.append(f"[{lo:.1e},{hi_s})")

    col_w = max(max((len(bl) + 2 for bl in bin_labels), default=14), 14)

    print(f"\n{'=' * 80}")
    print("RANKING BY SNAPSHOT x MASS BIN (rank within cell, dCE to best)")
    print(f"{'=' * 80}")

    for rank, entry in enumerate(ranked[:n_top], 1):
        comp, fidx, ce, bic_, k, n_halos = entry
        key = (comp, fidx)
        eq = _eq_str(eqs, comp, fidx, maxlen=60)
        print(f"\n  #{rank} (CE={ce:.6f}, k={k}): {eq}")

        # Header row
        header = f"    {'':<14}" + "".join(f"{bl:<{col_w}}" for bl in bin_labels)
        print(header)

        for si in range(n_snaps):
            row = f"    {snap_labels[si]:<14}"
            for bi in range(n_bins):
                row += _fmt_rank_dce(per_snap_mass.get((si, bi), {}),
                                     key, col_w)
            print(row)

    print(f"\n{'=' * 80}")


def print_function_inspector(data, ranked, subset, mass_edges,
                             rank=None, expr=None, snap_labels=None):
    """
    Inspect a single function in detail.

    Parameters
    ----------
    subset : dict
        Output of compute_subset_rankings.
    rank : int or None
        1-based rank in the ranked list.
    expr : str or None
        Expression string to search for (partial match).
    """
    if rank is not None:
        found_rank = rank
        entry = ranked[rank - 1]
    elif expr is not None:
        eqs = data["equations"]
        found_rank = None
        for ri, entry in enumerate(ranked, 1):
            comp, fidx = entry[0], entry[1]
            if expr in eqs[comp][fidx]:
                found_rank = ri
                break
        else:
            print(f"Expression '{expr}' not found in ranked list.")
            return
    else:
        print("Specify rank= or expr=")
        return

    comp, fidx, ce, bic_, k, n_halos = entry
    eqs = data["equations"]
    eq = eqs[comp][fidx]
    key = (comp, fidx)

    per_snap = subset["per_snap"]
    per_mass = subset["per_mass"]
    per_snap_mass = subset["per_snap_mass"]
    n_snaps = len(per_snap)
    n_bins = len(mass_edges) - 1

    if snap_labels is None:
        snap_labels = [f"Snap {i}" for i in range(n_snaps)]

    bin_labels = []
    for bi in range(n_bins):
        lo, hi = mass_edges[bi], mass_edges[bi + 1]
        hi_s = f"{hi:.1e}" if np.isfinite(hi) else "inf"
        bin_labels.append(f"[{lo:.1e}, {hi_s})")

    print(f"{'=' * 80}")
    print(f"FUNCTION INSPECTOR — Rank #{found_rank}/{len(ranked)}")
    print(f"{'=' * 80}")
    print(f"Expression : {eq}")
    print(f"Complexity : {comp}")
    print(f"Parameters : k = {k}")
    print(f"CE (overall): {ce:.6f}")
    print(f"BIC         : {bic_:.2f}")
    print(f"Halos fitted: {n_halos}/{data['n_halos']}")

    # Per-snapshot rank
    print(f"\n--- Per-snapshot ---")
    for si in range(n_snaps):
        info = per_snap[si].get(key)
        if info is not None:
            r, sc, dce = info
            print(f"  {snap_labels[si]:<14}: rank #{r:<6} CE={sc:.6f}  "
                  f"dCE={dce:.6f}")
        else:
            print(f"  {snap_labels[si]:<14}: ---")

    # Per-mass rank
    print(f"\n--- Per-mass-bin ---")
    for bi in range(n_bins):
        info = per_mass[bi].get(key)
        if info is not None:
            r, sc, dce = info
            print(f"  {bin_labels[bi]:<28}: rank #{r:<6} CE={sc:.6f}  "
                  f"dCE={dce:.6f}")
        else:
            print(f"  {bin_labels[bi]:<28}: ---")

    # 2D table: snapshot x mass bin
    col_w = max(14, max((len(bl) + 2 for bl in bin_labels), default=14))
    print(f"\n--- Snapshot x mass bin ---")
    header = f"  {'':<14}" + "".join(f"{bl:<{col_w}}" for bl in bin_labels)
    print(header)
    for si in range(n_snaps):
        row = f"  {snap_labels[si]:<14}"
        for bi in range(n_bins):
            row += _fmt_rank_dce(per_snap_mass.get((si, bi), {}), key, col_w)
        print(row)

    # Per-halo results
    res = get_function_results(data, comp, fidx)
    print(f"\n--- Per-halo CE stats ---")
    print(f"  min    : {res['ce'].min():.6f}")
    print(f"  median : {np.median(res['ce']):.6f}")
    print(f"  max    : {res['ce'].max():.6f}")
    print(f"  std    : {res['ce'].std():.6f}")
    print(f"{'=' * 80}")


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
