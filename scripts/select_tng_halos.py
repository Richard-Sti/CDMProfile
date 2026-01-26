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
Script to select TNG halos for density profile fitting.

Selection criteria:
    1. Minimum mass (Group_M_Crit200 > threshold)
    2. Centrals only (use central subhalo particles)
    3. Center offset (|SubhaloCM - SubhaloPos| / R200c < threshold)
    4. Cosmological origin (SubhaloFlag == 1)
    5. Isolation (no massive neighbor within some radius)

After selection, extracts DM particles within R200c (positions and radii).
Supports MPI parallelization for particle extraction.
"""
import tomllib
from argparse import ArgumentParser
from pathlib import Path

import h5py
import illustris_python as il
import numpy as np
from mpi4py import MPI
from scipy.spatial import cKDTree


def load_local_config():
    """Load local_config.toml from the project root."""
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    config_path = project_dir / "local_config.toml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"local_config.toml not found at {config_path}. "
            "Please create it with [path] section containing 'data' key."
        )

    with open(config_path, "rb") as f:
        return tomllib.load(f)


###############################################################################
#                           Catalog loading                                   #
###############################################################################


def load_group_catalog(basepath, snap_num):
    """Load TNG group catalog using illustris_python."""
    group_fields = [
        'GroupPos',
        'Group_M_Crit200',
        'Group_R_Crit200',
        'GroupFirstSub',
        'GroupNsubs',
        'GroupMass',
    ]
    subhalo_fields = [
        'SubhaloPos',
        'SubhaloCM',
        'SubhaloMass',
        'SubhaloLenType',
    ]

    groups = il.groupcat.loadHalos(basepath, snap_num, fields=group_fields)
    subhalos = il.groupcat.loadSubhalos(basepath, snap_num,
                                        fields=subhalo_fields)

    # SubhaloFlag only exists in full-physics runs, try to load separately
    try:
        flag_data = il.groupcat.loadSubhalos(basepath, snap_num,
                                             fields=['SubhaloFlag'])
        subhalos['SubhaloFlag'] = flag_data['SubhaloFlag']
    except Exception:
        subhalos['SubhaloFlag'] = None

    return groups, subhalos


def load_snapshot_header(basepath, snap_num):
    """Load snapshot header to get box size and other metadata."""
    return il.groupcat.loadHeader(basepath, snap_num)


###############################################################################
#                           Selection criteria                                #
###############################################################################


def apply_mass_cut(groups, min_mass, pre_mask):
    """Select groups above minimum mass (min_mass in Msun/h)."""
    M200 = groups['Group_M_Crit200'] * 1e10  # Convert to Msun/h
    mask = pre_mask & (M200 > min_mass)
    print(f"  Mass cut (M200 > {min_mass:.2e} Msun/h): "
          f"{mask.sum()}/{pre_mask.sum()} pass")
    return mask


def apply_offset_cut(groups, subhalos, max_offset_frac, box_size, pre_mask):
    """Select groups with small center offset (< max_offset_frac * R200c).

    Offset is computed between SubhaloCM (center of mass) and SubhaloPos
    (most bound particle position) of the central subhalo.
    """
    mask = pre_mask.copy()

    subhalo_pos = subhalos['SubhaloPos']
    subhalo_cm = subhalos['SubhaloCM']
    R200c = groups['Group_R_Crit200']
    first_sub = groups['GroupFirstSub']

    candidates = np.where(pre_mask)[0]
    offset_fracs = []

    for i in candidates:
        if R200c[i] <= 0:
            mask[i] = False
            continue

        central_idx = first_sub[i]
        delta = subhalo_cm[central_idx] - subhalo_pos[central_idx]
        # Apply periodic boundary conditions
        delta = delta - box_size * np.round(delta / box_size)
        offset = np.linalg.norm(delta)
        offset_frac = offset / R200c[i]
        offset_fracs.append(offset_frac)

        if offset_frac >= max_offset_frac:
            mask[i] = False

    offset_fracs = np.array(offset_fracs)
    print(f"  Offset cut (< {max_offset_frac}): "
          f"{mask.sum()}/{pre_mask.sum()} pass")
    if len(offset_fracs) > 0:
        print(f"    Offset fraction: min={offset_fracs.min():.3f}, "
              f"median={np.median(offset_fracs):.3f}, "
              f"max={offset_fracs.max():.3f}")

    return mask


def apply_cosmological_origin_cut(groups, subhalos, pre_mask):
    """Select groups whose central subhalo has SubhaloFlag == 1."""
    # SubhaloFlag only exists in full-physics runs. In DM-only runs,
    # all subhalos are cosmological by definition.
    if 'SubhaloFlag' not in subhalos or subhalos['SubhaloFlag'] is None:
        print("  Cosmological origin cut: skipped (DM-only run)")
        return pre_mask.copy()

    mask = pre_mask.copy()
    subhalo_flag = subhalos['SubhaloFlag']
    first_sub = groups['GroupFirstSub']

    for i in np.where(pre_mask)[0]:
        central_idx = first_sub[i]
        if subhalo_flag[central_idx] != 1:
            mask[i] = False

    print(f"  Cosmological origin cut: {mask.sum()}/{pre_mask.sum()} pass")
    return mask


def apply_max_satellite_cut(groups, subhalos, max_satellite_ratio, pre_mask):
    """Reject groups with a satellite subhalo more massive than threshold."""
    mask = pre_mask.copy()

    first_sub = groups['GroupFirstSub']
    n_subs = groups['GroupNsubs']
    subhalo_mass = subhalos['SubhaloMass']

    for i in np.where(pre_mask)[0]:
        if n_subs[i] <= 1:
            continue

        central_idx = first_sub[i]
        central_mass = subhalo_mass[central_idx]

        if central_mass <= 0:
            mask[i] = False
            continue

        # Check all satellites
        for j in range(1, n_subs[i]):
            sat_mass = subhalo_mass[central_idx + j]
            if sat_mass > max_satellite_ratio * central_mass:
                mask[i] = False
                break

    print(f"  Max satellite cut (M_sat < {max_satellite_ratio} M_central): "
          f"{mask.sum()}/{pre_mask.sum()} pass")
    return mask


def apply_isolation_cut(groups, subhalos, isolation_distance,
                        isolation_mass_ratio, pre_mask):
    """
    Select isolated groups.

    Rejects halo i if there exists a neighbor (FoF group or Subfind subhalo)
    within (isolation_distance * R200c) where M > (isolation_mass_ratio * M_i).

    Uses scipy KD-tree for efficient spatial queries.
    """
    mask = pre_mask.copy()
    n_groups = len(groups['Group_M_Crit200'])
    candidates = np.where(pre_mask)[0]

    if len(candidates) == 0:
        print("  Isolation cut: 0/0 pass (no candidates)")
        return mask

    group_pos = groups['GroupPos']
    M200 = groups['Group_M_Crit200']
    R200c = groups['Group_R_Crit200']
    first_sub = groups['GroupFirstSub']
    n_subs = groups['GroupNsubs']

    subhalo_pos = subhalos['SubhaloPos']
    subhalo_mass = subhalos['SubhaloMass']

    # Build subhalo to group mapping
    n_subhalos = len(subhalo_mass)
    subhalo_group = np.zeros(n_subhalos, dtype=int)
    for i in range(n_groups):
        if n_subs[i] > 0:
            subhalo_group[first_sub[i]:first_sub[i] + n_subs[i]] = i

    # Find the minimum mass threshold (for pre-filtering)
    min_mass_threshold = isolation_mass_ratio * M200[candidates].min()

    # Pre-filter groups by mass (only keep potentially problematic neighbors)
    massive_groups_mask = M200 > min_mass_threshold
    massive_group_indices = np.where(massive_groups_mask)[0]
    print(
        f"    {len(massive_group_indices)} massive groups after pre-filtering")

    # Build KD-tree for massive groups
    if len(massive_group_indices) > 0:
        group_tree = cKDTree(group_pos[massive_group_indices])
    else:
        group_tree = None

    # Pre-filter subhalos by mass
    massive_subhalo_mask = subhalo_mass > min_mass_threshold
    massive_subhalo_indices = np.where(massive_subhalo_mask)[0]
    print(f"    {len(massive_subhalo_indices)} massive subhalos after "
          f"pre-filtering")

    # Build KD-tree for massive subhalos
    if len(massive_subhalo_indices) > 0:
        subhalo_tree = cKDTree(subhalo_pos[massive_subhalo_indices])
    else:
        subhalo_tree = None

    print(f"    Checking isolation for {len(candidates)} candidates...")
    for i in candidates:
        if R200c[i] <= 0 or M200[i] <= 0:
            mask[i] = False
            continue

        search_radius = isolation_distance * R200c[i]
        mass_threshold = isolation_mass_ratio * M200[i]
        center = group_pos[i]

        # Check massive FoF groups using KD-tree
        if group_tree is not None:
            nearby_idx = group_tree.query_ball_point(center, search_radius)
            for local_j in nearby_idx:
                j = massive_group_indices[local_j]
                if j == i:
                    continue
                if M200[j] > mass_threshold:
                    mask[i] = False
                    break

        if not mask[i]:
            continue

        # Check massive subhalos using KD-tree
        if subhalo_tree is not None:
            nearby_idx = subhalo_tree.query_ball_point(center, search_radius)
            for local_k in nearby_idx:
                k = massive_subhalo_indices[local_k]
                if subhalo_group[k] == i:
                    continue
                if subhalo_mass[k] > mass_threshold:
                    mask[i] = False
                    break

    print(f"  Isolation cut (d < {isolation_distance} R200c, "
          f"M_neighbor > {isolation_mass_ratio} M_self): "
          f"{mask.sum()}/{pre_mask.sum()} pass")
    return mask


def select_halos(groups, subhalos, box_size, min_mass, max_offset_frac,
                 isolation_distance, isolation_mass_ratio,
                 max_satellite_ratio):
    """Apply all selection criteria and return indices of selected groups."""
    print("\nApplying selection criteria:")

    n_groups = len(groups['Group_M_Crit200'])
    mask = groups['GroupNsubs'] > 0
    print(f"  Groups with subhalos: {mask.sum()}/{n_groups}")

    mask = apply_mass_cut(groups, min_mass, mask)
    mask = apply_cosmological_origin_cut(groups, subhalos, mask)
    mask = apply_offset_cut(groups, subhalos, max_offset_frac, box_size, mask)
    mask = apply_max_satellite_cut(groups, subhalos, max_satellite_ratio, mask)
    mask = apply_isolation_cut(
        groups, subhalos, isolation_distance, isolation_mass_ratio, mask)

    selected_indices = np.where(mask)[0]
    print(f"\nTotal selected: {len(selected_indices)} halos")

    return selected_indices


def print_selection_summary(groups, subhalos, selected_indices):
    """Print summary statistics for selected halos."""
    if len(selected_indices) == 0:
        print("\nNo halos selected.")
        return

    M200 = groups['Group_M_Crit200'][selected_indices] * 1e10  # Msun/h
    R200 = groups['Group_R_Crit200'][selected_indices]
    first_sub = groups['GroupFirstSub'][selected_indices]
    npart = subhalos['SubhaloLenType'][first_sub, 1]  # DM particles

    print("\n" + "=" * 70)
    print("SELECTED HALOS SUMMARY")
    print("=" * 70)
    print(f"  Count: {len(selected_indices)}")
    print(f"  M200c [Msun/h]: min={M200.min():.2e}, "
          f"median={np.median(M200):.2e}, max={M200.max():.2e}")
    print(f"  R200c [ckpc/h]: min={R200.min():.1f}, "
          f"median={np.median(R200):.1f}, max={R200.max():.1f}")
    print(f"  DM particles:   min={npart.min()}, "
          f"median={int(np.median(npart))}, max={npart.max()}")
    print(f"  Total DM particles: {npart.sum():,}")

    # Mass histogram in log bins
    log_M200 = np.log10(M200)
    bin_width = 0.2
    bin_edges = np.arange(
        np.floor(log_M200.min() / bin_width) * bin_width,
        np.ceil(log_M200.max() / bin_width) * bin_width + bin_width,
        bin_width
    )
    counts, _ = np.histogram(log_M200, bins=bin_edges)

    print("\n  Mass distribution (log10 M200c [Msun/h]):")
    for i, count in enumerate(counts):
        if count > 0:
            bin_lo = bin_edges[i]
            bin_hi = bin_edges[i + 1]
            print(f"    [{bin_lo:.1f}, {bin_hi:.1f}): {count:6d}")

    print("=" * 70)


###############################################################################
#                           Particle loading                                  #
###############################################################################


def load_dm_particles_for_halo(basepath, snap_num, halo_id, center, radius,
                               box_size):
    """
    Load DM particles within radius of center, return 3D positions.

    Uses FoF halo particles as the source, then filters to within radius.
    This is an approximation - FoF membership != spherical R200c cut.
    """
    # Load all DM particles (PartType1) belonging to this FoF halo
    coords = il.snapshot.loadHalo(basepath, snap_num, halo_id, 'dm',
                                  fields=['Coordinates'])

    if coords is None or len(coords) == 0:
        return np.array([]).reshape(0, 3)

    # Compute positions relative to center with periodic boundary conditions
    delta = coords - center
    delta = delta - box_size * np.round(delta / box_size)

    # Filter to particles within radius
    dist = np.linalg.norm(delta, axis=1)
    within = dist < radius

    return delta[within]


def get_n_snapshot_chunks(basepath, snap_num):
    """Return the number of chunk files for a snapshot."""
    snap_dir = Path(basepath) / f"snapdir_{snap_num:03d}"
    first_file = snap_dir / f"snap_{snap_num:03d}.0.hdf5"
    with h5py.File(first_file, 'r') as f:
        return f['Header'].attrs['NumFilesPerSnapshot']


def sphere_cut_process_chunks(basepath, snap_num, chunk_ids, centers,
                              radii, box_size, output_path, rank):
    """
    Process assigned snapshot chunks, finding DM particles within each
    halo's radius. Write results to a temporary HDF5 file.

    Each chunk is read once. For each chunk, all halos are checked.
    Results are written as concatenated positions ordered by halo index,
    with a counts array to reconstruct per-halo data.

    Parameters
    ----------
    basepath : str
        Path to TNG output directory.
    snap_num : int
        Snapshot number.
    chunk_ids : array-like of int
        Chunk file IDs this rank should process.
    centers : ndarray (n_halos, 3)
        Halo center positions (ALL halos).
    radii : ndarray (n_halos,)
        Radii for spherical cuts (e.g. R200c).
    box_size : float
        Periodic box size.
    output_path : str or Path
        Path to write temporary HDF5 file.
    rank : int
        MPI rank (for progress reporting).
    """
    snap_dir = Path(basepath) / f"snapdir_{snap_num:03d}"
    n_halos = len(centers)
    results = [[] for _ in range(n_halos)]
    radii_sq = radii ** 2
    n_chunks = len(chunk_ids)
    report_every = max(1, n_chunks // 10)
    particle_batch = 5_000_000

    for ci, chunk_id in enumerate(chunk_ids):
        chunk_path = snap_dir / f"snap_{snap_num:03d}.{chunk_id}.hdf5"
        with h5py.File(chunk_path, 'r') as f:
            if 'PartType1' not in f:
                continue
            dataset = f['PartType1']['Coordinates']
            n_particles = dataset.shape[0]

            for start in range(0, n_particles, particle_batch):
                end = min(start + particle_batch, n_particles)
                coords = dataset[start:end]

                for i in range(n_halos):
                    delta = coords - centers[i]
                    delta -= box_size * np.round(delta / box_size)
                    dist_sq = np.sum(delta**2, axis=1)
                    mask = dist_sq < radii_sq[i]

                    if mask.any():
                        results[i].append(delta[mask])

        if (ci + 1) % report_every == 0:
            print(f"    [Rank {rank}] Chunk {ci + 1}/{n_chunks}")

    # Write results to temporary HDF5
    counts = np.zeros(n_halos, dtype=np.int64)
    all_positions = []
    for i in range(n_halos):
        if results[i]:
            cat = np.concatenate(results[i])
            counts[i] = len(cat)
            all_positions.append(cat)

    with h5py.File(output_path, 'w') as f:
        f.create_dataset("counts", data=counts)
        if all_positions:
            f.create_dataset("positions",
                             data=np.concatenate(all_positions))
        else:
            f.create_dataset("positions",
                             data=np.array([]).reshape(0, 3))


def check_decreasing_outer_profile(pos, r200c, r_min_frac=0.6, r_max_frac=1.0,
                                   n_bins=10):
    """
    Check if particle density decreases on average in outer radial bins.

    Fits a line to log(density) vs log(r) and checks if slope is negative.
    This is robust to Poisson noise in individual bins.

    Parameters
    ----------
    pos : ndarray
        Particle positions relative to halo center (N, 3).
    r200c : float
        R200c of the halo.
    r_min_frac : float
        Inner edge of check region as fraction of R200c.
    r_max_frac : float
        Outer edge of check region as fraction of R200c.
    n_bins : int
        Number of bins for fitting.

    Returns
    -------
    bool
        True if density is decreasing on average, False otherwise.
    """
    radii = np.linalg.norm(pos, axis=1)
    r_norm = radii / r200c

    # Select particles in the outer region
    mask = (r_norm >= r_min_frac) & (r_norm <= r_max_frac)
    if mask.sum() < 2 * n_bins:
        return True  # Not enough particles to check

    r_outer = r_norm[mask]
    bin_edges = np.linspace(r_min_frac, r_max_frac, n_bins + 1)
    counts, _ = np.histogram(r_outer, bins=bin_edges)

    # Compute density (counts / shell volume, proportional to r^2 * dr)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    shell_volumes = bin_edges[1:]**3 - bin_edges[:-1]**3  # proportional to V
    density = counts / shell_volumes

    # Only fit bins with non-zero counts
    valid = density > 0
    if valid.sum() < 3:
        return True  # Not enough valid bins

    # Fit line to log(density) vs log(r)
    log_r = np.log(bin_centers[valid])
    log_rho = np.log(density[valid])

    # Simple linear regression: slope = cov(x,y) / var(x)
    slope = np.cov(log_r, log_rho)[0, 1] / np.var(log_r)

    # Density should decrease (negative slope)
    return slope < 0


def balance_halos_by_mass(selected_indices, masses, size):
    """Distribute halos among ranks so total mass per rank is balanced."""
    # Sort by mass descending (assign largest first)
    order = np.argsort(-masses)
    sorted_indices = selected_indices[order]
    sorted_masses = masses[order]

    # Greedy assignment: assign each halo to rank with smallest current load
    rank_loads = np.zeros(size)
    rank_assignments = [[] for _ in range(size)]

    for idx, mass in zip(sorted_indices, sorted_masses):
        target_rank = np.argmin(rank_loads)
        rank_assignments[target_rank].append(idx)
        rank_loads[target_rank] += mass

    return rank_assignments, rank_loads


def extract_halo_particles_mpi(basepath, snap_num, groups, subhalos,
                               selected_indices, output_dir, header,
                               args, comm):
    """Extract DM particle positions for selected halos using MPI."""
    box_size = header['BoxSize']
    rank = comm.Get_rank()
    size = comm.Get_size()

    output_dir = Path(output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    first_sub = groups['GroupFirstSub']
    R200c = groups['Group_R_Crit200']
    subhalo_pos = subhalos['SubhaloPos']
    M200 = groups['Group_M_Crit200']

    if args.sphere_cut:
        # ---- Sphere cut: partition snapshot chunks across MPI ranks ----
        # Each rank reads a unique subset of chunks and checks ALL halos.
        # Results are written to per-rank temp files, then collected.
        n_files = get_n_snapshot_chunks(basepath, snap_num)
        all_chunk_ids = np.arange(n_files)
        my_chunk_ids = np.array_split(all_chunk_ids, size)[rank]

        # All halo centers and radii (every rank needs all halos)
        all_centers = np.array(
            [subhalo_pos[first_sub[g]] for g in selected_indices])
        all_radii_arr = np.array([R200c[g] for g in selected_indices])

        if rank == 0:
            print(f"\nSphere cut: partitioning {n_files} chunks across "
                  f"{size} ranks ({len(selected_indices)} halos)")

        tmp_path = output_dir / f"_sphere_tmp_rank{rank}.hdf5"
        print(f"  [Rank {rank}] Processing {len(my_chunk_ids)} chunks...")
        sphere_cut_process_chunks(
            basepath, snap_num, my_chunk_ids, all_centers, all_radii_arr,
            box_size, tmp_path, rank)
        print(f"  [Rank {rank}] Wrote temp file")

        comm.Barrier()

        # Rank 0: stream from temp files directly to output HDF5
        if rank == 0:
            print("\n  Collecting results and writing output...")
            n_halos = len(selected_indices)

            # Open all temp files and read counts (small arrays)
            tmp_handles = []
            tmp_counts = []
            tmp_offsets = []
            for r in range(size):
                tmp_file = output_dir / f"_sphere_tmp_rank{r}.hdf5"
                fh = h5py.File(tmp_file, 'r')
                counts = fh['counts'][:]
                file_offsets = np.zeros(n_halos + 1, dtype=np.int64)
                file_offsets[1:] = np.cumsum(counts)
                tmp_handles.append(fh)
                tmp_counts.append(counts)
                tmp_offsets.append(file_offsets)

            # Create output file with resizable datasets
            output_file = output_dir / f"particles_{args.snap:03d}.hdf5"
            out_f = h5py.File(output_file, 'w')
            dset_pos = out_f.create_dataset(
                "positions", shape=(0, 3), maxshape=(None, 3),
                dtype='f8', chunks=True)
            dset_rad = out_f.create_dataset(
                "radii", shape=(0,), maxshape=(None,),
                dtype='f8', chunks=True)

            halo_ids = []
            npart_list = []
            rejected_monotonic = 0

            if args.subsample is not None:
                np.random.seed(args.seed)

            for i in range(n_halos):
                group_idx = selected_indices[i]

                # Read this halo's particles from each temp file
                parts = []
                for r in range(size):
                    c = int(tmp_counts[r][i])
                    if c > 0:
                        start = int(tmp_offsets[r][i])
                        parts.append(
                            tmp_handles[r]['positions'][start:start + c])

                if not parts:
                    continue

                pos = np.concatenate(parts)
                del parts

                if len(pos) == 0:
                    continue

                radius = all_radii_arr[i]

                if args.check_monotonic:
                    if not check_decreasing_outer_profile(pos, radius):
                        rejected_monotonic += 1
                        continue

                if (args.subsample is not None
                        and len(pos) > args.subsample):
                    idx = np.random.choice(
                        len(pos), args.subsample, replace=False)
                    pos = pos[idx]

                # Append to output HDF5 immediately
                pos_radii = np.linalg.norm(pos, axis=1)
                n_old = dset_pos.shape[0]
                n_new = len(pos)
                dset_pos.resize(n_old + n_new, axis=0)
                dset_pos[n_old:] = pos
                dset_rad.resize(n_old + n_new, axis=0)
                dset_rad[n_old:] = pos_radii
                del pos, pos_radii

                halo_ids.append(group_idx)
                npart_list.append(n_new)

            # Close and clean up temp files
            for r in range(size):
                tmp_handles[r].close()
                (output_dir / f"_sphere_tmp_rank{r}.hdf5").unlink()

            if args.check_monotonic and rejected_monotonic > 0:
                print(f"  Rejected {rejected_monotonic} halos with "
                      f"non-monotonic outer profile")

            # Write metadata datasets
            halo_ids = np.array(halo_ids)
            npart = np.array(npart_list)
            total_particles = int(dset_pos.shape[0])

            offsets_arr = np.zeros(len(halo_ids) + 1, dtype=np.int64)
            offsets_arr[1:] = np.cumsum(npart)

            out_f.create_dataset("halo_id", data=halo_ids)
            out_f.create_dataset("offsets", data=offsets_arr)
            out_f.create_dataset(
                "M200c", data=M200[halo_ids] * 1e10)  # Msun/h
            out_f.create_dataset(
                "R200c", data=R200c[halo_ids])  # ckpc/h

            out_f.attrs["snap_num"] = args.snap
            out_f.attrs["redshift"] = header['Redshift']
            out_f.attrs["box_size"] = header['BoxSize']
            out_f.attrs["min_mass"] = args.min_mass
            out_f.attrs["max_offset"] = args.max_offset
            out_f.attrs["max_satellite_ratio"] = args.max_satellite_ratio
            out_f.attrs["isolation_distance"] = args.isolation_distance
            out_f.attrs["isolation_mass_ratio"] = args.isolation_mass_ratio
            if args.subsample is not None:
                out_f.attrs["subsample"] = args.subsample
                out_f.attrs["seed"] = args.seed
            out_f.attrs["sphere_cut"] = True
            out_f.attrs["units_M200c"] = "Msun/h"
            out_f.attrs["units_R200c"] = "ckpc/h"
            out_f.attrs["units_box_size"] = "ckpc/h"
            out_f.attrs["units_positions"] = (
                "ckpc/h (relative to halo center)")
            out_f.attrs["units_radii"] = (
                "ckpc/h (distance from halo center)")
            out_f.close()

            # Compact: rewrite with contiguous storage (chunked
            # datasets from streaming are slower to read)
            print("  Compacting output file...")
            tmp_compact = output_file.with_suffix('.tmp.hdf5')
            compact_batch = 5_000_000
            with (h5py.File(output_file, 'r') as fin,
                  h5py.File(tmp_compact, 'w') as fout):
                n_total = fin['positions'].shape[0]
                dset_p = fout.create_dataset(
                    'positions', shape=(n_total, 3), dtype='f8')
                dset_r = fout.create_dataset(
                    'radii', shape=(n_total,), dtype='f8')
                for s in range(0, n_total, compact_batch):
                    e = min(s + compact_batch, n_total)
                    dset_p[s:e] = fin['positions'][s:e]
                    dset_r[s:e] = fin['radii'][s:e]

                for key in fin:
                    if key not in ('positions', 'radii'):
                        fin.copy(key, fout)
                for key, val in fin.attrs.items():
                    fout.attrs[key] = val

            tmp_compact.replace(output_file)

            print(f"  Processed {len(halo_ids)}/{n_halos} halos")
            print(f"  Total particles: {total_particles:,}")

            file_size = output_file.stat().st_size
            if file_size > 1e9:
                size_str = f"{file_size / 1e9:.2f} GB"
            else:
                size_str = f"{file_size / 1e6:.1f} MB"

            print("\n" + "=" * 70)
            print("OUTPUT FILE")
            print("=" * 70)
            print(f"  Path:       {output_file}")
            print(f"  Size:       {size_str}")
            print(f"  N halos:    {len(halo_ids)}")
            print(f"  N particles: {total_particles:,}")
            print("=" * 70)

        return  # All ranks done for sphere-cut path

    else:
        # ---- FoF-based: partition halos across MPI ranks ----
        masses = M200[selected_indices]
        rank_assignments, rank_loads = balance_halos_by_mass(
            selected_indices, masses, size)
        my_indices = np.array(rank_assignments[rank])

        if rank == 0:
            print(f"\nExtracting particles for "
                  f"{len(selected_indices)} halos "
                  f"using {size} MPI ranks...")
            print(f"  Load balance: min={rank_loads.min():.2e}, "
                  f"max={rank_loads.max():.2e} (10^10 Msun/h)")

        if args.subsample is not None:
            np.random.seed(args.seed + rank)

        my_halo_ids = []
        my_positions = []
        my_npart = []
        my_rejected_monotonic = 0
        n_my_halos = len(my_indices)
        report_every = max(1, n_my_halos // 10)

        my_centers = np.array(
            [subhalo_pos[first_sub[g]] for g in my_indices])
        my_radii = np.array([R200c[g] for g in my_indices])

        for i, group_idx in enumerate(my_indices):
            radius = my_radii[i]
            pos = load_dm_particles_for_halo(
                basepath, snap_num, group_idx,
                my_centers[i], radius, box_size)

            if len(pos) == 0:
                print(f"  [Rank {rank}] Warning: No particles for "
                      f"halo {group_idx}")
                continue

            if args.check_monotonic:
                if not check_decreasing_outer_profile(pos, radius):
                    my_rejected_monotonic += 1
                    continue

            if args.subsample is not None and len(pos) > args.subsample:
                idx = np.random.choice(
                    len(pos), args.subsample, replace=False)
                pos = pos[idx]

            my_halo_ids.append(group_idx)
            my_positions.append(pos)
            my_npart.append(len(pos))

            if (i + 1) % report_every == 0 or i == n_my_halos - 1:
                print(f"  [Rank {rank}] Processed "
                      f"{i + 1}/{n_my_halos} halos")

        # Gather all data to rank 0
        all_halo_ids = comm.gather(my_halo_ids, root=0)
        all_positions = comm.gather(my_positions, root=0)
        all_npart = comm.gather(my_npart, root=0)
        all_rejected_monotonic = comm.gather(my_rejected_monotonic, root=0)

        if rank == 0:
            total_rejected_monotonic = sum(all_rejected_monotonic)
            if args.check_monotonic and total_rejected_monotonic > 0:
                print(f"  Rejected {total_rejected_monotonic} halos "
                      f"with non-monotonic outer profile")

            halo_ids = []
            positions = []
            npart = []
            for r in range(size):
                halo_ids.extend(all_halo_ids[r])
                positions.extend(all_positions[r])
                npart.extend(all_npart[r])

    # ---- Common output writing (rank 0 only) ----
    if rank == 0:
        if len(halo_ids) == 0:
            print("  Warning: No halos successfully extracted!")
            return

        # Sort by halo ID for consistent ordering
        order = np.argsort(halo_ids)
        halo_ids = np.array(halo_ids)[order]
        positions = [positions[i] for i in order]
        npart = np.array(npart)[order]

        # Build offset array and concatenate positions
        offsets = np.zeros(len(halo_ids) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(npart)
        all_pos = np.concatenate(positions, axis=0)

        print(f"  Processed {len(halo_ids)}/{len(selected_indices)} halos")
        print(f"  Total particles: {len(all_pos):,}")

        # Compute radii from positions (positions are relative to halo center)
        all_radii = np.linalg.norm(all_pos, axis=1)

        # Save to HDF5
        output_file = output_dir / f"particles_{args.snap:03d}.hdf5"
        with h5py.File(output_file, 'w') as f:
            # Particle data
            f.create_dataset("halo_id", data=halo_ids)
            f.create_dataset("positions", data=all_pos)
            f.create_dataset("radii", data=all_radii)
            f.create_dataset("offsets", data=offsets)

            # Halo properties (for halos that were successfully extracted)
            f.create_dataset("M200c", data=M200[halo_ids] * 1e10)  # Msun/h
            f.create_dataset("R200c", data=R200c[halo_ids])        # ckpc/h

            # Selection parameters as attributes
            f.attrs["snap_num"] = args.snap
            f.attrs["redshift"] = header['Redshift']
            f.attrs["box_size"] = header['BoxSize']
            f.attrs["min_mass"] = args.min_mass
            f.attrs["max_offset"] = args.max_offset
            f.attrs["max_satellite_ratio"] = args.max_satellite_ratio
            f.attrs["isolation_distance"] = args.isolation_distance
            f.attrs["isolation_mass_ratio"] = args.isolation_mass_ratio
            if args.subsample is not None:
                f.attrs["subsample"] = args.subsample
                f.attrs["seed"] = args.seed
            f.attrs["units_M200c"] = "Msun/h"
            f.attrs["units_R200c"] = "ckpc/h"
            f.attrs["units_box_size"] = "ckpc/h"
            f.attrs["units_positions"] = "ckpc/h (relative to halo center)"
            f.attrs["units_radii"] = "ckpc/h (distance from halo center)"

        # Get file size
        file_size = output_file.stat().st_size
        if file_size > 1e9:
            size_str = f"{file_size / 1e9:.2f} GB"
        else:
            size_str = f"{file_size / 1e6:.1f} MB"

        print("\n" + "=" * 70)
        print("OUTPUT FILE")
        print("=" * 70)
        print(f"  Path:       {output_file}")
        print(f"  Size:       {size_str}")
        print(f"  N halos:    {len(halo_ids)}")
        print(f"  N particles: {len(all_pos):,}")
        print("=" * 70)


###############################################################################
#                                  Main                                       #
###############################################################################


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    parser = ArgumentParser(description="Select TNG halos for profile fitting")
    parser.add_argument("--basepath", type=str, required=True,
                        help="Path to TNG output directory")
    parser.add_argument("--snap", type=int, default=99,
                        help="Snapshot number (default: 99)")
    parser.add_argument("--min-mass", type=float, default=1e11,
                        help="Minimum M200c in Msun/h (default: 1e11)")
    parser.add_argument("--max-offset", type=float, default=0.07,
                        help="Maximum center offset fraction (default: 0.07)")
    parser.add_argument("--isolation-distance", type=float, default=5.0,
                        help="Isolation search radius in R200c (default: 5.0)")
    parser.add_argument("--isolation-mass-ratio", type=float, default=1.0,
                        help="Isolation mass ratio threshold (default: 1.0)")
    parser.add_argument("--max-satellite-ratio", type=float, default=0.1,
                        help="Max satellite/central mass ratio (default: 0.1)")
    parser.add_argument("--extract", action="store_true",
                        help="Extract particles for selected halos")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Subsample to fixed number of particles per halo")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for subsampling (default: 42)")
    parser.add_argument("--check-monotonic", action="store_true",
                        help="Reject halos with non-monotonic outer profile")
    parser.add_argument(
        "--sphere-cut", action="store_true",
        help="Load ALL particles within R200c (not just FoF members). "
        "Loops through snapshot chunks to include particles from "
        "all structures.")
    args = parser.parse_args()

    # Only rank 0 prints header and does selection
    if rank == 0:
        print("=" * 70)
        print("TNG Halo Selection for Profile Fitting")
        print("=" * 70)
        print(f"Basepath: {args.basepath}")
        print(f"Snapshot: {args.snap}")
        print(f"Min mass: {args.min_mass:.2e} Msun/h")
        print(f"Max offset: {args.max_offset}")
        print(f"Max satellite ratio: {args.max_satellite_ratio}")
        print(f"Isolation distance: {args.isolation_distance} x R200c")
        print(f"Isolation mass ratio: {args.isolation_mass_ratio}")

        print("\nLoading snapshot header...")
        header = load_snapshot_header(args.basepath, args.snap)
        box_size = header['BoxSize']
        print(f"  Box size: {box_size}")
        print(f"  Redshift: {header['Redshift']}")

        print("\nLoading group catalog...")
        groups, subhalos = load_group_catalog(args.basepath, args.snap)
        print(f"  Loaded {len(groups['Group_M_Crit200'])} groups")
        print(f"  Loaded {len(subhalos['SubhaloPos'])} subhalos")

        selected = select_halos(
            groups, subhalos,
            box_size=box_size,
            min_mass=args.min_mass,
            max_offset_frac=args.max_offset,
            isolation_distance=args.isolation_distance,
            isolation_mass_ratio=args.isolation_mass_ratio,
            max_satellite_ratio=args.max_satellite_ratio,
        )

        print_selection_summary(groups, subhalos, selected)
    else:
        groups = None
        subhalos = None
        selected = None
        header = None

    # Broadcast data to all ranks
    groups = comm.bcast(groups, root=0)
    subhalos = comm.bcast(subhalos, root=0)
    selected = comm.bcast(selected, root=0)
    header = comm.bcast(header, root=0)

    if not args.extract:
        return

    # Get output directory from local_config.toml
    if rank == 0:
        config = load_local_config()
        data_dir = Path(config["path"]["data"])
        output_dir = data_dir / "tng_particles"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nOutput directory: {output_dir}")
        if args.sphere_cut:
            print("  Mode: sphere cut (all particles within "
                  "R200c, not just FoF)")
        else:
            print("  Mode: FoF halo particles")
        if args.subsample is not None:
            print(f"  Subsampling to {args.subsample} particles per halo "
                  f"(seed={args.seed})")
    else:
        output_dir = None

    output_dir = comm.bcast(output_dir, root=0)

    extract_halo_particles_mpi(
        args.basepath, args.snap, groups, subhalos,
        selected, output_dir, header, args, comm
    )

    if rank == 0:
        print("\nDone!")


if __name__ == "__main__":
    main()
