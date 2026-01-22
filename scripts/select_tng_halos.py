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

    # Balance halos by mass
    masses = M200[selected_indices]
    rank_assignments, rank_loads = balance_halos_by_mass(
        selected_indices, masses, size)
    my_indices = np.array(rank_assignments[rank])

    if rank == 0:
        print(f"\nExtracting particles for {len(selected_indices)} halos "
              f"using {size} MPI ranks...")
        print(f"  Load balance: min={rank_loads.min():.2e}, "
              f"max={rank_loads.max():.2e} (10^10 Msun/h)")

    # Set random seed for reproducible subsampling (different per rank)
    if args.subsample is not None:
        np.random.seed(args.seed + rank)

    # Each rank collects its halos' data
    my_halo_ids = []
    my_positions = []
    my_npart = []
    n_my_halos = len(my_indices)
    report_every = max(1, n_my_halos // 10)

    for i, group_idx in enumerate(my_indices):
        central_idx = first_sub[group_idx]
        center = subhalo_pos[central_idx]
        radius = R200c[group_idx]

        pos = load_dm_particles_for_halo(
            basepath, snap_num, group_idx, center, radius, box_size)

        if len(pos) == 0:
            print(f"  [Rank {rank}] Warning: No particles for halo "
                  f"{group_idx}")
            continue

        # Subsample if requested
        if args.subsample is not None and len(pos) > args.subsample:
            idx = np.random.choice(len(pos), args.subsample, replace=False)
            pos = pos[idx]

        my_halo_ids.append(group_idx)
        my_positions.append(pos)
        my_npart.append(len(pos))

        if (i + 1) % report_every == 0 or i == n_my_halos - 1:
            print(f"  [Rank {rank}] Processed {i + 1}/{n_my_halos} halos")

    # Gather all data to rank 0
    all_halo_ids = comm.gather(my_halo_ids, root=0)
    all_positions = comm.gather(my_positions, root=0)
    all_npart = comm.gather(my_npart, root=0)

    if rank == 0:
        # Flatten lists from all ranks
        halo_ids = []
        positions = []
        npart = []
        for r in range(size):
            halo_ids.extend(all_halo_ids[r])
            positions.extend(all_positions[r])
            npart.extend(all_npart[r])

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

        print(f"\nSaved particle data to {output_file}")


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
