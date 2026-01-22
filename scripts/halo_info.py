#!/usr/bin/env python
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
Print detailed information about a TNG halo.

Usage:
    python halo_info.py --halo-id 1234
    python halo_info.py --halo-id 1234 --snap 99
"""
from argparse import ArgumentParser

import illustris_python as il
import numpy as np


def load_catalogs(basepath, snap_num):
    """Load group and subhalo catalogs."""
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
    header = il.groupcat.loadHeader(basepath, snap_num)

    return groups, subhalos, header


def print_halo_info(halo_id, groups, subhalos, header):
    """Print detailed information about a halo."""
    box_size = header['BoxSize']

    # Basic properties
    M200c = groups['Group_M_Crit200'][halo_id] * 1e10  # Msun/h
    R200c = groups['Group_R_Crit200'][halo_id]  # ckpc/h
    group_pos = groups['GroupPos'][halo_id]
    n_subs = groups['GroupNsubs'][halo_id]
    first_sub = groups['GroupFirstSub'][halo_id]

    # Central subhalo properties
    central_pos = subhalos['SubhaloPos'][first_sub]
    central_cm = subhalos['SubhaloCM'][first_sub]
    central_mass = subhalos['SubhaloMass'][first_sub] * 1e10  # Msun/h
    central_npart_dm = subhalos['SubhaloLenType'][first_sub, 1]

    # Offset calculation with periodic boundaries
    delta = central_cm - central_pos
    delta = delta - box_size * np.round(delta / box_size)
    offset = np.linalg.norm(delta)
    offset_frac = offset / R200c

    print("=" * 60)
    print(f"HALO {halo_id}")
    print("=" * 60)

    print("\nBASIC PROPERTIES:")
    print(f"  M200c:           {M200c:.3e} Msun/h  (log M = {np.log10(M200c):.2f})")
    print(f"  R200c:           {R200c:.1f} ckpc/h")
    print(f"  GroupPos:        [{group_pos[0]:.1f}, {group_pos[1]:.1f}, {group_pos[2]:.1f}] ckpc/h")
    print(f"  N subhalos:      {n_subs}")

    print("\nCENTRAL SUBHALO (GroupFirstSub = {first_sub}):")
    print(f"  SubhaloPos:      [{central_pos[0]:.1f}, {central_pos[1]:.1f}, {central_pos[2]:.1f}] ckpc/h")
    print(f"  SubhaloCM:       [{central_cm[0]:.1f}, {central_cm[1]:.1f}, {central_cm[2]:.1f}] ckpc/h")
    print(f"  SubhaloMass:     {central_mass:.3e} Msun/h")
    print(f"  DM particles:    {central_npart_dm:,}")

    print("\nRELAXATION (offset = |SubhaloCM - SubhaloPos|):")
    print(f"  Offset:          {offset:.2f} ckpc/h")
    print(f"  Offset / R200c:  {offset_frac:.4f}  {'PASS' if offset_frac < 0.07 else 'FAIL'} (< 0.07)")

    # Satellites
    print("\nSATELLITES:")
    if n_subs <= 1:
        print("  No satellites")
    else:
        print(f"  {'Rank':<6} {'SubhaloID':<12} {'Mass [Msun/h]':<15} {'M/M_central':<12} {'DM particles'}")
        print("  " + "-" * 55)
        for i in range(1, min(n_subs, 11)):  # Show top 10 satellites
            sub_idx = first_sub + i
            sat_mass = subhalos['SubhaloMass'][sub_idx] * 1e10
            sat_ratio = sat_mass / central_mass
            sat_npart = subhalos['SubhaloLenType'][sub_idx, 1]
            print(f"  {i:<6} {sub_idx:<12} {sat_mass:<15.2e} {sat_ratio:<12.4f} {sat_npart:,}")
        if n_subs > 11:
            print(f"  ... and {n_subs - 11} more satellites")

        # Max satellite ratio
        max_sat_mass = 0
        for i in range(1, n_subs):
            sat_mass = subhalos['SubhaloMass'][first_sub + i] * 1e10
            max_sat_mass = max(max_sat_mass, sat_mass)
        max_sat_ratio = max_sat_mass / central_mass
        print(f"\n  Max satellite ratio: {max_sat_ratio:.4f}  {'PASS' if max_sat_ratio < 0.05 else 'FAIL'} (< 0.05)")

    # Neighbors (other FoF groups)
    print("\nNEARBY MASSIVE NEIGHBORS (within 5 R200c, M > 0.1 M_self):")
    search_radius = 5.0 * R200c
    mass_threshold = 0.1 * M200c
    n_groups = len(groups['Group_M_Crit200'])

    neighbors = []
    for j in range(n_groups):
        if j == halo_id:
            continue
        other_mass = groups['Group_M_Crit200'][j] * 1e10
        if other_mass <= mass_threshold:
            continue
        other_pos = groups['GroupPos'][j]
        delta = group_pos - other_pos
        delta = delta - box_size * np.round(delta / box_size)
        dist = np.linalg.norm(delta)
        if dist < search_radius:
            neighbors.append((j, other_mass, dist, dist / R200c))

    if not neighbors:
        print("  No massive neighbors within 5 R200c  PASS")
    else:
        neighbors.sort(key=lambda x: x[2])  # Sort by distance
        print(f"  {'HaloID':<10} {'Mass [Msun/h]':<15} {'Distance [ckpc/h]':<18} {'d / R200c'}")
        print("  " + "-" * 55)
        for hid, mass, dist, dist_frac in neighbors[:10]:
            print(f"  {hid:<10} {mass:<15.2e} {dist:<18.1f} {dist_frac:.2f}")
        if len(neighbors) > 10:
            print(f"  ... and {len(neighbors) - 10} more neighbors")
        print(f"\n  FAIL - {len(neighbors)} massive neighbor(s) found")

    print("=" * 60)


def main():
    parser = ArgumentParser(description="Print TNG halo information")
    parser.add_argument("--basepath", type=str,
                        default="/mnt/extraspace/rstiskalek/TNG300-1-Dark/output",
                        help="Path to TNG output directory")
    parser.add_argument("--snap", type=int, default=99,
                        help="Snapshot number (default: 99)")
    parser.add_argument("--halo-id", type=int, required=True,
                        help="Halo ID (FoF group index)")
    args = parser.parse_args()

    print(f"Loading catalogs from {args.basepath}, snap {args.snap}...")
    groups, subhalos, header = load_catalogs(args.basepath, args.snap)
    print(f"Loaded {len(groups['Group_M_Crit200'])} groups, "
          f"{len(subhalos['SubhaloMass'])} subhalos")
    print()

    if args.halo_id < 0 or args.halo_id >= len(groups['Group_M_Crit200']):
        print(f"Error: halo_id {args.halo_id} out of range")
        return

    print_halo_info(args.halo_id, groups, subhalos, header)


if __name__ == "__main__":
    main()
