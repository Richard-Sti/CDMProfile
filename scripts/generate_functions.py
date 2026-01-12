# Copyright (C) 2023 Richard Stiskalek, Deaglan Bartlett
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
Generate functions of a given complexity for a run.
"""
import os
from argparse import ArgumentParser
from os import listdir, makedirs
from os.path import abspath, dirname, exists, isfile, join
from shutil import move

import numpy as np
import esr.generation.duplicate_checker  # noqa
import esr.generation.generator as generator  # noqa
from mpi4py import MPI

from utils import read_config
from cdmprof.fitting import (is_bad_function, load_equations,
                             _has_normalization_only_param)


def generate_functions(runname, comp):
    """Generate functions of a given complexity for a run."""
    esr.generation.duplicate_checker.main(runname, comp)


def _has_x_dependence(expr_str):
    """Check if expression contains 'x' as a variable (not part of 'exp')."""
    import re
    # Remove 'exp' to avoid false positives, then check for 'x'
    cleaned = re.sub(r'exp', '', expr_str)
    return 'x' in cleaned


def _check_equation(idx, eq):
    """
    Check a single equation and return skip info if invalid.

    Returns
    -------
    tuple or None
        (idx, reason, eq) if should be skipped, None otherwise.
    """
    if not _has_x_dependence(eq):
        return (idx, "no_x_dependence", eq)
    elif _has_normalization_only_param(eq):
        return (idx, "normalization_only", eq)
    elif is_bad_function(eq):
        return (idx, "bad_function", eq)
    return None


def _compute_asymptote(idx, eq):
    """
    Compute limits as x -> 0+ and x -> infinity for an equation.

    Returns
    -------
    tuple
        (idx, eq, lim_zero, lim_inf) where limits are strings.
    """
    from sympy import symbols, limit, oo, sympify

    x = symbols('x', positive=True)
    # Don't assume sign for parameters - fitter allows [-500, 500]
    params = [symbols(f'a{i}', real=True) for i in range(4)]

    local_dict = {'x': x}
    local_dict.update({f'a{i}': params[i] for i in range(4)})

    try:
        expr = sympify(eq, locals=local_dict)
        lim_zero = limit(expr, x, 0, '+')
        lim_zero_str = str(lim_zero)
    except Exception:
        # If SymPy can't compute it, mark as unknown (will pass by default)
        lim_zero_str = "unknown"

    try:
        expr = sympify(eq, locals=local_dict)
        lim_inf = limit(expr, x, oo)
        lim_inf_str = str(lim_inf)
    except Exception:
        # If SymPy can't compute it, mark as unknown (will pass by default)
        lim_inf_str = "unknown"

    return (idx, eq, lim_zero_str, lim_inf_str)


def compute_asymptotes(targetdir, comp, comm, skip_idx=None):
    """
    Compute asymptotic behavior for all equations (MPI parallelized).

    Writes two files:
    - `asymptotes_zero_{comp}.txt`: idx | lim(x->0+) | equation
    - `asymptotes_inf_{comp}.txt`: idx | lim(x->inf) | equation

    Parameters
    ----------
    targetdir : str
        Directory containing unique_equations_{comp}.txt
    comp : int
        Complexity level.
    comm : MPI.Comm
        MPI communicator.
    skip_idx : set, optional
        Set of indices already rejected by validation (excluded from stats).

    Returns
    -------
    dict
        Dictionary with asymptote statistics (only for non-skipped functions).
    """
    import re
    param_pattern = re.compile(r'\ba[0-3]\b')

    if skip_idx is None:
        skip_idx = set()

    rank = comm.Get_rank()
    size = comm.Get_size()

    eq_path = join(targetdir, f"unique_equations_{comp}.txt")
    asymp_zero_path = join(targetdir, f"asymptotes_zero_{comp}.txt")
    asymp_inf_path = join(targetdir, f"asymptotes_inf_{comp}.txt")

    # Load equations on rank 0 and broadcast
    if rank == 0:
        if not exists(eq_path):
            print(f"No equations file found at {eq_path}")
            equations = None
        else:
            equations = load_equations(eq_path)
    else:
        equations = None

    equations = comm.bcast(equations, root=0)

    stats = {
        'zero_bad': 0,      # lim(x->0+) is 0 or negative
        'zero_param': 0,    # lim(x->0+) depends on parameters
        'zero_ok': 0,       # lim(x->0+) is positive (inf or const)
        'zero_unknown': 0,  # lim(x->0+) couldn't be computed
        'inf_bad': 0,       # lim(x->inf) is non-zero constant or inf
        'inf_param': 0,     # lim(x->inf) depends on parameters
        'inf_ok': 0,        # lim(x->inf) is 0
        'inf_unknown': 0,   # lim(x->inf) couldn't be computed
        'total_bad': 0,     # unique functions with ANY bad asymptote
        'total_param': 0,   # unique functions with param-dep (and no bad)
        'total_unknown': 0, # unique functions with unknown asymptotes (post-fit check)
        'total_ok': 0,      # unique functions with all OK asymptotes
    }

    if equations is None:
        return stats

    # Distribute work across ranks
    n_equations = len(equations)
    local_results = []

    for idx in range(rank, n_equations, size):
        result = _compute_asymptote(idx, equations[idx])
        local_results.append(result)

    # Gather results to rank 0
    all_results = comm.gather(local_results, root=0)

    # Rank 0 writes the asymptotes files
    if rank == 0:
        # Flatten and sort by index
        asymp_list = []
        for results in all_results:
            asymp_list.extend(results)
        asymp_list.sort(key=lambda x: x[0])

        # Write asymptotes_zero file (x -> 0+)
        with open(asymp_zero_path, 'w') as f:
            f.write("# idx | lim(x->0+) | equation\n")
            for idx, eq, lim_zero, lim_inf in asymp_list:
                f.write(f"{idx}\t{lim_zero}\t{eq}\n")

        # Write asymptotes_inf file (x -> inf)
        with open(asymp_inf_path, 'w') as f:
            f.write("# idx | lim(x->inf) | equation\n")
            for idx, eq, lim_zero, lim_inf in asymp_list:
                f.write(f"{idx}\t{lim_inf}\t{eq}\n")

        # Categorize asymptotes (only for non-skipped functions)
        for r in asymp_list:
            idx = r[0]
            if idx in skip_idx:
                continue  # Skip already-rejected functions

            lim_zero = r[2]
            lim_inf = r[3]

            zero_cat = None  # 'bad', 'param', or 'ok'
            inf_cat = None   # 'bad', 'param', or 'ok'

            # Categorize x -> 0+
            if lim_zero == "0":
                stats['zero_bad'] += 1
                zero_cat = 'bad'
            elif lim_zero in ("-oo", "-inf"):
                stats['zero_bad'] += 1
                zero_cat = 'bad'
            elif lim_zero == "unknown":
                # SymPy couldn't compute - needs post-fit check
                stats['zero_unknown'] += 1
                zero_cat = 'unknown'
            elif param_pattern.search(lim_zero):
                stats['zero_param'] += 1
                zero_cat = 'param'
            elif lim_zero in ("oo", "zoo", "inf"):
                stats['zero_ok'] += 1
                zero_cat = 'ok'
            else:
                # Try to parse as number (may be symbolic like "exp(-1)")
                try:
                    val = float(lim_zero)
                except ValueError:
                    # Try sympify + N for symbolic constants
                    try:
                        from sympy import sympify, N
                        val = float(N(sympify(lim_zero)))
                    except Exception:
                        val = None

                if val is None:
                    # Can't parse - needs post-fit check
                    stats['zero_unknown'] += 1
                    zero_cat = 'unknown'
                elif val <= 0:
                    stats['zero_bad'] += 1
                    zero_cat = 'bad'
                else:
                    stats['zero_ok'] += 1
                    zero_cat = 'ok'

            # Categorize x -> inf
            if lim_inf == "0":
                stats['inf_ok'] += 1
                inf_cat = 'ok'
            elif lim_inf in ("oo", "-oo", "zoo", "inf", "-inf"):
                stats['inf_bad'] += 1
                inf_cat = 'bad'
            elif lim_inf == "unknown":
                # SymPy couldn't compute - needs post-fit check
                stats['inf_unknown'] += 1
                inf_cat = 'unknown'
            elif param_pattern.search(lim_inf):
                stats['inf_param'] += 1
                inf_cat = 'param'
            else:
                # Try to parse as number - non-zero constant is bad
                try:
                    val = float(lim_inf)
                except ValueError:
                    # Try sympify + N for symbolic constants
                    try:
                        from sympy import sympify, N
                        val = float(N(sympify(lim_inf)))
                    except Exception:
                        val = None

                if val is None:
                    # Can't parse - needs post-fit check
                    stats['inf_unknown'] += 1
                    inf_cat = 'unknown'
                elif np.isclose(val, 0):
                    # Close enough to zero
                    stats['inf_ok'] += 1
                    inf_cat = 'ok'
                else:
                    stats['inf_bad'] += 1
                    inf_cat = 'bad'

            # Compute combined status for this function
            if zero_cat == 'bad' or inf_cat == 'bad':
                stats['total_bad'] += 1
            elif zero_cat == 'unknown' or inf_cat == 'unknown':
                stats['total_unknown'] += 1
            elif zero_cat == 'param' or inf_cat == 'param':
                stats['total_param'] += 1
            else:
                stats['total_ok'] += 1

    return stats


def validate_equations(targetdir, comp, comm):
    """
    Validate equations and write a skip file (MPI parallelized).

    Checks each equation for:
    - Missing x dependence (must depend on radius)
    - Normalization-only parameters (degenerate with mass normalization)
    - Bad functions (nan, inf, trig, etc.)

    Writes `skip_functions_{comp}.txt` with lines: `idx reason equation`

    Parameters
    ----------
    targetdir : str
        Directory containing unique_equations_{comp}.txt
    comp : int
        Complexity level.
    comm : MPI.Comm
        MPI communicator.

    Returns
    -------
    tuple
        (n_total, n_no_x, n_norm_only, n_bad, skip_idx) where skip_idx is a set.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    eq_path = join(targetdir, f"unique_equations_{comp}.txt")
    skip_path = join(targetdir, f"skip_functions_{comp}.txt")

    # Load equations on rank 0 and broadcast
    if rank == 0:
        if not exists(eq_path):
            print(f"No equations file found at {eq_path}")
            equations = None
        else:
            equations = load_equations(eq_path)
    else:
        equations = None

    equations = comm.bcast(equations, root=0)

    if equations is None:
        return (0, 0, 0, 0, set())

    # Distribute work across ranks
    n_equations = len(equations)
    local_results = []

    for idx in range(rank, n_equations, size):
        result = _check_equation(idx, equations[idx])
        if result is not None:
            local_results.append(result)

    # Gather results to rank 0
    all_results = comm.gather(local_results, root=0)

    # Rank 0 writes the skip file and returns counts
    n_no_x = 0
    n_norm_only = 0
    n_bad = 0
    skip_idx = set()

    if rank == 0:
        # Flatten and sort by index
        skip_list = []
        for results in all_results:
            skip_list.extend(results)
        skip_list.sort(key=lambda x: x[0])

        # Count by reason and collect skip indices
        for r in skip_list:
            skip_idx.add(r[0])
            if r[1] == "no_x_dependence":
                n_no_x += 1
            elif r[1] == "normalization_only":
                n_norm_only += 1
            elif r[1] == "bad_function":
                n_bad += 1

        # Write skip file
        with open(skip_path, 'w') as f:
            for idx, reason, eq in skip_list:
                f.write(f"{idx} {reason} {eq}\n")

    return (n_equations, n_no_x, n_norm_only, n_bad, skip_idx)


def print_summary(n_total, n_no_x, n_norm_only, n_bad, asymp_stats):
    """Print a progressive filtering summary."""
    print("")
    print("=" * 70)
    print("FUNCTION GENERATION SUMMARY")
    print("=" * 70)

    # Step 1: Unique equations
    print(f"\n1. Unique equations generated:           {n_total:>6}")

    # Step 2: No x dependence
    remaining = n_total - n_no_x
    pct = 100 * n_no_x / n_total if n_total > 0 else 0
    print(f"\n2. Remove no x-dependence:               -{n_no_x:>5} ({pct:>5.1f}%)")
    print(f"   Remaining:                            {remaining:>6}")

    # Step 3: Normalization-only
    remaining2 = remaining - n_norm_only
    pct = 100 * n_norm_only / n_total if n_total > 0 else 0
    print(f"\n3. Remove normalization-only params:     -{n_norm_only:>5} ({pct:>5.1f}%)")
    print(f"   Remaining:                            {remaining2:>6}")

    # Step 4: Bad functions (trig, nan, inf)
    remaining3 = remaining2 - n_bad
    pct = 100 * n_bad / n_total if n_total > 0 else 0
    print(f"\n4. Remove bad functions (trig/nan/inf):  -{n_bad:>5} ({pct:>5.1f}%)")
    print(f"   Remaining:                            {remaining3:>6}")

    # Step 5: Asymptotes
    print(f"\n5. Asymptotic behavior analysis:")

    # Show x->0+ breakdown
    print(f"   lim(x->0+):")
    print(f"     <= 0 (rejected):         {asymp_stats['zero_bad']:>5}")
    print(f"     > 0, definite (OK):      {asymp_stats['zero_ok']:>5}")
    print(f"     param-dependent:         {asymp_stats['zero_param']:>5}")
    print(f"     unknown (post-fit):      {asymp_stats['zero_unknown']:>5}")

    # Show x->inf breakdown
    print(f"   lim(x->inf):")
    print(f"     != 0 (rejected):         {asymp_stats['inf_bad']:>5}")
    print(f"     = 0, definite (OK):      {asymp_stats['inf_ok']:>5}")
    print(f"     param-dependent:         {asymp_stats['inf_param']:>5}")
    print(f"     unknown (post-fit):      {asymp_stats['inf_unknown']:>5}")

    # Combined rejections (use total_bad to avoid double-counting)
    n_asymp_bad = asymp_stats['total_bad']
    remaining4 = remaining3 - n_asymp_bad
    pct = 100 * n_asymp_bad / n_total if n_total > 0 else 0
    print(f"\n   Total rejected (bad asymptotes):      -{n_asymp_bad:>5} ({pct:>5.1f}%)")
    print(f"   Remaining:                            {remaining4:>6}")

    # Final summary (use total_param and total_unknown to avoid double-counting)
    n_param_dep = asymp_stats['total_param']
    n_unknown = asymp_stats['total_unknown']
    n_ready = asymp_stats['total_ok']
    print(f"\n" + "-" * 70)
    print(f"READY FOR FITTING:")
    print(f"  Definitely valid:                      {n_ready:>6}")
    print(f"  Parameter-dependent asymptotes:        {n_param_dep:>6}")
    print(f"  Unknown asymptotes (post-fit check):   {n_unknown:>6}")
    print(f"  Total to fit:                          {remaining4:>6}")

    total_eliminated = n_total - remaining4
    pct_eliminated = 100 * total_eliminated / n_total if n_total > 0 else 0
    pct_remaining = 100 * remaining4 / n_total if n_total > 0 else 0
    print(f"\n  Eliminated:                            {total_eliminated:>6} ({pct_eliminated:.1f}%)")
    print(f"  Kept:                                  {remaining4:>6} ({pct_remaining:.1f}%)")
    print("=" * 70)
    print("")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--runname", type=str,
                        help="ESR run name, defining the basis functions.")
    parser.add_argument("--comp", type=int, help="Function complexity.")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    if rank == 0:
        print(f"\nGenerating functions for complexity {args.comp}...")
        print(f"Run name: {args.runname}")
        print(f"MPI ranks: {size}")
        print("")

    # Run the generator
    generate_functions(args.runname, args.comp)

    # Move the generated functions to the CDMProfile directory
    targetdir = None
    if rank == 0:
        srcdir = abspath(
            join(dirname(generator.__file__), '..', 'function_library'))
        srcdir = join(srcdir, args.runname, f"compl_{args.comp}")

        config = read_config()
        targetdir = join(config["path"]["results"], args.runname,
                         f"compl_{args.comp}")
        if not exists(targetdir):
            makedirs(targetdir)

        print(f"Moving functions from `{srcdir}` to `{targetdir}`")
        for filename in listdir(srcdir):
            file_path = join(srcdir, filename)
            if isfile(file_path):
                move(file_path, os.path.join(targetdir, filename))

    # Broadcast targetdir to all ranks for validation
    targetdir = comm.bcast(targetdir, root=0)

    if rank == 0:
        print("Validating equations...")

    # Validate equations and write skip file (all ranks participate)
    n_total, n_no_x, n_norm_only, n_bad, skip_idx = validate_equations(
        targetdir, args.comp, comm)

    # Broadcast counts and skip_idx (they're only computed on rank 0)
    counts = comm.bcast((n_total, n_no_x, n_norm_only, n_bad), root=0)
    n_total, n_no_x, n_norm_only, n_bad = counts
    skip_idx = comm.bcast(skip_idx, root=0)

    if rank == 0:
        print("Computing asymptotes...")

    # Compute asymptotic behavior (all ranks participate)
    # Pass skip_idx so stats only count non-skipped functions
    asymp_stats = compute_asymptotes(targetdir, args.comp, comm, skip_idx)

    # Broadcast asymp_stats to rank 0
    asymp_stats = comm.bcast(asymp_stats, root=0)

    # Print summary on rank 0
    if rank == 0:
        print_summary(n_total, n_no_x, n_norm_only, n_bad, asymp_stats)

        # Print output file locations
        print(f"Output files:")
        print(f"  {join(targetdir, f'unique_equations_{args.comp}.txt')}")
        print(f"  {join(targetdir, f'skip_functions_{args.comp}.txt')}")
        print(f"  {join(targetdir, f'asymptotes_zero_{args.comp}.txt')}")
        print(f"  {join(targetdir, f'asymptotes_inf_{args.comp}.txt')}")
        print("")
