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
import re
from argparse import ArgumentParser
from os import listdir, makedirs
from os.path import abspath, dirname, exists, isfile, join
from shutil import move

import cdmprof
import esr.generation.duplicate_checker  # noqa
import esr.generation.generator as generator  # noqa
import numpy as np

from utils import read_config


def generate_functions(runname, comp):
    """Generate functions of a given complexity for a run."""
    esr.generation.duplicate_checker.main(runname, comp)


def _has_x_dependence(expr_str):
    """Check if expression contains 'x' as a variable (not part of 'exp')."""
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
    elif cdmprof.fitting._has_normalization_only_param(eq):
        return (idx, "normalization_only", eq)
    elif cdmprof.fitting.is_bad_function(eq):
        return (idx, "bad_function", eq)
    return None


def _compute_single_limit(eq, limit_type):
    """
    Compute a single limit (helper for timeout wrapper).

    Parameters
    ----------
    eq : str
        The equation string.
    limit_type : str
        'zero' for lim(x->0+) or 'inf' for lim(x->inf).

    Returns
    -------
    str
        The limit result as a string.
    """
    from sympy import limit, oo, symbols, sympify

    x = symbols('x', positive=True)
    params = [symbols(f'a{i}', real=True) for i in range(4)]
    local_dict = {'x': x}
    local_dict.update({f'a{i}': params[i] for i in range(4)})

    expr = sympify(eq, locals=local_dict)
    if limit_type == 'zero':
        return str(limit(expr, x, 0, '+'))
    else:  # 'inf'
        return str(limit(expr, x, oo))


def _limit_worker(queue, eq, limit_type):
    """Worker function for multiprocessing limit computation."""
    try:
        result = _compute_single_limit(eq, limit_type)
        queue.put(result)
    except Exception:
        queue.put("unknown")


def _compute_asymptote(idx, eq, verbose=False, rank=0, timeout=5):
    """
    Compute limits as x -> 0+ and x -> infinity for an equation.

    Parameters
    ----------
    idx : int
        Index of the equation.
    eq : str
        The equation string.
    verbose : bool, optional
        If True, print progress for each limit computation.
    rank : int, optional
        MPI rank (for verbose output).
    timeout : int, optional
        Timeout in seconds for each limit computation. Default: 5.

    Returns
    -------
    tuple
        (idx, eq, lim_zero, lim_inf) where limits are strings.
    """
    import multiprocessing as mp

    # Use 'spawn' context to avoid issues with MPI (fork can cause problems)
    ctx = mp.get_context('spawn')
    Process = ctx.Process
    Queue = ctx.Queue

    # Compute lim(x->0+)
    if verbose:
        print(f"  [Rank {rank}] idx={idx} computing lim(x->0+): {eq}",
              flush=True)
    queue = Queue()
    proc = Process(target=_limit_worker, args=(queue, eq, 'zero'))
    proc.start()
    proc.join(timeout=timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        lim_zero_str = "timeout"
        if verbose:
            print(f"  [Rank {rank}] idx={idx} TIMEOUT on lim(x->0+): {eq}",
                  flush=True)
    else:
        try:
            lim_zero_str = queue.get_nowait()
        except Exception:
            lim_zero_str = "unknown"

    # Compute lim(x->inf)
    if verbose:
        print(f"  [Rank {rank}] idx={idx} computing lim(x->inf): {eq}",
              flush=True)
    queue = Queue()
    proc = Process(target=_limit_worker, args=(queue, eq, 'inf'))
    proc.start()
    proc.join(timeout=timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        lim_inf_str = "timeout"
        if verbose:
            print(f"  [Rank {rank}] idx={idx} TIMEOUT on lim(x->inf): {eq}",
                  flush=True)
    else:
        try:
            lim_inf_str = queue.get_nowait()
        except Exception:
            lim_inf_str = "unknown"

    return (idx, eq, lim_zero_str, lim_inf_str)


def compute_asymptotes(targetdir, comp, comm, skip_idx=None, verbose=False,
                       timeout=5):
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
    verbose : bool, optional
        If True, print each limit computation (helps debug stuck limits).
    timeout : int, optional
        Timeout in seconds for each limit computation. Default: 5.

    Returns
    -------
    dict
        Dictionary with asymptote statistics (only for non-skipped functions).
    """
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
            equations = cdmprof.fitting.load_equations(eq_path)
    else:
        equations = None

    equations = comm.bcast(equations, root=0)

    stats = {
        'zero_bad': 0,      # lim(x->0+) is 0 or negative
        'zero_param': 0,    # lim(x->0+) depends on parameters
        'zero_ok': 0,       # lim(x->0+) is positive (inf or const)
        'zero_unknown': 0,  # lim(x->0+) couldn't be computed
        'zero_timeout': 0,  # lim(x->0+) computation timed out
        'inf_bad': 0,       # lim(x->inf) is non-zero constant or inf
        'inf_param': 0,     # lim(x->inf) depends on parameters
        'inf_ok': 0,        # lim(x->inf) is 0
        'inf_unknown': 0,   # lim(x->inf) couldn't be computed
        'inf_timeout': 0,   # lim(x->inf) computation timed out
        'total_bad': 0,     # unique functions with ANY bad asymptote
        'total_param': 0,   # unique functions with param-dep (no bad)
        'total_unknown': 0,  # unknown asymptotes (post-fit check)
        'total_ok': 0,      # unique functions with all OK asymptotes
    }

    if equations is None:
        return stats

    # Distribute work across ranks
    n_equations = len(equations)
    local_results = []

    # Get indices for this rank
    local_indices = list(range(rank, n_equations, size))

    n_local = len(local_indices)
    for i, idx in enumerate(local_indices):
        result = _compute_asymptote(idx, equations[idx], verbose=verbose,
                                    rank=rank, timeout=timeout)
        local_results.append(result)

        # Progress indicator (each rank prints its own progress)
        if not verbose and ((i + 1) % 100 == 0 or (i + 1) == n_local):
            pct = 100 * (i + 1) / n_local
            print(f"  Rank {rank}: {i + 1}/{n_local} ({pct:.0f}%)", flush=True)

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
            elif lim_zero == "timeout":
                # Computation timed out - needs post-fit check
                stats['zero_timeout'] += 1
                zero_cat = 'unknown'
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
                        from sympy import N, sympify
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
            elif lim_inf == "timeout":
                # Computation timed out - needs post-fit check
                stats['inf_timeout'] += 1
                inf_cat = 'unknown'
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
                        from sympy import N, sympify
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
        (n_total, n_no_x, n_norm_only, n_bad, skip_idx).
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
            equations = cdmprof.fitting.load_equations(eq_path)
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


def print_summary(n_total, n_no_x, n_norm_only, n_bad, asymp_stats,
                  asymptotes_computed=True):
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
    print(f"\n2. Remove no x-dependence:       -{n_no_x:>5} ({pct:>5.1f}%)")
    print(f"   Remaining:                    {remaining:>6}")

    # Step 3: Normalization-only
    remaining2 = remaining - n_norm_only
    pct = 100 * n_norm_only / n_total if n_total > 0 else 0
    print(f"\n3. Remove norm-only:         -{n_norm_only:>5} ({pct:>5.1f}%)")
    print(f"   Remaining:                    {remaining2:>6}")

    # Step 4: Bad functions (trig, nan, inf)
    remaining3 = remaining2 - n_bad
    pct = 100 * n_bad / n_total if n_total > 0 else 0
    print(f"\n4. Remove bad (trig/nan/inf):    -{n_bad:>5} ({pct:>5.1f}%)")
    print(f"   Remaining:                    {remaining3:>6}")

    # Step 5: Asymptotes (only if computed)
    if asymptotes_computed:
        print("\n5. Asymptotic behavior analysis:")

        # Show x->0+ breakdown
        print("   lim(x->0+):")
        print(f"     <= 0 (rejected):       {asymp_stats['zero_bad']:>5}")
        print(f"     > 0, definite (OK):    {asymp_stats['zero_ok']:>5}")
        print(f"     param-dependent:       {asymp_stats['zero_param']:>5}")
        print(f"     unknown (post-fit):    {asymp_stats['zero_unknown']:>5}")
        print(f"     timeout (post-fit):    {asymp_stats['zero_timeout']:>5}")

        # Show x->inf breakdown
        print("   lim(x->inf):")
        print(f"     != 0 (rejected):       {asymp_stats['inf_bad']:>5}")
        print(f"     = 0, definite (OK):    {asymp_stats['inf_ok']:>5}")
        print(f"     param-dependent:       {asymp_stats['inf_param']:>5}")
        print(f"     unknown (post-fit):    {asymp_stats['inf_unknown']:>5}")
        print(f"     timeout (post-fit):    {asymp_stats['inf_timeout']:>5}")

        # Combined rejections (use total_bad to avoid double-counting)
        n_asymp_bad = asymp_stats['total_bad']
        remaining4 = remaining3 - n_asymp_bad
        pct = 100 * n_asymp_bad / n_total if n_total > 0 else 0
        print(
            f"\n   Rejected (bad asymp):     -{n_asymp_bad:>5} ({pct:>5.1f}%)")
        print(f"   Remaining:                    {remaining4:>6}")

        # Final summary
        n_param_dep = asymp_stats['total_param']
        n_unknown = asymp_stats['total_unknown']
        n_ready = asymp_stats['total_ok']
        print("\n" + "-" * 70)
        print("READY FOR FITTING:")
        print(f"  Definitely valid:              {n_ready:>6}")
        print(f"  Parameter-dependent:           {n_param_dep:>6}")
        print(f"  Unknown (post-fit check):      {n_unknown:>6}")
        print(f"  Total to fit:                  {remaining4:>6}")

        total_eliminated = n_total - remaining4
    else:
        print("\n5. Asymptotic behavior analysis: SKIPPED")
        print("   (use --asymptotes flag to enable)")
        remaining4 = remaining3

        print("\n" + "-" * 70)
        print("READY FOR FITTING:")
        print(f"  Total to fit:                  {remaining4:>6}")
        print("  (asymptote filtering will happen during fitting)")

        total_eliminated = n_total - remaining4

    pct_eliminated = 100 * total_eliminated / n_total if n_total > 0 else 0
    pct_remaining = 100 * remaining4 / n_total if n_total > 0 else 0
    print(f"\n  Eliminated:  {total_eliminated:>6} ({pct_eliminated:.1f}%)")
    print(f"  Kept:        {remaining4:>6} ({pct_remaining:.1f}%)")
    print("=" * 70)
    print("")


if __name__ == "__main__":
    from mpi4py import MPI

    parser = ArgumentParser()
    parser.add_argument("--runname", type=str,
                        help="ESR run name, defining the basis functions.")
    parser.add_argument("--comp", type=int, help="Function complexity.")
    parser.add_argument("--asymptotes", action="store_true",
                        help="Compute asymptotic limits (slow). "
                             "If not set, asymptote computation is skipped.")
    parser.add_argument("--asymp-timeout", type=int, default=5,
                        help="Timeout (seconds) for each asymptote limit "
                             "computation. Default: 5.")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    if rank == 0:
        print(f"\nGenerating functions for complexity {args.comp}...")
        print(f"Run name: {args.runname}")
        print(f"MPI ranks: {size}")
        if args.asymptotes:
            print(f"Asymptote computation: enabled (timeout={args.asymp_timeout}s)")  # noqa
        else:
            print("Asymptote computation: disabled")
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

    # Compute asymptotic behavior if requested
    if args.asymptotes:
        # Read config for verbose flag
        config = None
        if rank == 0:
            config = read_config()
            print("Computing asymptotes...")
        config = comm.bcast(config, root=0)
        asymp_verbose = config.get("fitting", {}).get("asymp_verbose", False)

        # Compute asymptotic behavior (all ranks participate)
        # Pass skip_idx so stats only count non-skipped functions
        asymp_stats = compute_asymptotes(targetdir, args.comp, comm, skip_idx,
                                         verbose=asymp_verbose,
                                         timeout=args.asymp_timeout)

        # Broadcast asymp_stats to rank 0
        asymp_stats = comm.bcast(asymp_stats, root=0)
    else:
        # Empty stats when asymptotes are skipped
        asymp_stats = {
            'zero_bad': 0, 'zero_param': 0, 'zero_ok': 0,
            'zero_unknown': 0, 'zero_timeout': 0,
            'inf_bad': 0, 'inf_param': 0, 'inf_ok': 0,
            'inf_unknown': 0, 'inf_timeout': 0,
            'total_bad': 0, 'total_param': 0, 'total_unknown': 0,
            'total_ok': 0,
        }

    # Print summary on rank 0
    if rank == 0:
        print_summary(n_total, n_no_x, n_norm_only, n_bad, asymp_stats,
                      asymptotes_computed=args.asymptotes)

        # Print output file locations
        print("Output files:")
        print(f"  {join(targetdir, f'unique_equations_{args.comp}.txt')}")
        print(f"  {join(targetdir, f'skip_functions_{args.comp}.txt')}")
        if args.asymptotes:
            print(f"  {join(targetdir, f'asymptotes_zero_{args.comp}.txt')}")
            print(f"  {join(targetdir, f'asymptotes_inf_{args.comp}.txt')}")
        print("")
