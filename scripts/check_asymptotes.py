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
MPI script for validating parameter-dependent asymptotes on top-ranked
functions.

Run after fit_functions.py to validate asymptotic behavior of the best
functions. Only functions with parameter-dependent or unknown asymptotes
are checked.
"""
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from time import time

import h5py
import numpy as np
from mpi4py import MPI

import cdmprof
from utils import (
    compute_function_scores,
    print_asymptote_summary,
    print_best_results,
    read_config,
)

# MPI tags
WORK_TAG = 1
DONE_TAG = 0


def load_params_for_function(results_path, func_idx, n_sample=None, seed=42):
    """
    Load fitted parameters for a function from HDF5 results.

    Parameters
    ----------
    results_path : Path
        Path to HDF5 results file.
    func_idx : int
        Function index to load.
    n_sample : int, optional
        Number of halos to sample. If None or >= total, return all.
    seed : int
        Random seed for sampling.

    Returns
    -------
    list of tuples
        List of (halo_idx, params) for the function.
    """
    with h5py.File(results_path, 'r') as f:
        all_func_idx = f['func_idx'][:]
        all_halo_idx = f['halo_idx'][:]
        all_params = f['params'][:]

    # Find results for this function
    mask = all_func_idx == func_idx
    halo_idx = all_halo_idx[mask]
    params = all_params[mask]

    results = [(h, p) for h, p in zip(halo_idx, params)]

    # Sample if requested
    if n_sample is not None and n_sample > 0 and n_sample < len(results):
        rng = np.random.default_rng(seed=seed)
        indices = rng.choice(len(results), size=n_sample, replace=False)
        results = [results[i] for i in sorted(indices)]

    return results


def check_function_asymptotes(func_idx, expr_str, results_path, config):
    """
    Check asymptotes for a single function.

    Returns
    -------
    tuple
        (func_idx, n_pass_inf, n_pass_zero, n_checked)
    """
    n_halos = config.get('n_halos', 20)
    round_dec = config.get('round_decimals', 4)
    timeout = config.get('timeout', 0)
    verbose = config.get('verbose', False)

    # Load params for this function
    params_list = load_params_for_function(
        results_path, func_idx, n_sample=n_halos, seed=42 + func_idx)

    if len(params_list) == 0:
        return func_idx, 0, 0, 0

    n_pass_inf = 0
    n_pass_zero = 0
    n_checked = len(params_list)

    if verbose:
        print(f"  func {func_idx}: checking {n_checked} halos for "
              f"'{expr_str}'", flush=True)

    for halo_idx, params in params_list:
        lim_zero, lim_inf = cdmprof.compute_asymptotes_with_params(
            expr_str, params, round_decimals=round_dec, timeout=timeout)

        if cdmprof.asymptotes.check_asymptote(lim_inf, "inf"):
            n_pass_inf += 1
        if cdmprof.asymptotes.check_asymptote(lim_zero, "zero"):
            n_pass_zero += 1

        if verbose:
            print(f"    halo {halo_idx}: inf={lim_inf}, zero={lim_zero}",
                  flush=True)

    return func_idx, n_pass_inf, n_pass_zero, n_checked


def master_loop(comm, job_queue, batch_size):
    """Master process: distribute work to workers."""
    size = comm.Get_size()
    n_workers = size - 1

    job_queue = list(job_queue)
    n_total = len(job_queue)
    workers_done = 0

    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] Master: {n_total} functions to check, "
          f"{n_workers} workers, batch size {batch_size}", flush=True)

    while workers_done < n_workers:
        status = MPI.Status()
        worker_rank = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG,
                                status=status)

        now = datetime.now().strftime("%H:%M:%S")
        if len(job_queue) > 0:
            batch = job_queue[:batch_size]
            job_queue = job_queue[batch_size:]
            comm.send(batch, dest=worker_rank, tag=WORK_TAG)
            print(f"[{now}] Master: sent {len(batch)} jobs to "
                  f"rank {worker_rank}, {len(job_queue)} remaining",
                  flush=True)
        else:
            comm.send(None, dest=worker_rank, tag=DONE_TAG)
            workers_done += 1
            print(f"[{now}] Master: rank {worker_rank} done, "
                  f"{n_workers - workers_done} workers remaining", flush=True)


def worker_loop(comm, equations, results_path, config):
    """Worker process: check asymptotes for assigned functions."""
    rank = comm.Get_rank()
    verbose = config.get('verbose', False)

    results = []

    while True:
        comm.send(rank, dest=0)
        status = MPI.Status()
        batch = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

        if status.tag == DONE_TAG:
            break

        for func_idx in batch:
            expr_str = equations[func_idx]
            result = check_function_asymptotes(
                func_idx, expr_str, results_path, config)
            results.append(result)

            if verbose:
                _, n_inf, n_zero, n_checked = result
                print(f"Rank {rank}: func {func_idx} done - "
                      f"inf={n_inf}/{n_checked}, zero={n_zero}/{n_checked}",
                      flush=True)

    return results


def update_results_file(results_path, asymp_results, inf_threshold,
                        zero_threshold):
    """
    Update HDF5 results with asymptote validation results.

    Parameters
    ----------
    results_path : Path
        Path to HDF5 results file.
    asymp_results : list of tuples
        List of (func_idx, n_pass_inf, n_pass_zero, n_checked).
    inf_threshold : float
        Threshold for x->inf pass rate.
    zero_threshold : float
        Threshold for x->0+ pass rate.
    """
    # Compute rejections
    reject_funcs = set()
    for func_idx, n_inf, n_zero, n_checked in asymp_results:
        if n_checked == 0:
            continue
        frac_inf = n_inf / n_checked
        frac_zero = n_zero / n_checked
        if frac_inf < inf_threshold or frac_zero < zero_threshold:
            reject_funcs.add(func_idx)

    with h5py.File(results_path, 'a') as f:
        # Update/create asymptote pass fractions
        for name in ['asymp_pass_func_idx', 'asymp_pass_inf',
                     'asymp_pass_zero', 'asymp_pass_total']:
            if name in f:
                del f[name]

        n_funcs = len(asymp_results)
        if n_funcs > 0:
            apf_func_idx = np.zeros(n_funcs, dtype=np.int32)
            apf_pass_inf = np.zeros(n_funcs, dtype=np.int32)
            apf_pass_zero = np.zeros(n_funcs, dtype=np.int32)
            apf_total = np.zeros(n_funcs, dtype=np.int32)

            for i, (fidx, n_inf, n_zero, n_checked) in enumerate(
                    sorted(asymp_results)):
                apf_func_idx[i] = fidx
                apf_pass_inf[i] = n_inf
                apf_pass_zero[i] = n_zero
                apf_total[i] = n_checked

            f.create_dataset('asymp_pass_func_idx', data=apf_func_idx)
            f.create_dataset('asymp_pass_inf', data=apf_pass_inf)
            f.create_dataset('asymp_pass_zero', data=apf_pass_zero)
            f.create_dataset('asymp_pass_total', data=apf_total)

        # Update/create asymptote rejection list
        if 'asymp_reject_func_idx' in f:
            del f['asymp_reject_func_idx']
        if len(reject_funcs) > 0:
            reject_arr = np.array(sorted(reject_funcs), dtype=np.int32)
            f.create_dataset('asymp_reject_func_idx', data=reject_arr)

    return reject_funcs


if __name__ == "__main__":
    config = read_config()
    asymp_config = config.get('asymptotes', {})
    fit_config = config['fitting']

    parser = ArgumentParser(
        description="Validate asymptotes for top-ranked functions")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to HDF5 results file from fit_functions.py")
    parser.add_argument("--complexity", type=int, required=True,
                        help="Equation complexity level")
    parser.add_argument("--top-n", type=int, default=None,
                        help="Override config top_n (number of functions)")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    # Construct equations and asymptote paths from config
    runname = fit_config['runname']
    results_base = Path(config['path']['results'])
    equations_dir = results_base / runname / f"compl_{args.complexity}"
    equations_path = equations_dir / f"unique_equations_{args.complexity}.txt"
    asymp_inf_path = equations_dir / f"asymptotes_inf_{args.complexity}.txt"
    asymp_zero_path = equations_dir / f"asymptotes_zero_{args.complexity}.txt"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Load equations
    equations = cdmprof.fitting.load_equations(equations_path)

    # Load asymptote info
    if rank == 0:
        (_, asymp_inf_param_dep, asymp_zero_param_dep,
         asymp_unknown) = cdmprof.asymptotes.process_asymptotes(
            asymp_inf_path, asymp_zero_path)

        # Combine all functions needing post-fit check
        needs_check = (set(asymp_inf_param_dep.keys()) |
                       set(asymp_zero_param_dep.keys()) |
                       asymp_unknown)
        print(f"Functions needing asymptote check: {len(needs_check)}",
              flush=True)
    else:
        needs_check = None

    needs_check = comm.bcast(needs_check, root=0)

    # Load npart_per_halo for ranking
    if rank == 0:
        with h5py.File(results_path, 'r') as f:
            halo_idx = f['halo_idx'][:]
            n_halos = int(np.max(halo_idx)) + 1
            # Compute npart from bin_counts if available, else use loss counts
            if 'npart' in f.attrs:
                npart_per_halo = f.attrs['npart']
            else:
                # Estimate npart from loss - each halo appears once per func
                unique_funcs = np.unique(f['func_idx'][:])
                npart_per_halo = np.ones(n_halos) * 10000  # fallback

        # Compute rankings
        top_n = args.top_n or asymp_config.get('top_n', 100)
        scores, _, _, _ = compute_function_scores(
            results_path, npart_per_halo, min_success_fraction=0.0)

        # Select top N functions that need asymptote check
        job_queue = []
        for fidx, score, n_halos in scores[:top_n]:
            if fidx in needs_check:
                job_queue.append(fidx)

        print(f"Top {top_n} functions: {len(job_queue)} need asymptote check",
              flush=True)
    else:
        job_queue = None
        npart_per_halo = None

    job_queue = comm.bcast(job_queue, root=0)
    npart_per_halo = comm.bcast(npart_per_halo, root=0)

    if len(job_queue) == 0:
        if rank == 0:
            print("No functions need asymptote checking. Done.", flush=True)
        comm.Barrier()
        exit(0)

    # Get config values
    inf_threshold = asymp_config.get('inf_threshold', 0.5)
    zero_threshold = asymp_config.get('zero_threshold', 0.5)
    batch_size = asymp_config.get('batch_size', 10)

    t_start = time()

    if size == 1:
        # Single process mode
        print("Running in single-process mode", flush=True)
        all_results = []
        for i, func_idx in enumerate(job_queue):
            expr_str = equations[func_idx]
            print(f"[{i+1}/{len(job_queue)}] Checking '{expr_str}'",
                  flush=True)
            result = check_function_asymptotes(
                func_idx, expr_str, results_path, asymp_config)
            all_results.append(result)
            _, n_inf, n_zero, n_checked = result
            print(f"  Result: inf={n_inf}/{n_checked}, "
                  f"zero={n_zero}/{n_checked}", flush=True)
    else:
        # MPI mode
        if rank == 0:
            master_loop(comm, job_queue, batch_size)
            all_results = None
        else:
            all_results = worker_loop(
                comm, equations, results_path, asymp_config)

        # Gather results to rank 0
        gathered = comm.gather(all_results, root=0)

        if rank == 0:
            all_results = []
            for worker_results in gathered:
                if worker_results is not None:
                    all_results.extend(worker_results)

    comm.Barrier()

    if rank == 0:
        total_time = time() - t_start
        print(f"\nCompleted {len(all_results)} asymptote checks in "
              f"{total_time:.1f}s", flush=True)

        # Update results file
        reject_funcs = update_results_file(
            results_path, all_results, inf_threshold, zero_threshold)

        # Print summary
        n_param_dep = len(job_queue)
        n_rejected = len(reject_funcs)
        print_asymptote_summary(n_param_dep, 0, n_rejected,
                                inf_threshold, zero_threshold)

        # Print updated rankings
        print_best_results(results_path, equations, npart_per_halo,
                           nfw_score=None, min_success_fraction=0.0)

        print(f"\nUpdated results saved to: {results_path}", flush=True)
