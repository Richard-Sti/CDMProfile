#!/bin/bash
#
# Submit select_tng_halos.py to glamdring queue.
#
# Usage:
#   ./select_tng_halos.sh <on_login> <nprocs> <snap> [--extract]
#

memory=7
queue="berg"

on_login=${1}
nprocs=${2}
snap=${3}
extract_flag=${4}

if [ -z "$on_login" ] || [ -z "$nprocs" ] || [ -z "$snap" ]; then
    echo "Usage: ./select_tng_halos.sh <on_login> <nprocs> <snap> [--extract]"
    echo ""
    echo "Arguments:"
    echo "  on_login    1 to run locally, 0 to submit to queue"
    echo "  nprocs      Number of MPI processes"
    echo "  snap        Snapshot number (e.g., 99)"
    echo "  --extract   Extract particles for selected halos (optional)"
    echo ""
    echo "Example:"
    echo "  ./select_tng_halos.sh 1 1 99"
    echo "  ./select_tng_halos.sh 0 16 99 --extract"
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "$script_dir/../.." && pwd)"
file="$script_dir/../select_tng_halos.py"
local_config="$project_dir/local_config.toml"

if [ ! -f "$local_config" ]; then
    echo "Error: local_config.toml not found at $local_config"
    exit 1
fi

venv=$(grep "venv" "$local_config" | sed 's/.*= *"//' | sed 's/".*//')

if [ -z "$venv" ]; then
    echo "Error: venv not found in $local_config"
    exit 1
fi

env="$venv/bin/python"

basepath="/mnt/extraspace/rstiskalek/TNG300-1-Dark/output"

# Selection criteria
min_mass=1e11               # Msun/h
max_offset=0.07             # Fraction of R200c
max_satellite_ratio=0.1     # Max satellite/central mass ratio
isolation_distance=5.0      # In units of R200c
isolation_mass_ratio=1.0    # Reject if neighbor > ratio * M_self
subsample=""                # Set to e.g. 10000 to subsample particles
seed=42                     # Random seed for subsampling

pythoncm="$env $file --basepath $basepath --snap $snap"
pythoncm="$pythoncm --min-mass $min_mass --max-offset $max_offset"
pythoncm="$pythoncm --max-satellite-ratio $max_satellite_ratio"
pythoncm="$pythoncm --isolation-distance $isolation_distance"
pythoncm="$pythoncm --isolation-mass-ratio $isolation_mass_ratio"
if [ "$extract_flag" == "--extract" ]; then
    pythoncm="$pythoncm --extract"
fi
if [ -n "$subsample" ]; then
    pythoncm="$pythoncm --subsample $subsample --seed $seed"
fi

mpicm="mpirun -n $nprocs $pythoncm"

if [ "$on_login" -eq 1 ]; then
    echo "Running locally:"
    echo $mpicm
    echo
    eval $mpicm
else
    cm="addqueue -q $queue -n $nprocs -m $memory $mpicm"
    echo "Submitting:"
    echo $cm
    echo
    eval $cm
fi
