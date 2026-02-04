#!/bin/bash
#
# Submit select_tng_halos.py to glamdring queue.
#
# Usage:
#   ./select_tng_halos.sh <on_login> <nprocs> <snap> [--extract] [--sphere-cut]
#

memory=7
queue="berg"

on_login=${1}
nprocs=${2}
snap=${3}
shift 3 2>/dev/null

extract_flag=false
sphere_cut_flag=false
for arg in "$@"; do
    case "$arg" in
        --extract) extract_flag=true ;;
        --sphere-cut) sphere_cut_flag=true ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

if [ -z "$on_login" ] || [ -z "$nprocs" ] || [ -z "$snap" ]; then
    echo "Usage: ./select_tng_halos.sh <on_login> <nprocs> <snap> [--extract] [--sphere-cut]"
    echo ""
    echo "Arguments:"
    echo "  on_login      1 to run locally, 0 to submit to queue"
    echo "  nprocs        Number of MPI processes"
    echo "  snap          Snapshot number (e.g., 99)"
    echo "  --extract     Extract particles for selected halos (optional)"
    echo "  --sphere-cut  Load ALL particles within R200c, not just FoF members (optional)"
    echo ""
    echo "Example:"
    echo "  ./select_tng_halos.sh 1 1 99"
    echo "  ./select_tng_halos.sh 0 16 99 --extract"
    echo "  ./select_tng_halos.sh 0 16 99 --extract --sphere-cut"
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
min_mass=1e12               # Msun/h
max_offset=0.1              # Fraction of R200c
max_satellite_ratio=0.005   # Max satellite/central mass ratio
isolation_distance=5.0      # In units of R200c
isolation_mass_ratio=0.1    # Reject if neighbor > ratio * M_self
subsample="10000"           # Set to e.g. 10000 to subsample particles
seed=42                     # Random seed for subsampling
check_monotonic=false       # Reject halos with non-monotonic outer profile

pythoncm="$env $file --basepath $basepath --snap $snap"
pythoncm="$pythoncm --min-mass $min_mass --max-offset $max_offset"
pythoncm="$pythoncm --max-satellite-ratio $max_satellite_ratio"
pythoncm="$pythoncm --isolation-distance $isolation_distance"
pythoncm="$pythoncm --isolation-mass-ratio $isolation_mass_ratio"
if [ "$extract_flag" == "true" ]; then
    pythoncm="$pythoncm --extract"
fi
if [ "$sphere_cut_flag" == "true" ]; then
    pythoncm="$pythoncm --sphere-cut"
fi
if [ -n "$subsample" ]; then
    pythoncm="$pythoncm --subsample $subsample --seed $seed"
fi
if [ "$check_monotonic" == "true" ]; then
    pythoncm="$pythoncm --check-monotonic"
fi

if [ "$on_login" -eq 1 ]; then
    cm="mpirun -n $nprocs $pythoncm"
    echo "Running locally:"
    echo $cm
    echo
    eval $cm
else
    cm="addqueue -q $queue -n $nprocs -m $memory $pythoncm"
    echo "Submitting:"
    echo $cm
    echo
    eval $cm
fi
