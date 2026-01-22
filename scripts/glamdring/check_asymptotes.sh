#!/bin/bash
#
# Submit check_asymptotes.py to glamdring queue.
#
# Usage:
#   ./check_asymptotes.sh <on_login> <nprocs> <complexity> <results> [--top-n N]
#
# Arguments:
#   on_login   : 1 to run locally, 0 to submit to queue
#   nprocs     : Number of MPI processes
#   complexity : Equation complexity level
#   results    : Path to HDF5 results file from fit_functions.py
#   --top-n N  : Optional, override config top_n (number of functions)
#

memory=4
queue="berg"

on_login=${1}
nprocs=${2}
complexity=${3}
results=${4}
shift 4  # Shift past the required arguments to get optional ones

# Check required arguments
if [ -z "$on_login" ] || [ -z "$nprocs" ] || [ -z "$complexity" ] || [ -z "$results" ]; then
    echo "Usage: ./check_asymptotes.sh <on_login> <nprocs> <complexity> <results> [--top-n N]"
    echo ""
    echo "Arguments:"
    echo "  on_login    1 to run locally, 0 to submit to queue (required)"
    echo "  nprocs      Number of MPI processes (required)"
    echo "  complexity  Equation complexity level (required)"
    echo "  results     Path to HDF5 results file (required)"
    echo "  --top-n N   Override config top_n (optional)"
    echo ""
    echo "Example:"
    echo "  ./check_asymptotes.sh 1 4 3 /path/to/results.hdf5"
    echo "  ./check_asymptotes.sh 0 16 5 /path/to/results.hdf5 --top-n 200"
    exit 1
fi

# Get paths relative to this script
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "$script_dir/../.." && pwd)"
file="$script_dir/../check_asymptotes.py"
local_config="$project_dir/local_config.toml"

# Check local_config.toml exists
if [ ! -f "$local_config" ]; then
    echo "Error: local_config.toml not found at $local_config"
    echo ""
    echo "Please create it with:"
    echo "[path]"
    echo "results = \"/path/to/results\""
    echo "data = \"/path/to/data\""
    echo "venv = \"/path/to/venv\""
    exit 1
fi

# Read venv path from local_config.toml
venv=$(grep "venv" "$local_config" | sed 's/.*= *"//' | sed 's/".*//')

if [ -z "$venv" ]; then
    echo "Error: venv not found in $local_config"
    exit 1
fi

env="$venv/bin/python"

pythoncm="$env $file --complexity $complexity --results $results"

# Add optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --top-n)
            pythoncm="$pythoncm --top-n $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

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
