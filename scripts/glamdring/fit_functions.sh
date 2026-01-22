#!/bin/bash
#
# Submit fit_functions.py to glamdring queue.
#
# Usage:
#   ./fit_functions.sh <on_login> <nprocs> <complexity> <halos> [--resume]
#
# Arguments:
#   on_login   : 1 to run locally, 0 to submit to queue
#   nprocs     : Number of MPI processes
#   complexity : Equation complexity level
#   halos      : Path to folder containing halo data
#   --resume   : Optional flag to resume from existing results
#

memory=7
queue="berg"

on_login=${1}
nprocs=${2}
complexity=${3}
halos=${4}
resume_flag=${5}

# Check required arguments
if [ -z "$on_login" ] || [ -z "$nprocs" ] || [ -z "$complexity" ] || [ -z "$halos" ]; then
    echo "Usage: ./fit_functions.sh <on_login> <nprocs> <complexity> <halos> [--resume]"
    echo ""
    echo "Arguments:"
    echo "  on_login    1 to run locally, 0 to submit to queue (required)"
    echo "  nprocs      Number of MPI processes (required)"
    echo "  complexity  Equation complexity level (required)"
    echo "  halos       Path to folder containing halo data (required)"
    echo "  --resume    Resume from existing results (optional)"
    echo ""
    echo "Example:"
    echo "  ./fit_functions.sh 1 4 3 /path/to/halos"
    echo "  ./fit_functions.sh 0 32 5 /path/to/halos --resume"
    exit 1
fi

# Get paths relative to this script
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "$script_dir/../.." && pwd)"
file="$script_dir/../fit_functions.py"
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

pythoncm="$env $file --complexity $complexity --halos $halos"
if [ "$resume_flag" == "--resume" ]; then
    pythoncm="$pythoncm --resume"
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
