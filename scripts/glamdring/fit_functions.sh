#!/bin/bash
#
# Submit fit_functions.py to glamdring queue.
#
# Usage:
#   ./fit_functions.sh <on_login> <nprocs> <complexity> [--snap <num>] [--halos <path>] [--queue <name>] [--resume]
#
# Arguments:
#   on_login   : 1 to run locally, 0 to submit to queue
#   nprocs     : Number of MPI processes
#   complexity : Equation complexity level, or comma-separated list
#                (e.g. 3 or 3,4,5). Each level is a separate job.
#   --snap     : Optional snapshot number (default: 99)
#   --halos    : Optional path to halo data (overrides config.toml)
#   --queue    : Optional queue/node name (default: berg)
#   --resume   : Optional flag to resume from existing results
#

memory=3

on_login=${1}
nprocs=${2}
complexity=${3}
shift 3

# Parse optional arguments
snap="99"
halos=""
queue="berg"
resume_flag=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --snap)
            snap="$2"
            shift 2
            ;;
        --halos)
            halos="$2"
            shift 2
            ;;
        --queue)
            queue="$2"
            shift 2
            ;;
        --resume)
            resume_flag="--resume"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$on_login" ] || [ -z "$nprocs" ] || [ -z "$complexity" ]; then
    echo "Usage: ./fit_functions.sh <on_login> <nprocs> <complexity> [--snap <num>] [--halos <path>] [--queue <name>] [--resume]"
    echo ""
    echo "Arguments:"
    echo "  on_login    1 to run locally, 0 to submit to queue (required)"
    echo "  nprocs      Number of MPI processes (required)"
    echo "  complexity  Equation complexity level or comma-separated list (required)"
    echo "              A separate job is submitted for each level."
    echo "  --snap      Snapshot number (optional, default: 99)"
    echo "  --halos     Path to halo data (optional, overrides config.toml + --snap)"
    echo "  --queue     Queue/node name (optional, default: berg)"
    echo "  --resume    Resume from existing results (optional)"
    echo ""
    echo "Example:"
    echo "  ./fit_functions.sh 1 4 3"
    echo "  ./fit_functions.sh 0 32 3,4,5,6,7"
    echo "  ./fit_functions.sh 1 1 3 --snap 99"
    echo "  ./fit_functions.sh 0 32 5 --snap 50 --queue jaffe"
    echo "  ./fit_functions.sh 0 32 5 --resume"
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

# Split complexity on commas and submit one job per level.
IFS=',' read -ra comp_list <<< "$complexity"

for comp in "${comp_list[@]}"; do
    pythoncm="$env $file --complexity $comp --snap $snap"
    if [ -n "$halos" ]; then
        pythoncm="$pythoncm --halos $halos"
    fi
    if [ -n "$resume_flag" ]; then
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
done
