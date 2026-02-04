#!/bin/bash
memory=21
queue="berg"

on_login=${1}
nthreads=${2}
complexities=${3}
runname=${4:-"ext_maths_DM"}

# Check required arguments
if [ -z "$on_login" ] || [ -z "$nthreads" ] || [ -z "$complexities" ]; then
    echo "Usage: ./generate_functions.sh <on_login> <nthreads> <complexities> [runname]"
    echo ""
    echo "Arguments:"
    echo "  on_login     1 to run locally, 0 to submit to queue (required)"
    echo "  nthreads     Number of threads (required)"
    echo "  complexities Function complexity or comma-separated list (required)"
    echo "  runname      ESR run name (default: ext_maths_DM)"
    echo ""
    echo "Note: Asymptote computation is SKIPPED by default (slow)."
    echo "      To enable, edit the script and add --asymptotes flag."
    echo ""
    echo "Example:"
    echo "  ./generate_functions.sh 1 4 5             # Run complexity 5 locally"
    echo "  ./generate_functions.sh 1 4 1,2,3,4,5     # Run complexities 1-5 locally"
    echo "  ./generate_functions.sh 0 8 10 my_run     # Submit complexity 10 to queue"
    exit 1
fi

# Get paths relative to this script
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "$script_dir/../.." && pwd)"
file="$script_dir/../generate_functions.py"
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

# Convert comma-separated list to array
IFS=',' read -ra COMP_ARRAY <<< "$complexities"

for complexity in "${COMP_ARRAY[@]}"; do
    pythoncm="$env $file --runname $runname --comp $complexity"

    if [ "$on_login" -eq 1 ]; then
        echo "Running locally (complexity $complexity):"
        echo $pythoncm
        echo
        eval $pythoncm
        echo
    else
        cm="addqueue -q $queue -n $nthreads -m $memory $pythoncm"
        echo "Submitting (complexity $complexity):"
        echo $cm
        echo
        eval $cm
    fi
done
