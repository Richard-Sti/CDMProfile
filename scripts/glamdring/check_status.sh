#!/bin/bash
#
# Check fitting status for each snapshot × complexity combination.
#
# Usage:
#   ./check_status.sh [--runname <name>] [--snaps <list>] [--comps <list>]
#
# Defaults:
#   --runname  ext_maths_DM
#   --snaps    33,50,99
#   --comps    1,2,3,4,5,6,7
#

# Defaults
runname="ext_maths_DM"
snaps="33,50,99"
comps="1,2,3,4,5,6,7"

while [[ $# -gt 0 ]]; do
    case $1 in
        --runname) runname="$2"; shift 2 ;;
        --snaps)   snaps="$2";   shift 2 ;;
        --comps)   comps="$2";   shift 2 ;;
        *)         echo "Unknown: $1"; exit 1 ;;
    esac
done

# Resolve results directory from local_config.toml
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "$script_dir/../.." && pwd)"
local_config="$project_dir/local_config.toml"

if [ ! -f "$local_config" ]; then
    echo "Error: local_config.toml not found at $local_config"
    exit 1
fi

results=$(grep "results" "$local_config" | sed 's/.*= *"//' | sed 's/".*//')
if [ -z "$results" ]; then
    echo "Error: results path not found in $local_config"
    exit 1
fi

IFS=',' read -ra snap_list <<< "$snaps"
IFS=',' read -ra comp_list <<< "$comps"

# Header
printf "%-6s" ""
for comp in "${comp_list[@]}"; do
    printf "  %-6s" "c$comp"
done
echo ""

printf "%-6s" ""
for comp in "${comp_list[@]}"; do
    printf "  %-6s" "------"
done
echo ""

# One row per snapshot
for snap in "${snap_list[@]}"; do
    snap_pad=$(printf "%03d" "$snap")
    halos_name="particles_${snap_pad}"
    printf "s%-5s" "$snap_pad"

    for comp in "${comp_list[@]}"; do
        result="$results/fit_${runname}_compl${comp}_${halos_name}.hdf5"
        tmpdir="$results/tmp_fit_${runname}_compl${comp}_${halos_name}"
        cache="$results/cffi_cache_${runname}_compl${comp}_${halos_name}"

        if [ -f "$result" ]; then
            printf "  %-6s" "done"
        elif [ -d "$tmpdir" ] || [ -d "$cache" ]; then
            printf "  %-6s" "run.."
        else
            printf "  %-6s" "-"
        fi
    done
    echo ""
done
