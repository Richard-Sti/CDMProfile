#!/bin/bash
set -euo pipefail

# ---- local source ----
SRC_BASE="$HOME/Projects/CDMprofile"

# ---- glamdring destination ----
DEST_USER="rstiskalek"
DEST_HOST="glamdring.physics.ox.ac.uk"
SSH_KEY="$HOME/.ssh/glamdring"

# Destination paths on glamdring (match glamdring's local_config.toml)
DEST_RESULTS="/mnt/users/rstiskalek/CDMProfile/results"
DEST_DATA="/mnt/extraspace/rstiskalek/CDMProfile/data"

usage() {
    echo "Usage: $0 [results|data]"
    exit 1
}

# ---- parse argument ----
if [[ $# -ne 1 ]]; then
    usage
fi

case "$1" in
    results)
        echo "[INFO] Syncing 'results' -> glamdring:${DEST_RESULTS}"
        rsync -avh --progress -e "ssh -i $SSH_KEY" \
          "$SRC_BASE/results/" \
          "$DEST_USER@$DEST_HOST:$DEST_RESULTS/"
        ;;
    data)
        echo "[INFO] Syncing 'data' -> glamdring:${DEST_DATA}"
        rsync -avh --progress -e "ssh -i $SSH_KEY" \
          "$SRC_BASE/data/" \
          "$DEST_USER@$DEST_HOST:$DEST_DATA/"
        ;;
    *)
        usage
        ;;
esac

echo "[INFO] Sync complete."
