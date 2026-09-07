#!/usr/bin/env zsh
# Run a queue of v2 experiments sequentially, logging each to artifacts/v2/logs/<name>.log.
# Usage: scripts/v2/queue.sh "<name> <args...>" "<name> <args...>" ...
set -u
cd "$(dirname "$0")/../.."
mkdir -p artifacts/v2/logs
for spec in "$@"; do
  name=${spec%% *}
  args=${spec#* }
  echo "[$(date +%H:%M:%S)] start $name"
  eval uv run --no-sync python scripts/v2/run_experiment.py --name "$name" $args \
    > "artifacts/v2/logs/$name.log" 2>&1
  echo "[$(date +%H:%M:%S)] done  $name  $(grep -o 'Headline.*persistence [0-9.]*' artifacts/v2/logs/$name.log | tail -1)"
done
echo QUEUE_DONE
