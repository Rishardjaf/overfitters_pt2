#!/usr/bin/env zsh
# Run a queue of v3 experiments sequentially, logging each to artifacts/v3/logs/<name>.log.
# Usage: scripts/v3/queue.sh "<name> <args...>" "<name> <args...>" ...
set -u
cd "$(dirname "$0")/../.."
mkdir -p artifacts/v3/logs
for spec in "$@"; do
  name=${spec%% *}
  args=${spec#* }
  echo "[$(date +%H:%M:%S)] start $name"
  eval uv run --no-sync python scripts/v3/run_experiment.py --name "$name" $args \
    > "artifacts/v3/logs/$name.log" 2>&1
  echo "[$(date +%H:%M:%S)] done  $name  $(grep -o 'Headline.*' artifacts/v3/logs/$name.log | tail -1 | cut -c1-80)"
done
echo QUEUE_DONE
