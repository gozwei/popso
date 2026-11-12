#!/usr/bin/env bash
# Heap snapshot of a running solver (macOS): which allocations hold the memory at a given moment.
#
#   tools/memsnap.sh <seconds> <out-prefix> <pattern> -- <command>
#
# Starts the command (with malloc stack logging), waits <seconds> once, then writes for the newest process
# whose command line matches <pattern>:
#   <out-prefix>.heap     `heap -s`: allocations by size class (count, bytes)
#   <out-prefix>.stacks   `malloc_history -allBySize`: the call stacks of the largest allocations
# and waits for the command to end. Run solver commands through benchmarks/run.py --mem-limit so the job is capped.
set -u
secs=$1; out=$2; pat=$3; shift 3
[ "$1" = "--" ] && shift
MallocStackLogging=lite "$@" &
job=$!
python3 -c "import sys, time; time.sleep(float(sys.argv[1]))" "$secs"
pid=$(pgrep -n -f "$pat")
if [ -n "$pid" ]; then
  heap -s "$pid" > "$out.heap" 2>&1
  malloc_history "$pid" -allBySize > "$out.stacks" 2>&1
  echo "memsnap: pid $pid, $(ps -o rss= -p "$pid" | awk '{printf "%.0f MB resident", $1 / 1024}')"
else
  echo "memsnap: no process matches $pat"
fi
wait $job
