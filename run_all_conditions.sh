#!/usr/bin/env bash
# =============================================================================
# run_all_conditions.sh
# Runs run_sim.sh under three perception conditions on the same seeds:
#   nopred  - planner only, no prediction node
#   henav   - roslaunch perception_henav inference.launch
#   ours    - roslaunch perception inference.launch
# and prints a side-by-side summary.
#
# Each condition gets its own roscore, and its inference node is started once
# and stays loaded for all trials of all repeats of that condition. run_sim.sh
# starts and stops only the simulation (simple_run.launch) for each trial,
# which reuses the running master and prediction node.
#
# Run inside the drone container, from the workspace root:
#   docker exec -it drone bash -lc 'cd ~/HE-Nav && ./run_all_conditions.sh -n 100'
#
# Options:
#   -n  NUM_TRIALS   Trials per condition                    (default: 100)
#   -s  SEEDS_FILE   Seeds file, shared by all conditions     (default: seeds.txt)
#   -r  REPEATS      Repeat each condition this many times, with its model
#                    kept loaded between repeats            (default: 1)
#   -c  CONDITIONS   Comma-separated subset to run            (default: nopred,henav,ours)
#   -t  TIMEOUT      Per-trial timeout passed to run_sim.sh   (default: 120)
#   -o  OUT_PREFIX   Results go to <OUT_PREFIX>_<cond>[_r<k>] (default: results)
#   -d  LOAD_WAIT    Max seconds to wait for the model to load (default: 120)
#   -w  SETUP_BASH   Workspace setup file                      (default: devel/setup.bash)
# =============================================================================

set -o pipefail

NUM_TRIALS=100
SEEDS_FILE="seeds.txt"
REPEATS=1
CONDITIONS="nopred,henav,ours"
TIMEOUT=120
OUT_PREFIX="results"
LOAD_WAIT=120
SETUP_BASH="devel/setup.bash"

while getopts "n:s:r:c:t:o:d:w:" opt; do
  case $opt in
    n) NUM_TRIALS="$OPTARG" ;;
    s) SEEDS_FILE="$OPTARG" ;;
    r) REPEATS="$OPTARG" ;;
    c) CONDITIONS="$OPTARG" ;;
    t) TIMEOUT="$OPTARG" ;;
    o) OUT_PREFIX="$OPTARG" ;;
    d) LOAD_WAIT="$OPTARG" ;;
    w) SETUP_BASH="$OPTARG" ;;
    *) echo "Unknown option: -$OPTARG"; exit 1 ;;
  esac
done

cd "$(dirname "$(readlink -f "$0")")"
# shellcheck disable=SC1090
source "$SETUP_BASH"
set -u  # only after sourcing: ROS setup scripts read unset variables

# condition -> ROS package providing inference.launch ("" = no prediction)
declare -A PKG=( [nopred]="" [henav]="perception_henav" [ours]="perception" )

ROSCORE_PID=""
INFER_PID=""

stop_pid() {  # SIGINT, then SIGKILL if it does not exit
  local pid="$1"
  [[ -z "$pid" ]] && return
  kill -INT "$pid" 2>/dev/null || return
  for _ in $(seq 1 15); do kill -0 "$pid" 2>/dev/null || return; sleep 1; done
  kill -KILL "$pid" 2>/dev/null || true
}

cleanup() {
  stop_pid "$INFER_PID";   INFER_PID=""
  stop_pid "$ROSCORE_PID"; ROSCORE_PID=""
  pkill -f inference_ros_lbscnet.py 2>/dev/null || true
}
trap 'echo "[INTERRUPTED] Cleaning up..."; cleanup; exit 130' INT TERM

start_roscore() {
  if rostopic list >/dev/null 2>&1; then
    echo "[ERROR] A ROS master is already running. Stop it first so each condition starts clean."
    exit 1
  fi
  roscore > "$1/roscore.log" 2>&1 &
  ROSCORE_PID=$!
  for _ in $(seq 1 30); do rostopic list >/dev/null 2>&1 && return 0; sleep 1; done
  echo "[ERROR] roscore did not come up"; cleanup; exit 1
}

start_inference() {  # $1 = package, $2 = log file
  roslaunch "$1" inference.launch > "$2" 2>&1 &
  INFER_PID=$!
  echo "  Waiting for $1 model to load (max ${LOAD_WAIT}s)..."
  for _ in $(seq 1 "$LOAD_WAIT"); do
    if ! kill -0 "$INFER_PID" 2>/dev/null; then
      echo "[ERROR] $1 inference exited during startup, see $2"; cleanup; exit 1
    fi
    # Both nodes subscribe to the planner's raw map once the model is loaded.
    if rostopic info /grid_map/occupancy_inflate_raw 2>/dev/null | grep -q inference; then
      echo "  Model ready."; return 0
    fi
    sleep 1
  done
  echo "[ERROR] $1 inference not ready after ${LOAD_WAIT}s, see $2"; cleanup; exit 1
}

RUN_DIRS=()
IFS=',' read -ra CONDS <<< "$CONDITIONS"
for cond in "${CONDS[@]}"; do
  if [[ -z "${PKG[$cond]+x}" ]]; then echo "[ERROR] Unknown condition: $cond"; exit 1; fi
  base_dir="${OUT_PREFIX}_${cond}"   # roscore/inference logs live here
  mkdir -p "$base_dir"

  echo ""
  echo "############################################################"
  echo "  Condition: $cond  ->  $base_dir"
  echo "############################################################"

  # Load the model once; it stays up for every trial of every repeat.
  start_roscore "$base_dir"
  [[ -n "${PKG[$cond]}" ]] && start_inference "${PKG[$cond]}" "$base_dir/inference.log"

  for (( rep=1; rep<=REPEATS; rep++ )); do
    out_dir="$base_dir"
    (( REPEATS > 1 )) && out_dir="${base_dir}_r${rep}"
    mkdir -p "$out_dir"
    echo ""
    echo "  ---- $cond: repeat $rep/$REPEATS  ->  $out_dir"

    if [[ -n "${PKG[$cond]}" ]] && ! kill -0 "$INFER_PID" 2>/dev/null; then
      echo "[ERROR] $cond inference node died, see $base_dir/inference.log"; cleanup; exit 1
    fi

    ./run_sim.sh -n "$NUM_TRIALS" -s "$SEEDS_FILE" -t "$TIMEOUT" \
      -o "$out_dir/metrics.csv" -w "$SETUP_BASH" | tee "$out_dir/run_sim.out"
    RUN_DIRS+=("$out_dir")
  done

  cleanup
  sleep 3
done

# ── Side-by-side summary ─────────────────────────────────────────────────────
python3 - "${RUN_DIRS[@]}" <<'PYEOF'
import os, sys, csv, statistics as st
print("\n" + "=" * 96)
print(f"{'run':24s} {'success':>9s} {'collision':>9s} {'timeout':>8s} {'fly_t(s)':>9s} "
      f"{'plan(ms)':>9s} {'e-stops':>8s} {'pred_msgs':>10s}")
print("-" * 96)
for d in sys.argv[1:]:
    name = os.path.basename(d.rstrip("/"))
    try:
        rows = list(csv.DictReader(open(f"{d}/metrics.csv")))
    except FileNotFoundError:
        print(f"{name:24s} (no metrics.csv)"); continue
    if not rows:
        print(f"{name:24s} (empty)"); continue
    def col(k, rs=rows):
        out = []
        for r in rs:
            try: out.append(float(r[k]))
            except (KeyError, ValueError): pass
        return out
    mean = lambda v: st.mean(v) if v else float("nan")
    n = len(rows)
    ok = [r for r in rows if r["success"] == "1"]
    outc = lambda o: sum(r.get("outcome") == o for r in rows)
    pred = mean(col("prediction_messages"))
    print(f"{name:24s} {len(ok):4d}/{n:<4d} {outc('COLLISION'):9d} {outc('TIMEOUT'):8d} "
          f"{mean(col('flying_time_s', ok)):9.2f} {mean(col('avg_plan_time_ms')):9.2f} "
          f"{mean(col('emergency_stops')):8.2f} {pred:10.1f}")
    if "nopred" not in d and pred == 0:
        print(f"  [WARN] {name}: no prediction messages reached the planner")
print("=" * 96)
print("fly_t: mean over successful trials.  pred_msgs: mean prediction messages per trial.")
PYEOF
