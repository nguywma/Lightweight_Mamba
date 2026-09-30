#!/usr/bin/env bash
# =============================================================================
# run_benchmark.sh
# Runs the EGO-Planner simulation N times, each with a different random seed
# loaded from a seeds file, parses per-trial metrics from ROS logs, writes
# every trial to a CSV, and prints the averaged results at the end.
#
# Usage:
#   chmod +x run_benchmark.sh
#   ./run_benchmark.sh [OPTIONS]
#
# Options:
#   -n  NUM_TRIALS   Number of trials to run          (default: 100)
#   -s  SEEDS_FILE   Path to file with one seed/line  (default: seeds.txt)
#   -o  OUTPUT_CSV   Path for per-trial CSV output    (default: results/metrics.csv)
#   -t  TIMEOUT      Max seconds to wait per trial    (default: 120)
#   -p  ROS_PKG      ROS package name                 (default: ego_planner)
#   -l  LAUNCH_FILE  Launch file name inside that pkg  (default: simple_run.launch)
#   -w  SETUP_BASH   Path to ROS workspace setup.bash (default: devel/setup.bash)
#
# Seeds file format (one integer per line, lines starting with # are ignored):
#   42
#   137
#   999
#   ...
#
# Dependencies: ROS (roslaunch, rostopic), awk, python3
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
NUM_TRIALS=100
SEEDS_FILE="seeds.txt"
OUTPUT_CSV="results/metrics.csv"
TIMEOUT=120
ROS_PKG="ego_planner"
LAUNCH_FILE="simple_run.launch"
SETUP_BASH="devel/setup.bash"

# ── Argument parsing ──────────────────────────────────────────────────────────
while getopts "n:s:o:t:l:w:p:" opt; do
  case $opt in
    n) NUM_TRIALS="$OPTARG" ;;
    s) SEEDS_FILE="$OPTARG" ;;
    o) OUTPUT_CSV="$OPTARG" ;;
    t) TIMEOUT="$OPTARG" ;;
    l) LAUNCH_FILE="$OPTARG" ;;
    p) ROS_PKG="$OPTARG" ;;
    w) SETUP_BASH="$OPTARG" ;;
    *) echo "Unknown option: -$OPTARG"; exit 1 ;;
  esac
done

# ── Source ROS workspace ──────────────────────────────────────────────────────
if [[ ! -f "$SETUP_BASH" ]]; then
  echo "[ERROR] ROS workspace setup file not found: $SETUP_BASH"
  echo "        Build your workspace first:  catkin_make  (or catkin build)"
  echo "        Or pass a custom path with:  -w /path/to/devel/setup.bash"
  exit 1
fi

# shellcheck disable=SC1090
source "$SETUP_BASH"
echo "[OK] Sourced ROS workspace: $SETUP_BASH"
echo "     ROS_PACKAGE_PATH: $ROS_PACKAGE_PATH"
echo "no retry "

# ── Validate remaining inputs ─────────────────────────────────────────────────
if [[ ! -f "$SEEDS_FILE" ]]; then
  echo "[WARN] Seeds file not found: $SEEDS_FILE — auto-generating $NUM_TRIALS seeds..."
  python3 - "$SEEDS_FILE" "$NUM_TRIALS" << 'PYEOF'
import sys, random
out_path  = sys.argv[1]
n         = int(sys.argv[2])
rng       = random.Random()          # non-deterministic master seed
seeds     = [rng.randint(0, 99999) for _ in range(n)]
with open(out_path, "w") as fh:
    fh.write("# Auto-generated seeds file\n")
    fh.write(f"# {n} random seeds\n\n")
    for s in seeds:
        fh.write(f"{s}\n")
print(f"[OK] Written {n} seeds to '{out_path}'")
PYEOF
fi

PKG_PATH=$(rospack find "$ROS_PKG" 2>/dev/null || true)
if [[ -z "$PKG_PATH" ]]; then
  echo "[ERROR] ROS package '$ROS_PKG' not found. Is the workspace built and sourced?"
  exit 1
fi
LAUNCH_PATH="$PKG_PATH/launch/$LAUNCH_FILE"
if [[ ! -f "$LAUNCH_PATH" ]]; then
  echo "[ERROR] Launch file not found: $LAUNCH_PATH"
  echo "        Available launch files in $ROS_PKG:"
  ls "$PKG_PATH/launch/" 2>/dev/null || echo "        (no launch/ directory found)"
  exit 1
fi
echo "[OK] Launch file found: $LAUNCH_PATH"

# Read seeds (skip blank lines and comments)
mapfile -t ALL_SEEDS < <(grep -v '^\s*#' "$SEEDS_FILE" | grep -v '^\s*$')
TOTAL_SEEDS=${#ALL_SEEDS[@]}

if [[ $TOTAL_SEEDS -lt $NUM_TRIALS ]]; then
  echo "[ERROR] Seeds file has only $TOTAL_SEEDS seeds, but NUM_TRIALS=$NUM_TRIALS"
  echo "        Add more seeds or reduce -n."
  exit 1
fi

# ── Setup output directory ────────────────────────────────────────────────────
OUTPUT_DIR=$(dirname "$OUTPUT_CSV")
mkdir -p "$OUTPUT_DIR"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

# ── CSV header ────────────────────────────────────────────────────────────────
CSV_HEADER="trial,seed,success,flying_time_s,total_dist_m,max_vel_ms,avg_vel_ms,energy_jerk,mission_plan_time_s,avg_plan_time_ms,overall_success_rate_pct,progress_time_errors,collision_count,outcome,gt_collision,min_clearance_m,mean_clearance_m,time_in_danger_s,clearance_valid,straight_line_m,path_efficiency,prediction_messages,prediction_voxels,prediction_opportunities,prediction_avoiding_replans,prediction_only_contacts,prediction_age_s,emergency_stops"
echo "$CSV_HEADER" > "$OUTPUT_CSV"

echo "============================================================"
echo "  EGO-Planner Benchmark"
echo "  Trials     : $NUM_TRIALS"
echo "  Seeds file : $SEEDS_FILE"
echo "  Package    : $ROS_PKG"
echo "  Launch file: $LAUNCH_FILE"
echo "  Setup bash : $SETUP_BASH"
echo "  Timeout    : ${TIMEOUT}s per trial"
echo "  Output CSV : $OUTPUT_CSV"
echo "  Log dir    : $LOG_DIR"
echo "============================================================"

# ── Helper: kill all ROS nodes cleanly ───────────────────────────────────────
cleanup_ros() {
  # Kill roslaunch and all child processes
  if [[ -n "${LAUNCH_PID:-}" ]]; then
    kill -TERM "$LAUNCH_PID" 2>/dev/null || true
    wait "$LAUNCH_PID" 2>/dev/null || true
    unset LAUNCH_PID
  fi
  # Give nodes time to die, then force-kill any stragglers
  sleep 2
  pkill -f "ego_planner" 2>/dev/null || true
  pkill -f "traj_server"  2>/dev/null || true
  pkill -f "pcl_render"   2>/dev/null || true
#   pkill -f "roslaunch"    2>/dev/null || true
  sleep 1
}

# ── Helper: parse metrics from a single trial log file ───────────────────────
# Reads the LAST "Navigation Metrics" block printed by printMetrics().
# Returns a pipe-separated string with trajectory, planner, and prediction metrics.
parse_metrics() {
  local logfile="$1"

  python3 - "$logfile" <<'PYEOF'
import sys, re

logfile = sys.argv[1]
try:
    with open(logfile, "r", errors="replace") as f:
        content = f.read()
except FileNotFoundError:
    print("0|0|0|0|0|0|0|0|0|0|NONE|0|-1|-1|0|0|0|-1|0|0|0|0|0|0|0")
    sys.exit(0)

# Find all Navigation Metrics blocks and take the LAST one
# (the FSM prints one per mission completion)
blocks = re.findall(
    r"={5,} Navigation Metrics ={5,}(.*?)={5,}",
    content, re.DOTALL
)

if not blocks:
    print("0|0|0|0|0|0|0|0|0|0|NONE|0|-1|-1|0|0|0|-1|0|0|0|0|0|0|0")
    sys.exit(0)

block = blocks[-1]

def get(pattern, default="0"):
    m = re.search(pattern, block)
    return m.group(1).strip() if m else default

# SUCCESS or FAIL
success_match = re.search(r"Mission #\d+:\s*(SUCCESS|FAIL)", block)
success = "1" if (success_match and success_match.group(1) == "SUCCESS") else "0"

flying_time  = get(r"Flying Time:\s*([\d.]+)")
total_dist   = get(r"Total Distance:\s*([\d.]+)")
max_vel      = get(r"Velocity - Max:\s*([\d.]+)")
avg_vel      = get(r"Avg:\s*([\d.]+)")
energy_jerk  = get(r"Energy \(Jerk Integral\):\s*([\d.eE+\-]+)")
plan_time    = get(r"Planning Time \(this mission\):\s*([\d.]+)")
avg_plan_ms  = get(r"Avg:\s*([\d.]+)\s*ms")
pred = re.search(r"Prediction Metrics:\s*messages=(\d+)\s+voxels=(\d+)\s+opportunities=(\d+)\s+avoiding_replans=(\d+)\s+prediction_only_contacts=(\d+)\s+age_s=([\d.eE+\-]+)", block)
prediction_metrics = pred.groups() if pred else ("0", "0", "0", "0", "0", "0")

# Collision count for this mission (printed inside the Navigation Metrics block)
collisions = get(r"Collisions:\s*(\d+)")
emergency_stops = get(r"Emergency Stops:\s*(\d+)")

# Outcome label: REACHED / COLLISION / TIMEOUT / ABORTED
outcome = get(r"Outcome:\s*(\w+)", "NONE")

# Ground-truth safety metrics (graded). -1 => no ground-truth cloud received.
safety = re.search(
    r"Safety Metrics:\s*gt_collision=(\d+)\s+min_clearance_m=([\d.eE+\-]+)"
    r"\s+mean_clearance_m=([\d.eE+\-]+)\s+time_in_danger_s=([\d.eE+\-]+)"
    r"\s+clearance_valid=(\d+)", block)
if safety:
    gt_collision, min_clear, mean_clear, time_danger, clear_valid = safety.groups()
else:
    gt_collision, min_clear, mean_clear, time_danger, clear_valid = "0", "-1", "-1", "0", "0"

path = re.search(r"Path Metrics:\s*straight_line_m=([\d.eE+\-]+)\s+path_efficiency=([\d.eE+\-]+)", block)
straight_line, path_eff = path.groups() if path else ("0", "-1")

# Count last_progress_time_ errors
progress_errors = len(re.findall(r"last_progress_time_ ERROR", content))

safety_metrics = (outcome, gt_collision, min_clear, mean_clear, time_danger, clear_valid, straight_line, path_eff)
print(f"{success}|{flying_time}|{total_dist}|{max_vel}|{avg_vel}|{energy_jerk}|{plan_time}|{avg_plan_ms}|{progress_errors}|{collisions}|{'|'.join(safety_metrics)}|{'|'.join(prediction_metrics)}|{emergency_stops}")
PYEOF
}

# ── Main trial loop ───────────────────────────────────────────────────────────
trap 'echo "[INTERRUPTED] Cleaning up..."; cleanup_ros; exit 130' INT TERM

PASS=0
FAIL=0

for (( i=1; i<=NUM_TRIALS; i++ )); do
  SEED="${ALL_SEEDS[$((i-1))]}"
  LOG_FILE="$LOG_DIR/trial_${i}_seed_${SEED}.log"

  echo ""
  echo "────────────────────────────────────────────────────────────"
  echo "  Trial $i / $NUM_TRIALS  |  seed=$SEED"
  echo "────────────────────────────────────────────────────────────"

  # Launch ROS simulation in background, redirect all output to log
  roslaunch "$ROS_PKG" "$LAUNCH_FILE" map_seed:="$SEED" flight_type:=2 \
    > "$LOG_FILE" 2>&1 &
  LAUNCH_PID=$!

  # Wait for mission complete or timeout
  ELAPSED=0
  DONE=0
  while (( ELAPSED < TIMEOUT )); do
    sleep 1
    (( ELAPSED++ )) || true

    if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
      DONE=1; break
    fi

    if grep -q "Navigation Metrics" "$LOG_FILE" 2>/dev/null; then
      sleep 2
      DONE=1; break
    fi
  done

  if (( DONE == 0 )); then
    echo "  [TIMEOUT] Trial $i exceeded ${TIMEOUT}s — marking as FAIL"
    echo "TIMEOUT" >> "$LOG_FILE"
  fi

  cleanup_ros

  # Parse metrics from log
  RAW=$(parse_metrics "$LOG_FILE")
  IFS='|' read -r success flying_time total_dist max_vel avg_vel energy_jerk plan_time avg_plan_ms progress_errors collisions outcome gt_collision min_clearance mean_clearance time_in_danger clearance_valid straight_line path_efficiency prediction_messages prediction_voxels prediction_opportunities prediction_avoiding_replans prediction_only_contacts prediction_age_s emergency_stops <<< "$RAW"

  if [[ $success == "1" ]]; then
    (( PASS++ )) || true
    STATUS="SUCCESS"
  else
    (( FAIL++ )) || true
    STATUS="FAIL"
  fi

  # Overall success rate so far
  SR=$(awk "BEGIN {printf \"%.1f\", $PASS / $i * 100}")

  # Append to CSV
  echo "$i,$SEED,$success,$flying_time,$total_dist,$max_vel,$avg_vel,$energy_jerk,$plan_time,$avg_plan_ms,$SR,$progress_errors,$collisions,$outcome,$gt_collision,$min_clearance,$mean_clearance,$time_in_danger,$clearance_valid,$straight_line,$path_efficiency,$prediction_messages,$prediction_voxels,$prediction_opportunities,$prediction_avoiding_replans,$prediction_only_contacts,$prediction_age_s,$emergency_stops" \
    >> "$OUTPUT_CSV"

  echo "  Result : $STATUS ($outcome)"
  echo "  Flying : ${flying_time}s  |  Dist: ${total_dist}m  |  MaxVel: ${max_vel} m/s"
  echo "  Energy (jerk): $energy_jerk  |  PlanTime: ${plan_time}s (avg ${avg_plan_ms}ms)"
  echo "  Safety : gt_collision=$gt_collision  min_clear=${min_clearance}m  mean_clear=${mean_clearance}m  time_in_danger=${time_in_danger}s  (valid=$clearance_valid)"
  echo "  Path   : straight_line=${straight_line}m  path_efficiency=$path_efficiency"
  echo "  Collisions: $collisions  |  Emergency stops: $emergency_stops  |  Pred: msgs=$prediction_messages voxels=$prediction_voxels opp=$prediction_opportunities avoid=$prediction_avoiding_replans only_contacts=$prediction_only_contacts age=${prediction_age_s}s"
  if [[ "$progress_errors" -gt 0 ]]; then
    echo "  [WARN] last_progress_time_ errors: $progress_errors (planner used stale target — metrics may be unreliable)"
  fi
  echo "  Running success rate: $PASS/$i ($SR%)"
done

# ── Compute and print averages ─────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  BENCHMARK COMPLETE — AVERAGED RESULTS"
echo "  ($NUM_TRIALS trials, seeds from $SEEDS_FILE)"
echo "============================================================"

python3 - "$OUTPUT_CSV" "$NUM_TRIALS" "$PASS" "$FAIL" <<'PYEOF'
import sys, csv, statistics

csv_file   = sys.argv[1]
n_trials   = int(sys.argv[2])
n_pass     = int(sys.argv[3])
n_fail     = int(sys.argv[4])

rows = []
with open(csv_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

def col_floats(key):
    vals = []
    for r in rows:
        try:
            vals.append(float(r[key]))
        except (ValueError, KeyError):
            pass
    return vals

def mean(lst):
    return statistics.mean(lst) if lst else 0.0

def stdev(lst):
    return statistics.stdev(lst) if len(lst) > 1 else 0.0

# Only average over successful trials for time/distance/energy metrics
success_rows = [r for r in rows if r.get("success") == "1"]

def scol(key):
    vals = []
    for r in success_rows:
        try:
            vals.append(float(r[key]))
        except (ValueError, KeyError):
            pass
    return vals

fly_times   = scol("flying_time_s")
dists       = scol("total_dist_m")
max_vels    = scol("max_vel_ms")
avg_vels    = scol("avg_vel_ms")
energies    = scol("energy_jerk")
plan_times  = col_floats("mission_plan_time_s")   # all trials (including fails)
avg_plan_ms = col_floats("avg_plan_time_ms")

success_rate = n_pass / n_trials * 100.0

print(f"\n  Total trials           : {n_trials}")
print(f"  Success                : {n_pass}  ({success_rate:.1f}%)")
print(f"  Fail (collision/timeout): {n_fail}  ({100-success_rate:.1f}%)")
print()
print(f"  ── Metrics averaged over {len(success_rows)} successful trials ──")
print(f"  Flying Time            : {mean(fly_times):.3f} s  ± {stdev(fly_times):.3f}")
print(f"  Total Distance         : {mean(dists):.3f} m  ± {stdev(dists):.3f}")
print(f"  Max Velocity           : {mean(max_vels):.3f} m/s ± {stdev(max_vels):.3f}")
print(f"  Avg Velocity           : {mean(avg_vels):.3f} m/s ± {stdev(avg_vels):.3f}")
print(f"  Energy (Jerk Integral) : {mean(energies):.4f}  ± {stdev(energies):.4f}")
print()
print(f"  ── Planning time (all {n_trials} trials) ──")
print(f"  Mission Plan Time      : {mean(plan_times):.3f} s  ± {stdev(plan_times):.3f}")
print(f"  Avg Plan Time per call : {mean(avg_plan_ms):.3f} ms ± {stdev(avg_plan_ms):.3f}")
print()

prog_errors = col_floats("progress_time_errors")
trials_with_errors = sum(1 for v in prog_errors if v > 0)
total_errors = sum(prog_errors)
print(f"  ── last_progress_time_ errors ──")
print(f"  Trials with errors     : {trials_with_errors} / {n_trials}")
print(f"  Total error count      : {int(total_errors)}")
if trials_with_errors > 0:
    print(f"  [WARN] These trials had stale local targets — treat their metrics with caution")
print()
PYEOF

echo "  Per-trial CSV : $OUTPUT_CSV"
echo "  Raw logs      : $LOG_DIR/"
echo "============================================================"
#!/usr/bin/env bash
# =============================================================================
# run_benchmark.sh
# Runs the EGO-Planner simulation N times, each with a different random seed
# loaded from a seeds file, parses per-trial metrics from ROS logs, writes
# every trial to a CSV, and prints the averaged results at the end.
#
# Usage:
#   chmod +x run_benchmark.sh
#   ./run_benchmark.sh [OPTIONS]
#
# Options:
#   -n  NUM_TRIALS   Number of trials to run          (default: 100)
#   -s  SEEDS_FILE   Path to file with one seed/line  (default: seeds.txt)
#   -o  OUTPUT_CSV   Path for per-trial CSV output    (default: results/metrics.csv)
#   -t  TIMEOUT      Max seconds to wait per trial    (default: 120)
#   -p  ROS_PKG      ROS package name                 (default: ego_planner)
#   -l  LAUNCH_FILE  Launch file name inside that pkg  (default: simple_run.launch)
#   -w  SETUP_BASH   Path to ROS workspace setup.bash (default: devel/setup.bash)
#   -k  WAYPOINT_NUM  Number of circuit legs/waypoints to wait for (default: 1)
#
# Seeds file format (one integer per line, lines starting with # are ignored):
#   42
#   137
#   999
#   ...
#
# Dependencies: ROS (roslaunch, rostopic), awk, python3
# =============================================================================

# set -euo pipefail

# # ── Defaults ──────────────────────────────────────────────────────────────────
# NUM_TRIALS=100
# SEEDS_FILE="seeds.txt"
# OUTPUT_CSV="results/metrics.csv"
# TIMEOUT=120
# WAYPOINT_NUM=1      # number of waypoints (legs) per circuit
# ROS_PKG="ego_planner"
# LAUNCH_FILE="simple_run.launch"
# SETUP_BASH="devel/setup.bash"

# # ── Argument parsing ──────────────────────────────────────────────────────────
# while getopts "n:s:o:t:l:w:p:k:" opt; do
#   case $opt in
#     n) NUM_TRIALS="$OPTARG" ;;
#     s) SEEDS_FILE="$OPTARG" ;;
#     o) OUTPUT_CSV="$OPTARG" ;;
#     t) TIMEOUT="$OPTARG" ;;
#     l) LAUNCH_FILE="$OPTARG" ;;
#     p) ROS_PKG="$OPTARG" ;;
#     k) WAYPOINT_NUM="$OPTARG" ;;
#     w) SETUP_BASH="$OPTARG" ;;
#     *) echo "Unknown option: -$OPTARG"; exit 1 ;;
#   esac
# done

# # ── Source ROS workspace ──────────────────────────────────────────────────────
# if [[ ! -f "$SETUP_BASH" ]]; then
#   echo "[ERROR] ROS workspace setup file not found: $SETUP_BASH"
#   echo "        Build your workspace first:  catkin_make  (or catkin build)"
#   echo "        Or pass a custom path with:  -w /path/to/devel/setup.bash"
#   exit 1
# fi

# # shellcheck disable=SC1090
# source "$SETUP_BASH"
# echo "[OK] Sourced ROS workspace: $SETUP_BASH"
# echo "     ROS_PACKAGE_PATH: $ROS_PACKAGE_PATH"

# # ── Validate remaining inputs ─────────────────────────────────────────────────
# if [[ ! -f "$SEEDS_FILE" ]]; then
#   echo "[WARN] Seeds file not found: $SEEDS_FILE — auto-generating $NUM_TRIALS seeds..."
#   python3 - "$SEEDS_FILE" "$NUM_TRIALS" << 'PYEOF'
# import sys, random
# out_path  = sys.argv[1]
# n         = int(sys.argv[2])
# rng       = random.Random()          # non-deterministic master seed
# seeds     = [rng.randint(0, 99999) for _ in range(n)]
# with open(out_path, "w") as fh:
#     fh.write("# Auto-generated seeds file\n")
#     fh.write(f"# {n} random seeds\n\n")
#     for s in seeds:
#         fh.write(f"{s}\n")
# print(f"[OK] Written {n} seeds to '{out_path}'")
# PYEOF
# fi

# PKG_PATH=$(rospack find "$ROS_PKG" 2>/dev/null || true)
# if [[ -z "$PKG_PATH" ]]; then
#   echo "[ERROR] ROS package '$ROS_PKG' not found. Is the workspace built and sourced?"
#   exit 1
# fi
# LAUNCH_PATH="$PKG_PATH/launch/$LAUNCH_FILE"
# if [[ ! -f "$LAUNCH_PATH" ]]; then
#   echo "[ERROR] Launch file not found: $LAUNCH_PATH"
#   echo "        Available launch files in $ROS_PKG:"
#   ls "$PKG_PATH/launch/" 2>/dev/null || echo "        (no launch/ directory found)"
#   exit 1
# fi
# echo "[OK] Launch file found: $LAUNCH_PATH"

# # Read seeds (skip blank lines and comments)
# mapfile -t ALL_SEEDS < <(grep -v '^\s*#' "$SEEDS_FILE" | grep -v '^\s*$')
# TOTAL_SEEDS=${#ALL_SEEDS[@]}

# if [[ $TOTAL_SEEDS -lt $NUM_TRIALS ]]; then
#   echo "[ERROR] Seeds file has only $TOTAL_SEEDS seeds, but NUM_TRIALS=$NUM_TRIALS"
#   echo "        Add more seeds or reduce -n."
#   exit 1
# fi

# # ── Setup output directory ────────────────────────────────────────────────────
# OUTPUT_DIR=$(dirname "$OUTPUT_CSV")
# mkdir -p "$OUTPUT_DIR"
# LOG_DIR="$OUTPUT_DIR/logs"
# mkdir -p "$LOG_DIR"

# # ── CSV header ────────────────────────────────────────────────────────────────
# CSV_HEADER="trial,seed,circuit_success,legs_total,legs_success,legs_fail,total_flying_time_s,total_dist_m,max_vel_ms,avg_vel_ms,total_energy_jerk,total_plan_time_s,avg_plan_time_ms,overall_success_rate_pct,progress_time_errors"
# echo "$CSV_HEADER" > "$OUTPUT_CSV"

# echo "============================================================"
# echo "  EGO-Planner Benchmark"
# echo "  Trials     : $NUM_TRIALS"
# echo "  Seeds file : $SEEDS_FILE"
# echo "  Package    : $ROS_PKG"
# echo "  Launch file: $LAUNCH_FILE"
# echo "  Setup bash : $SETUP_BASH"
# echo "  Timeout    : ${TIMEOUT}s per trial"
# echo "  Output CSV : $OUTPUT_CSV"
# echo "  Log dir    : $LOG_DIR"
# echo "============================================================"

# # ── Helper: kill all ROS nodes cleanly ───────────────────────────────────────
# cleanup_ros() {
#   # Kill roslaunch and all child processes
#   if [[ -n "${LAUNCH_PID:-}" ]]; then
#     kill -TERM "$LAUNCH_PID" 2>/dev/null || true
#     wait "$LAUNCH_PID" 2>/dev/null || true
#     unset LAUNCH_PID
#   fi
#   # Give nodes time to die, then force-kill any stragglers
#   sleep 2
#   pkill -f "ego_planner" 2>/dev/null || true
#   pkill -f "traj_server"  2>/dev/null || true
#   pkill -f "pcl_render"   2>/dev/null || true
#   pkill -f "roslaunch"    2>/dev/null || true
#   sleep 1
# }

# # ── Helper: parse metrics from a single trial log file ───────────────────────
# # Reads ALL "Navigation Metrics" blocks (one per circuit leg) and aggregates.
# # Circuit is SUCCESS only if every leg succeeded.
# # Returns pipe-separated: circuit_success|legs_total|legs_success|legs_fail|
# #   total_flying_time|total_dist|max_vel|avg_vel|total_energy|total_plan_time|avg_plan_ms|progress_errors
# parse_metrics() {
#   local logfile="$1"

#   python3 - "$logfile" <<'PYEOF'
# import sys, re, statistics

# logfile = sys.argv[1]
# try:
#     with open(logfile, "r", errors="replace") as f:
#         content = f.read()
# except FileNotFoundError:
#     print("0|0|0|0|0|0|0|0|0|0|0|0")
#     sys.exit(0)

# # Find ALL Navigation Metrics blocks — one per leg
# blocks = re.findall(
#     r"={5,} Navigation Metrics ={5,}(.*?)={5,}",
#     content, re.DOTALL
# )

# if not blocks:
#     print("0|0|0|0|0|0|0|0|0|0|0|0")
#     sys.exit(0)

# def get(block, pattern, default=0.0):
#     m = re.search(pattern, block)
#     try:    return float(m.group(1).strip()) if m else default
#     except: return default

# legs_total    = len(blocks)
# legs_success  = 0
# legs_fail     = 0
# total_fly     = 0.0
# total_dist    = 0.0
# all_max_vel   = []
# total_energy  = 0.0
# total_plan    = 0.0
# all_plan_ms   = []

# for block in blocks:
#     s_match = re.search(r"Mission #\d+:\s*(SUCCESS|FAIL)", block)
#     leg_ok  = s_match and s_match.group(1) == "SUCCESS"
#     if leg_ok: legs_success += 1
#     else:      legs_fail    += 1

#     total_fly    += get(block, r"Flying Time:\s*([\d.]+)")
#     total_dist   += get(block, r"Total Distance:\s*([\d.]+)")
#     all_max_vel.append(get(block, r"Velocity - Max:\s*([\d.]+)"))
#     total_energy += get(block, r"Energy \(Jerk Integral\):\s*([\d.eE+\-]+)")
#     total_plan   += get(block, r"Planning Time \(this mission\):\s*([\d.]+)")
#     pm = get(block, r"Avg:\s*([\d.]+)\s*ms")
#     if pm > 0: all_plan_ms.append(pm)

# # Circuit success = all legs succeeded
# circuit_success = "1" if legs_fail == 0 else "0"
# max_vel         = max(all_max_vel) if all_max_vel else 0.0
# avg_vel         = total_dist / total_fly if total_fly > 0 else 0.0
# avg_plan_ms     = statistics.mean(all_plan_ms) if all_plan_ms else 0.0

# # Count last_progress_time_ errors across entire log
# progress_errors = len(re.findall(r"last_progress_time_ ERROR", content))

# print(f"{circuit_success}|{legs_total}|{legs_success}|{legs_fail}|"
#       f"{total_fly:.3f}|{total_dist:.3f}|{max_vel:.3f}|{avg_vel:.3f}|"
#       f"{total_energy:.4f}|{total_plan:.4f}|{avg_plan_ms:.3f}|{progress_errors}")
# PYEOF
# }

# # ── Main trial loop ───────────────────────────────────────────────────────────
# trap 'echo "[INTERRUPTED] Cleaning up..."; cleanup_ros; exit 130' INT TERM

# PASS=0
# FAIL=0

# for (( i=1; i<=NUM_TRIALS; i++ )); do
#   SEED="${ALL_SEEDS[$((i-1))]}"
#   LOG_FILE="$LOG_DIR/trial_${i}_seed_${SEED}.log"

#   echo ""
#   echo "────────────────────────────────────────────────────────────"
#   echo "  Trial $i / $NUM_TRIALS  |  seed=$SEED"
#   echo "────────────────────────────────────────────────────────────"

#   # Launch ROS simulation in background, redirect all output to log
#   roslaunch "$ROS_PKG" "$LAUNCH_FILE" map_seed:="$SEED" \
#     > "$LOG_FILE" 2>&1 &
#   LAUNCH_PID=$!

#   # Wait for mission complete or timeout
#   ELAPSED=0
#   DONE=0
#   while (( ELAPSED < TIMEOUT )); do
#     sleep 1
#     (( ELAPSED++ )) || true

#     if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
#       DONE=1; break
#     fi

#     BLOCKS_FOUND=$(grep -c "Navigation Metrics" "$LOG_FILE" 2>/dev/null || echo 0)
#     if (( BLOCKS_FOUND >= WAYPOINT_NUM )); then
#       sleep 2
#       DONE=1; break
#     fi
#   done

#   if (( DONE == 0 )); then
#     echo "  [TIMEOUT] Trial $i exceeded ${TIMEOUT}s — marking as FAIL"
#     echo "TIMEOUT" >> "$LOG_FILE"
#   fi

#   cleanup_ros

#   # Parse metrics from log
#   RAW=$(parse_metrics "$LOG_FILE")
#   IFS='|' read -r circuit_success legs_total legs_success legs_fail total_fly total_dist max_vel avg_vel total_energy total_plan avg_plan_ms progress_errors <<< "$RAW"

#   if [[ $circuit_success == "1" ]]; then
#     (( PASS++ )) || true
#     STATUS="SUCCESS"
#   else
#     (( FAIL++ )) || true
#     STATUS="FAIL"
#   fi

#   # Overall success rate so far
#   SR=$(awk "BEGIN {printf \"%.1f\", $PASS / $i * 100}")

#   # Append to CSV
#   echo "$i,$SEED,$circuit_success,$legs_total,$legs_success,$legs_fail,$total_fly,$total_dist,$max_vel,$avg_vel,$total_energy,$total_plan,$avg_plan_ms,$SR,$progress_errors" \
#     >> "$OUTPUT_CSV"

#   echo "  Result : $STATUS  ($legs_success/$legs_total legs succeeded)"
#   echo "  Flying : ${total_fly}s  |  Dist: ${total_dist}m  |  MaxVel: ${max_vel} m/s  |  AvgVel: ${avg_vel} m/s"
#   echo "  Energy (jerk): $total_energy  |  PlanTime: ${total_plan}s (avg ${avg_plan_ms}ms)"
#   if [[ "$progress_errors" -gt 0 ]]; then
#     echo "  [WARN] last_progress_time_ errors: $progress_errors"
#   fi
#   echo "  Running success rate: $PASS/$i ($SR%)"
# done

# # ── Compute and print averages ─────────────────────────────────────────────────
# echo ""
# echo "============================================================"
# echo "  BENCHMARK COMPLETE — AVERAGED RESULTS"
# echo "  ($NUM_TRIALS trials, seeds from $SEEDS_FILE)"
# echo "============================================================"

# python3 - "$OUTPUT_CSV" "$NUM_TRIALS" "$PASS" "$FAIL" <<'PYEOF'
# import sys, csv, statistics

# csv_file   = sys.argv[1]
# n_trials   = int(sys.argv[2])
# n_pass     = int(sys.argv[3])
# n_fail     = int(sys.argv[4])

# rows = []
# with open(csv_file) as f:
#     reader = csv.DictReader(f)
#     for row in reader:
#         rows.append(row)

# def col_floats(key):
#     vals = []
#     for r in rows:
#         try:
#             vals.append(float(r[key]))
#         except (ValueError, KeyError):
#             pass
#     return vals

# def mean(lst):
#     return statistics.mean(lst) if lst else 0.0

# def stdev(lst):
#     return statistics.stdev(lst) if len(lst) > 1 else 0.0

# # Successful circuits only for time/distance/energy metrics
# success_rows = [r for r in rows if r.get("circuit_success") == "1"]

# def scol(key):
#     vals = []
#     for r in success_rows:
#         try:
#             vals.append(float(r[key]))
#         except (ValueError, KeyError):
#             pass
#     return vals

# # Per-leg stats across all trials
# legs_total   = col_floats("legs_total")
# legs_success = col_floats("legs_success")
# legs_fail    = col_floats("legs_fail")

# # Circuit-level metrics (summed per trial, averaged across trials)
# fly_times    = scol("total_flying_time_s")
# dists        = scol("total_dist_m")
# max_vels     = scol("max_vel_ms")
# avg_vels     = scol("avg_vel_ms")
# energies     = scol("total_energy_jerk")
# plan_times   = col_floats("total_plan_time_s")
# avg_plan_ms  = col_floats("avg_plan_time_ms")

# success_rate = n_pass / n_trials * 100.0
# total_legs   = int(sum(legs_total))
# total_leg_ok = int(sum(legs_success))
# total_leg_fail = int(sum(legs_fail))
# leg_success_rate = total_leg_ok / total_legs * 100.0 if total_legs > 0 else 0.0

# print(f"\n  ── Circuit Results ({n_trials} trials) ──────────────────────────")
# print(f"  Full circuit success   : {n_pass} / {n_trials}  ({success_rate:.1f}%)")
# print(f"  Full circuit fail      : {n_fail} / {n_trials}  ({100-success_rate:.1f}%)")
# print()
# print(f"  ── Per-leg Results ({total_legs} legs total) ─────────────────────")
# print(f"  Legs succeeded         : {total_leg_ok} / {total_legs}  ({leg_success_rate:.1f}%)")
# print(f"  Legs failed            : {total_leg_fail} / {total_legs}  ({100-leg_success_rate:.1f}%)")
# print()
# print(f"  ── Metrics averaged over {len(success_rows)} successful circuits ──")
# print(f"  Total Flying Time      : {mean(fly_times):.3f} s  ± {stdev(fly_times):.3f}")
# print(f"  Total Distance         : {mean(dists):.3f} m  ± {stdev(dists):.3f}")
# print(f"  Max Velocity           : {mean(max_vels):.3f} m/s ± {stdev(max_vels):.3f}")
# print(f"  Avg Velocity           : {mean(avg_vels):.3f} m/s ± {stdev(avg_vels):.3f}")
# print(f"  Total Energy (Jerk)    : {mean(energies):.4f}  ± {stdev(energies):.4f}")
# print()
# print(f"  ── Planning time (all {n_trials} trials) ──────────────────────────")
# print(f"  Total Plan Time        : {mean(plan_times):.4f} s  ± {stdev(plan_times):.4f}")
# print(f"  Avg Plan Time per call : {mean(avg_plan_ms):.3f} ms ± {stdev(avg_plan_ms):.3f}")
# print()

# prog_errors = col_floats("progress_time_errors")
# trials_with_errors = sum(1 for v in prog_errors if v > 0)
# total_errors = sum(prog_errors)
# print(f"  ── last_progress_time_ errors ──────────────────────────────")
# print(f"  Trials with errors     : {trials_with_errors} / {n_trials}")
# print(f"  Total error count      : {int(total_errors)}")
# if trials_with_errors > 0:
#     print(f"  [WARN] These trials had stale local targets — treat their metrics with caution")
# print()
# PYEOF

# echo "  Per-trial CSV : $OUTPUT_CSV"
# echo "  Raw logs      : $LOG_DIR/"
# echo "============================================================"
