# Prediction-Assisted Planning Benchmark

This protocol evaluates whether feeding LBSCNet predictions into the planner improves autonomous navigation. Success Rate is reported only as a secondary safety outcome.

## Paired comparisons

Run identical held-out map seeds for: (1) no prediction, (2) LBSCNet prediction through the production ROS path, and optionally (3) an oracle ground-truth prediction as an upper bound. Keep start/goal poses, simulator physics, sensor topics, robot limits, timeout, ROS/simulator versions, GPU, git revision, and checkpoint fixed. Do not tune on evaluation seeds. Use at least 100 held-out seeds per scenario cell; 20 is a smoke test only.

Use a balanced matrix of low/medium/high clutter, nominal sensing plus 20% range dropout and 100 ms cloud delay, short/nominal/long routes, and corridor/open-room maps. Verify that `map_seed` reaches the map generator before collection.

## Measurements

Log synchronized odometry and planner/prediction timestamps for every trial: time to goal (and partial time at timeout), path length, integrated squared jerk, peak speed/acceleration, minimum and 5th-percentile body clearance to ground-truth obstacles, collision count/first contact, replans, emergency stops, planner failures, timeout reason, prediction latency/age/dropped frames, planner compute time, and command-loop deadline misses. The existing `Navigation Metrics` block provides time, distance, velocity, jerk, and planning latency.

Score collisions against simulator ground truth, not the predicted occupancy buffer used by planning. This prevents conservative hallucinations from being rewarded and false negatives from being hidden.

## Primary metric: Safety-Adjusted Execution Cost

For each paired seed, compute:

```
SAEC = 0.35*T + 0.25*L + 0.20*J + 0.20*P + failure_penalty
```

Normalize `T` (time), `L` (path length), `J` (jerk integral), and `P` (mean planning latency) by the no-prediction value for the same seed. Cap each normalized component at 5.0 to prevent near-zero numerical denominators (especially jerk) from dominating; report the cap in the configuration. Set `failure_penalty = 1.0` for a failed/timed-out run and add another `1.0` for each collision. Lower is better. Report paired mean and median deltas (prediction minus baseline), 95% paired bootstrap CIs over seeds (10,000 resamples), and the fraction of seeds improved. This keeps efficiency and smoothness informative even when both methods have identical Success Rate.

## Secondary metrics and decision rule

Report collision rate, clearance distribution, Success Rate, timeout/emergency-stop rate, conditional time/path ratios on completed trials, jerk, peak dynamics, replan count, deadline misses, prediction age/latency, and end-to-end planning latency. Stratify by every scenario factor and delay level. Use Holm correction for multiple scenario-cell claims or label intervals descriptive.

Claim an improvement only when the SAEC 95% CI is below zero, collision rate does not significantly increase, and minimum-clearance safety does not regress. If safety improves but SAEC is neutral, report a safety benefit without claiming overall planning improvement. Publish per-seed CSVs and exact commands/configuration.
