#!/usr/bin/env python3
"""Paired baseline (no-prediction) vs LBSCNet-prediction navigation benchmark.

Metrics follow standard navigation-evaluation practice and are reported
individually with paired statistics (bootstrap 95% CI + Wilcoxon signed-rank),
because a single composite score hides the safety/efficiency trade-off that is
the whole point of predictive planning.

  PRIMARY — safety:
    - Success rate            (reached goal, no collision)      higher better
    - Ground-truth collision  (body within drone_radius of GT)  lower  better
    - Min clearance [m]       (to nearest GT obstacle)          higher better
    - Time-in-danger [s]      (clearance < safety threshold)    lower  better

  SECONDARY — efficiency:
    - SPL                     (success weighted by path length) higher better
    - Path efficiency         (actual / straight-line)          lower  better
    - Flight time, distance, jerk, planning latency             lower  better

  OPTIONAL — SAEC: a custom weighted composite, reported LAST only. Its weights
  are arbitrary; it is NOT the headline result.

Inputs are the per-seed CSVs emitted by run_sim.sh (updated header with
outcome / clearance / path columns). Old CSVs missing those columns are flagged.
"""
import argparse
import csv
import math
import random
import statistics


def read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {r["seed"]: r for r in rows}, (rows[0].keys() if rows else [])


def num(row, key):
    """Parsed float, or None when the column is absent / blank / non-finite."""
    v = row.get(key)
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# --------------------------------------------------------------------------- #
# statistics helpers
# --------------------------------------------------------------------------- #
def bootstrap(values, rng, samples):
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    means = [statistics.mean(values[rng.randrange(n)] for _ in range(n))
             for _ in range(samples)]
    means.sort()
    return means[int(.025 * samples)], means[int(.975 * samples)]


def _avg_ranks(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0          # 1-based average rank for the tie group
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def wilcoxon_p(deltas):
    """Two-sided Wilcoxon signed-rank p-value, normal approx w/ continuity
    correction (pure-python, no scipy). Approximate for small n (<~10)."""
    diffs = [d for d in deltas if d != 0.0]
    n = len(diffs)
    if n == 0:
        return 1.0
    ranks = _avg_ranks([abs(d) for d in diffs])
    w_plus = sum(r for d, r in zip(diffs, ranks) if d > 0)
    w_minus = sum(r for d, r in zip(diffs, ranks) if d < 0)
    t = min(w_plus, w_minus)
    mean_t = n * (n + 1) / 4.0
    sd_t = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sd_t == 0:
        return 1.0
    z = (t - mean_t + 0.5) / sd_t
    return min(1.0, 2.0 * _norm_cdf(z))


def paired_values(baseline, prediction, seeds, key):
    """(base_vals, pred_vals) over seeds where BOTH have a finite value."""
    b, p = [], []
    for s in seeds:
        bv, pv = num(baseline[s], key), num(prediction[s], key)
        if bv is not None and pv is not None:
            b.append(bv)
            p.append(pv)
    return b, p


def report_paired(label, baseline, prediction, seeds, key, rng, samples,
                  higher_is_better, unit=""):
    b, p = paired_values(baseline, prediction, seeds, key)
    if not b:
        print(f"  {label:24s}: (column '{key}' missing — re-run run_sim.sh)")
        return
    deltas = [pv - bv for bv, pv in zip(b, p)]
    lo, hi = bootstrap(deltas, rng, samples)
    pval = wilcoxon_p(deltas)
    better = sum((d > 0) if higher_is_better else (d < 0) for d in deltas)
    arrow = "↑ better" if higher_is_better else "↓ better"
    print(f"  {label:24s}: base={statistics.mean(b):.4g}{unit}  "
          f"pred={statistics.mean(p):.4g}{unit}  "
          f"Δ={statistics.mean(deltas):+.4g}  95%CI=[{lo:+.4g},{hi:+.4g}]  "
          f"p={pval:.3f}  improved={better}/{len(deltas)}  ({arrow})")


def rate(rows, seeds, key, positive_is_one=True):
    vals = [num(rows[s], key) for s in seeds]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    hits = sum(1 for v in vals if (v >= 0.5) == positive_is_one)
    return 100.0 * hits / len(vals)


def spl(rows, seeds):
    """Success weighted by Path Length: mean_i S_i * l_i / max(p_i, l_i),
    l_i = straight-line reference, p_i = actual distance."""
    out = []
    for s in seeds:
        succ = num(rows[s], "success")
        l = num(rows[s], "straight_line_m")
        p = num(rows[s], "total_dist_m")
        if succ is None or l is None or p is None or l <= 0 or p <= 0:
            continue
        out.append((1.0 if succ >= 0.5 else 0.0) * (l / max(p, l)))
    return out


# --------------------------------------------------------------------------- #
# SAEC (optional composite — reported last)
# --------------------------------------------------------------------------- #
def _n(row, key, default=0.0):
    v = num(row, key)
    return v if v is not None else default


def saec(row, base, failure_penalty=1.0):
    ratios = [
        min(_n(row, "flying_time_s") / max(_n(base, "flying_time_s"), 1e-9), 5.0),
        min(_n(row, "total_dist_m") / max(_n(base, "total_dist_m"), 1e-9), 5.0),
        min(_n(row, "energy_jerk") / max(_n(base, "energy_jerk"), 1e-9), 5.0),
        min(_n(row, "avg_plan_time_ms") / max(_n(base, "avg_plan_time_ms"), 1e-9), 5.0),
    ]
    cost = sum(w * x for w, x in zip((.35, .25, .20, .20), ratios))
    if _n(row, "success") < 0.5:
        cost += failure_penalty
    # prefer ground-truth collision; fall back to legacy collision_count
    coll = num(row, "gt_collision")
    if coll is None:
        coll = _n(row, "collision_count")
    if coll > 0:
        cost += failure_penalty
    return cost


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", help="no-prediction metrics.csv")
    ap.add_argument("prediction", help="LBSCNet metrics.csv")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260909)
    args = ap.parse_args()

    baseline, base_cols = read_csv(args.baseline)
    prediction, pred_cols = read_csv(args.prediction)
    seeds = sorted(set(baseline) & set(prediction))
    if not seeds:
        raise SystemExit("No paired seeds found")

    for needed in ("gt_collision", "min_clearance_m", "time_in_danger_s",
                   "straight_line_m", "path_efficiency"):
        if needed not in base_cols or needed not in pred_cols:
            print(f"[WARN] column '{needed}' absent in one or both CSVs — "
                  f"rebuild ego_planner + re-run run_sim.sh to populate it.")

    rng = random.Random(args.seed)
    N = args.bootstrap

    print("=" * 70)
    print(f"Paired prediction benchmark   (trials={len(seeds)})")
    print("=" * 70)

    print("\nPRIMARY — Safety")
    bsr, psr = rate(baseline, seeds, "success"), rate(prediction, seeds, "success")
    print(f"  {'Success rate':24s}: base={bsr:.1f}%  pred={psr:.1f}%  (↑ better)")
    bcr = rate(baseline, seeds, "gt_collision")
    pcr = rate(prediction, seeds, "gt_collision")
    if bcr is not None and pcr is not None:
        print(f"  {'GT collision rate':24s}: base={bcr:.1f}%  pred={pcr:.1f}%  (↓ better)")
    else:
        print(f"  {'GT collision rate':24s}: (column 'gt_collision' missing)")
    report_paired("Min clearance", baseline, prediction, seeds, "min_clearance_m",
                  rng, N, higher_is_better=True, unit="m")
    report_paired("Time-in-danger", baseline, prediction, seeds, "time_in_danger_s",
                  rng, N, higher_is_better=False, unit="s")

    print("\nSECONDARY — Efficiency")
    b_spl, p_spl = spl(baseline, seeds), spl(prediction, seeds)
    if b_spl and p_spl:
        print(f"  {'SPL':24s}: base={statistics.mean(b_spl):.4f}  "
              f"pred={statistics.mean(p_spl):.4f}  (↑ better)")
    report_paired("Path efficiency", baseline, prediction, seeds, "path_efficiency",
                  rng, N, higher_is_better=False)
    report_paired("Flight time", baseline, prediction, seeds, "flying_time_s",
                  rng, N, higher_is_better=False, unit="s")
    report_paired("Distance", baseline, prediction, seeds, "total_dist_m",
                  rng, N, higher_is_better=False, unit="m")
    report_paired("Jerk (energy)", baseline, prediction, seeds, "energy_jerk",
                  rng, N, higher_is_better=False)
    report_paired("Planning latency", baseline, prediction, seeds, "avg_plan_time_ms",
                  rng, N, higher_is_better=False, unit="ms")

    print("\nOPTIONAL — SAEC composite (arbitrary weights; not the headline)")
    deltas = [saec(prediction[s], baseline[s]) - saec(baseline[s], baseline[s])
              for s in seeds]
    lo, hi = bootstrap(deltas, rng, N)
    improved = sum(d < 0 for d in deltas)
    print(f"  SAEC Δ (pred-base): mean={statistics.mean(deltas):+.4f}  "
          f"median={statistics.median(deltas):+.4f}  95%CI=[{lo:+.4f},{hi:+.4f}]  "
          f"p={wilcoxon_p(deltas):.3f}  improved={improved}/{len(deltas)}  (↓ better)")
    print("=" * 70)


if __name__ == "__main__":
    main()
