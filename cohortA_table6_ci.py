"""Recompute Table 6 (strategy ROIs) on the NEW Cohort A 608-event dataset
(cohortA_dataset.csv, 12 models), with:

  (1) MARGINAL event-clustered 90% CIs per ROI cell (point +/- 1.645*SE), and
  (2) PAIRED event-clustered CIs on the ROI DIFFERENCE (Proper - each baseline).

The paired test is the point of this script. ROI is a ratio of sums
(100*sum_profit/sum_cost), so ROI_proper - ROI_baseline is a difference of two
ratios with *different* denominators. Comparing the two marginal CIs is the
wrong test: it ignores that both strategies bet on the same events and their
per-event P&Ls are correlated. Instead we resample events ONCE per bootstrap
iteration and apply the SAME resample to both strategies, so the difference
distribution retains that covariance. Dominance is then read off the paired CI.

Resampling unit = EVENT (all of a question's markets move together).
Kelly (full leverage) is degenerate (sequential bankroll -> bankruptcy for
every model); it is not additive across events, so it gets a point estimate
only and is excluded from the paired test, matching the paper.

Convention matches the rest of the repo: B=6000, seed=12345, 90% CI = +/-1.645*SE.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "paper_table_scripts"))

from scripts.scoring_rule_roi_analysis import (  # noqa: E402
    is_within_spread, run_pqo_brier, run_inv_edge, _run_kelly_internal,
)
from scripts.run_kelly_norm import run_flat_alloc, run_kelly_norm  # noqa: E402
from scripts.canonical_strats_200 import run_max_margin, per_forecast_brier  # noqa: E402

B, SEED, Z90 = 6000, 12345, 1.645
DATA = os.path.join(os.path.dirname(__file__), "cohortA_dataset.csv")

LABELS = {
    "agent-anthropic/claude-opus-4.6": "Claude Opus 4.6",
    "agent-gemini-3": "Gemini 3",
    "gpt-5.2-none": "GPT-5.2 (Base)",
    "gpt-5.2-high": "GPT-5.2 (High)",
    "anthropic/claude-opus-4.5": "Claude Opus 4.5",
    "anthropic/claude-sonnet-4.5": "Claude Sonnet 4.5",
    "google/gemini-3-pro-preview": "Gemini 3 Pro",
    "deepseek/deepseek-v3.2": "DeepSeek V3.2",
    "moonshotai/kimi-k2-thinking": "Kimi K2 Thinking",
    "minimax/minimax-m2": "Minimax M2",
    "qwen/qwen3-235b-a22b-2507": "Qwen 3 235B",
    "meta-llama/llama-4-maverick": "LLaMA 4 Maverick",
}

# Proper + the five baseline columns of Table 6. Kelly handled separately.
PROPER = ("Proper", run_pqo_brier)
BASELINES = [
    ("Max (Mkt)", run_flat_alloc),
    ("Max (Grp)", run_max_margin),
    ("Inv-Margin", run_inv_edge),
    ("Kelly-Alike", run_kelly_norm),
]


def build_events(sub):
    """Per-event outcome lists (+ event time for Kelly's sequential sim)."""
    events = []
    for ev, g in sub.groupby("event_ticker", sort=True):
        outs, t = [], None
        for _, r in g.iterrows():
            p, qy, qn, y = r["model_prob_yes"], r["yes_ask_prob"], r["no_ask_prob"], r["outcome"]
            if pd.isna(p) or pd.isna(qy) or pd.isna(qn) or pd.isna(y):
                continue
            if qy <= 0 or qn <= 0:
                continue
            outs.append({"p": float(p), "q_yes": float(qy), "q_no": float(qn), "actual": int(y)})
            if t is None:
                t = r.get("prediction_created_at")
        if outs:
            events.append({"event_ticker": ev, "outcomes": outs, "time": t})
    return events


def bet_pc(b):
    cost = b["weight"] * b["price"]
    win = (b["actual"] == 1) if b["side"] == "YES" else (b["actual"] == 0)
    return ((b["weight"] - cost) if win else -cost), cost


def per_event_pc(events, runner):
    """Per-event (profit, cost) for an *additive* strategy. Running the runner
    on one event at a time is exact for every strategy except Kelly."""
    n = len(events)
    ep, ec = np.zeros(n), np.zeros(n)
    for i, ev in enumerate(events):
        for b in runner([ev]):
            dp, dc = bet_pc(b)
            ep[i] += dp
            ec[i] += dc
    return ep, ec


def per_event_dS(sub, event_order):
    """Per-event (sum of kb-mb, forecast count), aligned to `event_order` (the
    same event index space as the strategy P&L arrays) so one shared bootstrap
    resample applies to dS and every strategy. ROI-independent Table-6 dS."""
    pos = {e: i for i, e in enumerate(event_order)}
    n = len(event_order)
    num, cnt = np.zeros(n), np.zeros(n)
    fb = per_forecast_brier(sub)
    if len(fb) == 0:
        return num, cnt
    v = (fb["kb"] - fb["mb"]).to_numpy(float)
    idx = fb["event_ticker"].map(pos).to_numpy()
    for i, val in zip(idx, v):
        if i == i:  # skip forecasts on events not in event_order (NaN index)
            num[int(i)] += val
            cnt[int(i)] += 1.0
    return num, cnt


def roi(num, cost):
    s = cost.sum()
    return 100.0 * num.sum() / s if s > 0 else np.nan


def boot_ratio(num, den, counts):
    """Bootstrap distribution of 100*sum(num)/sum(den) under event resamples
    given as multinomial `counts` (shape (B, n))."""
    d = counts @ den
    with np.errstate(divide="ignore", invalid="ignore"):
        est = 100.0 * (counts @ num) / d
    return est


def main():
    df = pd.read_csv(DATA, low_memory=False)
    df["outcome"] = pd.to_numeric(df["outcome"], errors="coerce")
    df = df.dropna(subset=["model_prob_yes", "yes_ask_prob", "no_ask_prob", "outcome"])
    df = df[(df["yes_ask_prob"] > 0) & (df["no_ask_prob"] > 0)]

    roster = [m for m in LABELS if m in set(df["predictor_name"])]

    marg_rows, pair_rows = [], []
    for m in roster:
        sub = df[df["predictor_name"] == m]
        events = build_events(sub)
        n = len(events)
        counts = np.random.default_rng(SEED).multinomial(n, np.full(n, 1.0 / n), size=B).astype(float)

        # ---- dS (Table-6 relative-Brier gap, mean(kb-mb)) ----
        ds_num, ds_cnt = per_event_dS(sub, [e["event_ticker"] for e in events])
        ds_pt = ds_num.sum() / ds_cnt.sum() if ds_cnt.sum() > 0 else np.nan
        ds_se = np.nanstd(boot_ratio(ds_num, ds_cnt, counts) / 100.0)  # undo the *100 in boot_ratio

        # ---- per-event P&L for Proper + additive baselines ----
        pc = {}
        for name, runner in [PROPER] + BASELINES:
            pc[name] = per_event_pc(events, runner)  # (profit[], cost[])

        # marginal ROI + CI for each additive strategy
        marg = {"dS": ds_pt, "dS_se": ds_se}
        boot = {}
        for name, _ in [PROPER] + BASELINES:
            ep, ec = pc[name]
            pt = roi(ep, ec)
            bd = boot_ratio(ep, ec, counts)
            se = float(np.nanstd(bd))
            marg[name] = pt
            marg[name + "_se"] = se
            boot[name] = bd

        # Kelly (full leverage): sequential, degenerate -> point only
        kelly_roi = _run_kelly_internal(events, fraction=1.0, allow_leverage=True)["roi"]
        marg["Kelly"] = kelly_roi
        marg["Kelly_se"] = np.nan

        marg_rows.append({"model": LABELS[m], **marg})

        # ---- PAIRED diff: Proper - each additive baseline (same resample) ----
        pb = boot[PROPER[0]]
        for name, _ in BASELINES:
            diff_boot = pb - boot[name]
            diff_boot = diff_boot[np.isfinite(diff_boot)]
            diff_pt = marg[PROPER[0]] - marg[name]
            se = float(np.std(diff_boot, ddof=1))
            lo90n, hi90n = diff_pt - Z90 * se, diff_pt + Z90 * se       # normal-approx 90% CI
            lo90p, hi90p = np.percentile(diff_boot, [5, 95])            # percentile 90% CI
            # two-sided bootstrap p-value for H0: diff == 0
            p_le = float(np.mean(diff_boot <= 0))
            p_ge = float(np.mean(diff_boot >= 0))
            pval = min(1.0, 2 * min(p_le, p_ge))
            pair_rows.append({
                "model": LABELS[m], "baseline": name, "diff_ROI": diff_pt, "se": se,
                "lo90_normal": lo90n, "hi90_normal": hi90n,
                "lo90_pct": lo90p, "hi90_pct": hi90p, "pval2": pval,
            })

    marg_df = pd.DataFrame(marg_rows).sort_values("dS", ascending=False).reset_index(drop=True)
    pair_df = pd.DataFrame(pair_rows)
    order = {lbl: i for i, lbl in enumerate(marg_df["model"])}
    pair_df = pair_df.sort_values(by=["model", "baseline"],
                                  key=lambda s: s.map(order) if s.name == "model" else s).reset_index(drop=True)

    # ---------------- print: Table 6 (marginal) ----------------
    cols = ["Proper", "Max (Mkt)", "Max (Grp)", "Inv-Margin", "Kelly-Alike", "Kelly"]
    print(f"\n===== Table 6 (recomputed on {df['event_ticker'].nunique()} events, "
          f"{len(roster)} models) -- ROI% point +/- 90% halfwidth =====\n")
    print(f"{'Model':<18} {'dS':>9} | " + " ".join(f"{c:>14}" for c in cols))
    for _, r in marg_df.iterrows():
        cells = []
        for c in cols:
            if c == "Kelly":
                cells.append(f"{r[c]:>+8.1f}       ")
            else:
                cells.append(f"{r[c]:>+7.1f}±{Z90*r[c+'_se']:>5.1f} ")
        print(f"{r['model']:<18} {r['dS']:>+9.4f} | " + " ".join(cells))

    # ---------------- print: paired ROI-difference CIs ----------------
    print(f"\n\n===== Paired event-clustered CIs: ROI(Proper) - ROI(baseline) =====")
    print("(same resampled events for both strategies; 90% CIs; dominance = CI excludes 0)\n")
    print(f"{'Model':<18} {'Baseline':<12} {'Delta ROI':>10} "
          f"{'90% CI (percentile)':>24} {'p':>7}  verdict")
    for _, r in pair_df.iterrows():
        excl = (r["lo90_pct"] > 0) or (r["hi90_pct"] < 0)
        verdict = ("Proper >" if r["lo90_pct"] > 0 else
                   "baseline >" if r["hi90_pct"] < 0 else "n.s.")
        ci = f"[{r['lo90_pct']:>+7.1f}, {r['hi90_pct']:>+7.1f}]"
        star = "*" if excl else " "
        print(f"{r['model']:<18} {r['baseline']:<12} {r['diff_ROI']:>+10.1f} "
              f"{ci:>24} {r['pval2']:>7.3f}  {star}{verdict}")

    # ---------------- dominance summary calibrated to paired CIs ----------------
    print("\n\n===== Dominance summary (calibrated to paired CIs) =====")
    for name, _ in BASELINES:
        sub = pair_df[pair_df["baseline"] == name]
        win = int(((sub["lo90_pct"] > 0)).sum())
        lose = int(((sub["hi90_pct"] < 0)).sum())
        ns = len(sub) - win - lose
        print(f"  Proper vs {name:<12}: Proper sig. higher on {win}/{len(sub)} models, "
              f"baseline sig. higher on {lose}, not significant on {ns}")

    # ---------------- write CSVs ----------------
    outdir = os.path.dirname(__file__)
    m_out = marg_df.copy()
    for c in cols:
        if c == "Kelly":
            continue
        m_out[c + "_ci90"] = (Z90 * m_out[c + "_se"]).round(2)
    m_out["dS_ci90"] = (Z90 * m_out["dS_se"]).round(4)
    keep = ["model", "dS", "dS_ci90"]
    for c in cols:
        keep.append(c)
        if c != "Kelly":
            keep.append(c + "_ci90")
    m_out[keep].round(4).to_csv(os.path.join(outdir, "cohortA_table6_ci.csv"), index=False)
    pair_df.round(4).to_csv(os.path.join(outdir, "cohortA_table6_paired_ci.csv"), index=False)

    write_latex(marg_df, pair_df, cols, outdir)
    print("\nWrote cohortA_table6_ci.csv, cohortA_table6_paired_ci.csv, "
          "cohortA_table6_ci.tex, cohortA_table6_paired_ci.tex")


def _m(x):
    return f"{x:.1f}".replace("-", "$-$")


def write_latex(marg_df, pair_df, cols, outdir):
    # ---- Table 6 with marginal event-clustered 90% CIs (bold best ROI per row) ----
    L = [r"\begin{tabular}{l r *{5}{r} r}", r"\toprule",
         r"& & \multicolumn{6}{c}{\textbf{ROI (\%)} $\pm$ 90\% CI (event-clustered)} \\",
         r"\cmidrule(lr){3-8}",
         r"\textbf{Model} & $\Delta S$ & \textbf{Proper} & Max (Mkt) & Max (Grp) "
         r"& Inv-Margin & Kelly-Alike & Kelly \\", r"\midrule"]
    add = [c for c in cols if c != "Kelly"]
    for _, r in marg_df.iterrows():
        vals = {c: r[c] for c in cols}
        best = max(vals, key=vals.get)
        cells = []
        for c in add:
            s = f"{r[c]:+.1f}\\pm{Z90*r[c+'_se']:.1f}".replace("-", "$-$")
            cells.append(f"\\textbf{{${s}$}}" if c == best else f"${s}$")
        kelly = f"{r['Kelly']:+.1f}".replace("-", "$-$")
        cells.append(f"\\textbf{{${kelly}$}}" if best == "Kelly" else f"${kelly}$")
        ds = f"{r['dS']:+.4f}\\pm{Z90*r['dS_se']:.4f}".replace("-", "$-$")
        L.append(f"{r['model']:<18} & ${ds}$ & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(outdir, "cohortA_table6_ci.tex"), "w") as f:
        f.write("\n".join(L) + "\n")

    # ---- Paired ROI-difference CIs, calibrated (ΔS>0) models -> main text ----
    calib = marg_df[marg_df["dS"] > 0]["model"].tolist()
    P = [r"\begin{tabular}{l l r c c}", r"\toprule",
         r"\textbf{Model} & \textbf{Baseline} & $\Delta$ROI & \textbf{90\% CI (paired)} & \textbf{Proper $>$?} \\",
         r"\midrule"]
    base_order = ["Max (Mkt)", "Max (Grp)", "Inv-Margin", "Kelly-Alike"]
    for mdl in calib:
        for j, b in enumerate(base_order):
            r = pair_df[(pair_df["model"] == mdl) & (pair_df["baseline"] == b)].iloc[0]
            ci = f"[{r['lo90_pct']:+.1f},\\,{r['hi90_pct']:+.1f}]".replace("-", "$-$")
            sig = r"\checkmark" if r["lo90_pct"] > 0 else (r"$\times$ (worse)" if r["hi90_pct"] < 0 else r"$\times$ (n.s.)")
            name = mdl if j == 0 else ""
            P.append(f"{name:<16} & {b:<12} & ${r['diff_ROI']:+.1f}$".replace("-", "$-$")
                     + f" & ${ci}$ & {sig} \\\\")
        P.append(r"\midrule")
    P[-1] = r"\bottomrule"
    P.append(r"\end{tabular}")
    with open(os.path.join(outdir, "cohortA_table6_paired_ci.tex"), "w") as f:
        f.write("\n".join(P) + "\n")


if __name__ == "__main__":
    main()
