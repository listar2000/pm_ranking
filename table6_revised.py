"""Table 6 (revised) for both cohorts, on the SAME event sets as the *_dataset.csv
files, with the roster cut to the models actually present in each set.

Two changes prompted by the compounding critique:

  (1) The main table reports every strategy on the SAME per-bet basis
      ROI = 100*sum(profit)/sum(cost), where each bet gets a fresh unit budget
      (no compounding, order-independent). That includes Kelly-Alike (the Kelly
      fraction f*=edge/(1-price) used as a static per-bet weight). This is the
      only like-for-like comparison.

  (2) The paper's "Kelly" column was a *compounded, leveraged* sequential
      bankroll -> it measures terminal wealth growth, a different quantity on a
      different scale (ruin for miscalibrated models, explosive for lucky ones).
      We surface full- AND half-Kelly, each capped (standard, bet<=bankroll) and
      leveraged (paper), in a separate diagnostic and do NOT mix them into the
      per-bet ROI columns.

Note on half-Kelly: on the per-bet basis, scaling every bet weight by 1/2 scales
profit and cost equally, so ROI is unchanged -> per-bet half-Kelly == Kelly-Alike
exactly. Half-Kelly only differs from full-Kelly once the bankroll compounds,
which is exactly the diagnostic panel.

Event-clustered bootstrap, B=6000, seed=12345, 90% CI = point +/- 1.645*SE.
CIs are reported only for the additive (per-bet) strategies; the compounded
Kelly variants are path-dependent and get point estimates only.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(HERE, "..")), "paper_table_scripts"))

from cohortA_table6_ci import (  # reuse the vetted per-event helpers
    build_events, per_event_pc, per_event_dS, roi, boot_ratio, PROPER, BASELINES,
)
from scripts.scoring_rule_roi_analysis import _run_kelly_internal  # noqa: E402

B, SEED, Z90 = 6000, 12345, 1.645

# --- Cohort A: image roster INTERSECTED with the 608-event dataset ---------
# (Grok 4.1 Fast covers 516/608 and DeepSeek R1 192/608 -> not on this event
#  set, excluded; Gemini 3 Pro / Claude Opus 4.5 are not in the image, dropped.)
COHORT_A = dict(
    data="cohortA_dataset.csv",
    roster=[
        ("agent-anthropic/claude-opus-4.6", "Claude Opus 4.6"),
        ("agent-gemini-3", "Gemini 3"),
        ("gpt-5.2-none", "GPT-5.2 (Base)"),
        ("anthropic/claude-sonnet-4.5", "Claude Sonnet 4.5"),
        ("meta-llama/llama-4-maverick", "LLaMA 4 Maverick"),
        ("gpt-5.2-high", "GPT-5.2 (High)"),
        ("x-ai/grok-4.1-fast", "Grok 4.1 Fast"),
        ("deepseek/deepseek-v3.2", "DeepSeek V3.2"),
        ("moonshotai/kimi-k2-thinking", "Kimi K2 Thinking"),
        ("minimax/minimax-m2", "Minimax M2"),
        ("qwen/qwen3-235b-a22b-2507", "Qwen 3 235B"),
    ],
)
# --- Cohort B: its native 6-model roster on the 663-event dataset ----------
COHORT_B = dict(
    data="cohortB_final_dataset.csv",
    roster=[
        ("agent-gemini-3.1-pro", "Gemini 3.1 Pro (agent)"),
        ("agent-gpt-5.5-high", "GPT-5.5 High (agent)"),
        ("claude-opus-4.8-thinking", "Claude Opus 4.8"),
        ("deepseek-v4-pro", "DeepSeek V4 Pro"),
        ("grok-4.3", "Grok 4.3"),
        ("qwen-3.6-plus", "Qwen 3.6 Plus"),
    ],
)

ADD_COLS = ["Proper", "Max (Mkt)", "Max (Grp)", "Inv-Margin", "Kelly-Alike"]
KELLY_VARIANTS = [  # label, fraction, allow_leverage
    ("K_full_cap", 1.0, False), ("K_full_lev", 1.0, True),
    ("K_half_cap", 0.5, False), ("K_half_lev", 0.5, True),
]


def load(path):
    df = pd.read_csv(os.path.join(HERE, path), low_memory=False)
    if "yes_ask_prob" not in df.columns and "yes_ask" in df.columns:
        df["yes_ask_prob"] = df["yes_ask"] / 100.0
        df["no_ask_prob"] = df["no_ask"] / 100.0
    df["outcome"] = pd.to_numeric(df["outcome"], errors="coerce")
    df = df.dropna(subset=["model_prob_yes", "yes_ask_prob", "no_ask_prob", "outcome"])
    return df[(df["yes_ask_prob"] > 0) & (df["no_ask_prob"] > 0)]


def run_cohort(name, cfg):
    df = load(cfg["data"])
    have = set(df["predictor_name"])
    roster = [(p, l) for p, l in cfg["roster"] if p in have]
    missing = [l for p, l in cfg["roster"] if p not in have]
    nev = df["event_ticker"].nunique()

    marg_rows, kelly_rows, pair_rows = [], [], []
    for pred, label in roster:
        sub = df[df["predictor_name"] == pred]
        events = build_events(sub)
        n = len(events)
        counts = np.random.default_rng(SEED).multinomial(n, np.full(n, 1.0 / n), size=B).astype(float)

        # dS (relative-Brier gap)
        ds_num, ds_cnt = per_event_dS(sub, [e["event_ticker"] for e in events])
        ds_pt = ds_num.sum() / ds_cnt.sum() if ds_cnt.sum() > 0 else np.nan
        ds_se = np.nanstd(boot_ratio(ds_num, ds_cnt, counts) / 100.0)

        # additive per-bet strategies: point ROI + marginal CI + paired boot draws
        rec = {"model": label, "dS": ds_pt, "dS_ci90": Z90 * ds_se}
        boot = {}
        for cname, runner in [PROPER] + BASELINES:
            ep, ec = per_event_pc(events, runner)
            rec[cname] = roi(ep, ec)
            bd = boot_ratio(ep, ec, counts)
            rec[cname + "_ci90"] = Z90 * float(np.nanstd(bd))
            boot[cname] = bd
        marg_rows.append(rec)

        # paired ROI-difference CIs: Proper - each baseline (same resample)
        for cname, _ in BASELINES:
            d = (boot[PROPER[0]] - boot[cname])
            d = d[np.isfinite(d)]
            lo, hi = np.percentile(d, [5, 95])
            pair_rows.append({"model": label, "baseline": cname,
                              "diff_ROI": rec[PROPER[0]] - rec[cname],
                              "lo90": lo, "hi90": hi,
                              "pval2": min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))})

        # compounded Kelly diagnostic (point only; path-dependent)
        krow = {"model": label, "Kelly-Alike (per-bet)": rec["Kelly-Alike"]}
        for kname, frac, lev in KELLY_VARIANTS:
            krow[kname] = _run_kelly_internal(events, fraction=frac, allow_leverage=lev)["roi"]
        kelly_rows.append(krow)

    marg = pd.DataFrame(marg_rows).sort_values("dS", ascending=False).reset_index(drop=True)
    order = {m: i for i, m in enumerate(marg["model"])}
    kelly = pd.DataFrame(kelly_rows).set_index("model").loc[marg["model"]].reset_index()
    pair = pd.DataFrame(pair_rows)
    pair = pair.sort_values(by=["model", "baseline"],
                            key=lambda s: s.map(order) if s.name == "model" else s).reset_index(drop=True)

    _print_cohort(name, nev, roster, missing, marg, kelly)
    _print_paired(name, marg, pair)

    marg.round(4).to_csv(os.path.join(HERE, f"cohort{name}_table6_perbet_ci.csv"), index=False)
    kelly.round(2).to_csv(os.path.join(HERE, f"cohort{name}_kelly_compounding.csv"), index=False)
    pair.round(4).to_csv(os.path.join(HERE, f"cohort{name}_table6_paired_ci.csv"), index=False)
    _write_paired_latex(name, marg, pair)
    return marg, kelly, pair


def _print_paired(name, marg, pair):
    """Headline inferential view: paired event-clustered 90% CIs on
    ROI(Proper) - ROI(baseline). Same resampled events for both strategies."""
    bases = [name for name, _ in BASELINES]
    proper = dict(zip(marg["model"], marg["Proper"]))
    print(f"\n  Table 6 CIs -- PAIRED event-clustered 90% CI on ROI(Proper) - ROI(baseline)")
    print(f"  (same resample for both arms; '*' = CI excludes 0; sign = which arm wins)")
    print(f"{'Model':<20}{'Proper ROI':>11} | " + " ".join(f"{b:>18}" for b in bases))
    for _, r in marg.iterrows():
        cells = []
        for b in bases:
            pr = pair[(pair["model"] == r["model"]) & (pair["baseline"] == b)].iloc[0]
            excl = (pr["lo90"] > 0) or (pr["hi90"] < 0)
            s = f"{pr['diff_ROI']:>+5.1f}[{pr['lo90']:>+5.1f},{pr['hi90']:>+5.1f}]"
            cells.append(s + ("*" if excl else " "))
        print(f"{r['model']:<20}{r['Proper']:>+11.1f} | " + " ".join(cells))


def _write_paired_latex(name, marg, pair):
    bases = [name for name, _ in BASELINES]
    nb = len(bases)
    L = [rf"\begin{{tabular}}{{l r *{{{nb}}}{{r}}}}", r"\toprule",
         rf"& & \multicolumn{{{nb}}}{{c}}{{$\Delta$ROI = ROI(Proper) $-$ ROI(baseline), "
         r"paired event-clustered 90\% CI}} \\",
         rf"\cmidrule(lr){{3-{2 + nb}}}",
         r"\textbf{Model} & \textbf{Proper} & " + " & ".join(bases) + r" \\", r"\midrule"]
    for _, r in marg.iterrows():
        cells = []
        for b in bases:
            pr = pair[(pair["model"] == r["model"]) & (pair["baseline"] == b)].iloc[0]
            excl = (pr["lo90"] > 0) or (pr["hi90"] < 0)
            body = f"{pr['diff_ROI']:+.1f}\\,[{pr['lo90']:+.1f},{pr['hi90']:+.1f}]".replace("-", "$-$")
            cells.append(f"\\textbf{{{body}}}" if excl else body)
        pr_roi = f"{r['Proper']:+.1f}".replace("-", "$-$")
        L.append(f"{r['model']:<18} & ${pr_roi}$ & " + " & ".join(f"${c}$" for c in cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(HERE, f"cohort{name}_table6_paired_ci.tex"), "w") as f:
        f.write("\n".join(L) + "\n")


def _fmt(v, hw=None):
    if hw is None:
        return f"{v:>+8.1f}"
    return f"{v:>+7.1f}±{hw:>4.1f}"


def _print_cohort(name, nev, roster, missing, marg, kelly):
    print(f"\n{'='*118}\nCOHORT {name}  --  Table 6 (per-bet ROI %, like-for-like) "
          f"on {nev} events, {len(roster)} models  [seed {SEED}, B={B}, 90% CI]")
    if missing:
        print(f"  excluded (not on this event set / not in roster): {', '.join(missing)}")
    print("="*118)
    print(f"{'Model':<20}{'dS':>18} | " + " ".join(f"{c:>13}" for c in ADD_COLS))
    for _, r in marg.iterrows():
        best = max(ADD_COLS, key=lambda c: r[c])
        cells = []
        for c in ADD_COLS:
            s = _fmt(r[c], r[c + "_ci90"])
            cells.append((s + "*") if c == best else (s + " "))
        print(f"{r['model']:<20}{r['dS']:>+9.4f}±{r['dS_ci90']:<7.4f} | " + " ".join(cells))
    print("  (*) best per-bet ROI in the row.")

    print(f"\n  Kelly diagnostic -- COMPOUNDED sequential bankroll (terminal growth %, "
          f"NOT comparable to per-bet ROI above):")
    print(f"{'Model':<20} {'Kelly-Alike':>12} | {'Full/capped':>12} {'Full/lev':>12} "
          f"{'Half/capped':>12} {'Half/lev':>12}")
    print(f"{'':<20} {'(per-bet)':>12} | {'(standard)':>12} {'(paper)':>12} "
          f"{'(standard)':>12} {'':>12}")
    for _, r in kelly.iterrows():
        def g(v):
            return f"{v:>+12.0f}" if abs(v) >= 1000 else f"{v:>+12.1f}"
        print(f"{r['model']:<20} {r['Kelly-Alike (per-bet)']:>+12.1f} | "
              f"{g(r['K_full_cap'])} {g(r['K_full_lev'])} {g(r['K_half_cap'])} {g(r['K_half_lev'])}")


def main():
    run_cohort("A", COHORT_A)
    run_cohort("B", COHORT_B)
    print("\nWrote cohort{A,B}_table6_perbet_ci.csv, cohort{A,B}_kelly_compounding.csv, "
          "cohort{A,B}_table6_paired_ci.csv, cohort{A,B}_table6_paired_ci.tex")


if __name__ == "__main__":
    main()
