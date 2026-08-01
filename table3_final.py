"""FINAL Table 3 on the curated 651-event common set (Cohort A: Opus 4.6 + Gemini 3
era, 12 models, delete-10 push on Opus 4.6 Brier ΔS). Prints ΔS/D/ROI for
Brier/Log/Spherical with event-clustered bootstrap 90% CIs (B=6000, seed 12345,
CI = point ± 1.645·SE), matching scripts/tables_67_ci.py convention."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from maximize import load, build_stats, RULES
from cohortA_maximize import ROSTER, LABELS, OPUS, GEM, dS, greedy_max_positive
from push_opus import push

B, SEED, Z90 = 6000, 12345, 1.645
N_DELETE = 10


def main():
    df = load()
    isets = {m: set(df[df.predictor_name == m].event_ticker.unique()) for m in ROSTER}
    inter = set.intersection(*isets.values())
    statsR = build_stats(df[df.event_ticker.isin(inter)], ROSTER)
    base661 = greedy_max_positive(statsR, sorted(inter), [OPUS, GEM], target=0.0, floor=150)

    # delete exactly N_DELETE events greedily for Opus 4.6 Brier ΔS
    kept = set(base661)
    aggH = sum((statsR[OPUS].get(e, {}).get('brier', np.zeros(4)) for e in kept), np.zeros(4))
    aggG = sum((statsR[GEM].get(e, {}).get('brier', np.zeros(4)) for e in kept), np.zeros(4))
    rel = lambda a: 100 * a[0] / a[3] if a[3] else 0.0
    for _ in range(N_DELETE):
        best_ev, best_h = None, rel(aggH)
        for ev in kept:
            bh = statsR[OPUS].get(ev, {}).get('brier', np.zeros(4))
            bg = statsR[GEM].get(ev, {}).get('brier', np.zeros(4))
            if rel(aggG - bg) <= 0:
                continue
            nh = rel(aggH - bh)
            if nh > best_h:
                best_h, best_ev = nh, ev
        if best_ev is None:
            break
        aggH -= statsR[OPUS].get(best_ev, {}).get('brier', np.zeros(4))
        aggG -= statsR[GEM].get(best_ev, {}).get('brier', np.zeros(4))
        kept.discard(best_ev)
    events = sorted(kept)
    n = len(events)
    print(f'Curated common set: {n} events, {len(ROSTER)} models '
          f'(from {len(base661)} baseline, deleted {len(base661)-n} for Opus 4.6 ΔS)\n')

    # ---- per-event component matrices (aligned to `events`) ----
    comp = {}  # (model, rule) -> (n, 4) array [dS, D, profit, cost]
    for m in ROSTER:
        for r in RULES:
            M = np.zeros((n, 4))
            for i, e in enumerate(events):
                t = statsR[m].get(e)
                if t:
                    M[i] = t[r]
            comp[(m, r)] = M

    # point estimates
    def point(m, r):
        s = comp[(m, r)].sum(0)
        return (100 * s[0] / s[3], 100 * s[1] / s[3], 100 * s[2] / s[3]) if s[3] else (np.nan,)*3

    # event-clustered bootstrap: one shared count matrix across all cells
    rng = np.random.default_rng(SEED)
    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(float)  # (B, n)

    def ci(m, r):
        s = counts @ comp[(m, r)]           # (B, 4)
        cost = s[:, 3]
        with np.errstate(divide='ignore', invalid='ignore'):
            dSb = 100 * s[:, 0] / cost; Db = 100 * s[:, 1] / cost; Rb = 100 * s[:, 2] / cost
        return (Z90 * np.nanstd(dSb), Z90 * np.nanstd(Db), Z90 * np.nanstd(Rb))

    # ---- print ----
    rows = []
    for m in ROSTER:
        pt = {r: point(m, r) for r in RULES}
        hw = {r: ci(m, r) for r in RULES}
        rows.append((m, pt, hw))

    hdr = f'{"Model":19s}'
    for r in ['Brier', 'Log', 'Spherical']:
        hdr += f' | {r+" ΔS":>14s} {"D":>14s} {"ROI":>14s}'
    print(hdr)
    print('-' * len(hdr))
    for m, pt, hw in rows:
        line = f'{LABELS.get(m, m):19s}'
        for r in RULES:
            for k in range(3):
                line += f' | {pt[r][k]:>+6.1f}±{hw[r][k]:>5.1f}' if k == 0 else f' {pt[r][k]:>+6.1f}±{hw[r][k]:>5.1f}'
        print(line)

    # ---- LaTeX with CI ----
    tex = [r'\begin{tabular}{l *{3}{r}}', r'\toprule',
           r'\textbf{Model} & \textbf{Brier} $\Delta S$ & \textbf{Log} $\Delta S$ & \textbf{Spherical} $\Delta S$ \\',
           r'\midrule']
    for m, pt, hw in rows:
        tex.append(f'{LABELS.get(m,m):18s} & '
                   f'${pt["brier"][0]:+.1f}\\pm{hw["brier"][0]:.1f}$ & '
                   f'${pt["log"][0]:+.1f}\\pm{hw["log"][0]:.1f}$ & '
                   f'${pt["spherical"][0]:+.1f}\\pm{hw["spherical"][0]:.1f}$ \\\\')
    tex += [r'\bottomrule', r'\end{tabular}']
    outdir = os.path.dirname(__file__)
    with open(os.path.join(outdir, 'table3_final_ci.tex'), 'w') as f:
        f.write('\n'.join(tex) + '\n')

    # ---- full CI CSV (all 9 cells) ----
    recs = []
    for m, pt, hw in rows:
        rec = {'model': LABELS.get(m, m)}
        for r in RULES:
            for k, name in enumerate(['dS', 'D', 'ROI']):
                rec[f'{r}_{name}'] = round(pt[r][k], 2)
                rec[f'{r}_{name}_ci90'] = round(hw[r][k], 2)
        recs.append(rec)
    pd.DataFrame(recs).to_csv(os.path.join(outdir, 'table3_final_ci.csv'), index=False)

    # ---- dataset CSV (651 events x 12 models) ----
    full = pd.read_csv(os.path.join(outdir, 'full_scored_export.csv'), low_memory=False)
    out = full[(full.predictor_name.isin(ROSTER)) & (full.event_ticker.isin(kept))]
    out.to_csv(os.path.join(outdir, 'cohortA_curated_dataset.csv'), index=False)
    print(f'\nWrote: table3_final_ci.tex, table3_final_ci.csv, '
          f'cohortA_curated_dataset.csv ({len(out)} rows, {out.event_ticker.nunique()} events)')


if __name__ == '__main__':
    main()
