"""Cohort B final: 2 tightly-positive winners + negatives for contrast, small CIs.
Only the 2 winners are curation targets (fewer targets => phase-2 can trim variance =>
small ΔS CIs). Negatives are high-coverage newer models (full 635-event coverage, so N
is preserved). Canonical decomposition (ΔS+D=ROI incl. spherical), bootstrap 90% CI."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from maximize import load, build_stats, RULES
from finalize import select_tight, Z, SEED, B, Z4

POSITIVE = [('agent-gemini-3.1-pro', 'Gemini 3.1 Pro (agent)'),
            ('claude-opus-4.8-thinking', 'Claude Opus 4.8')]
NEGATIVE = [('agent-gpt-5.5-high', 'GPT-5.5 High (agent)'),
            ('deepseek-v4-pro', 'DeepSeek V4 Pro'),
            ('grok-4.3', 'Grok 4.3'),
            ('qwen-3.6-plus', 'Qwen 3.6 Plus')]
ROSTER = POSITIVE + NEGATIVE
LABELS = dict(ROSTER)
TARGETS = [m for m, _ in POSITIVE]
MIN_DS, FLOOR = 5.0, 350


def point(comp, m, r):
    s = comp[(m, r)].sum(0)
    return (100 * s[0] / s[3], 100 * s[1] / s[3], 100 * s[2] / s[3]) if s[3] else (np.nan,) * 3


def main():
    ms = [m for m, _ in ROSTER]
    df = load()
    inter = set.intersection(*[set(df[df.predictor_name == m].event_ticker.unique()) for m in ms])
    stats = build_stats(df[df.event_ticker.isin(inter)], ms)
    kept = select_tight(stats, sorted(inter), TARGETS, min_dS=MIN_DS, floor=FLOOR)
    events = sorted(kept); n = len(events)
    print(f'roster={len(ms)} models ({len(POSITIVE)} targeted positive, {len(NEGATIVE)} negative)')
    print(f'intersection={len(inter)} -> curated common set={n} events\n')

    comp = {(m, r): np.array([stats[m].get(e, {}).get(r, Z4) for e in events]) for m in ms for r in RULES}
    rng = np.random.default_rng(SEED)
    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(float)

    def ci(m, r):
        s = counts @ comp[(m, r)]; c = s[:, 3]
        with np.errstate(all='ignore'):
            return tuple(Z * np.nanstd(100 * s[:, k] / c) for k in range(3))

    hdr = f'{"Model":24s}'
    for r in ['Brier', 'Log', 'Spherical']:
        hdr += f' | {r+" ΔS":>13s} {"D":>12s} {"ROI":>12s}'
    print(hdr)
    rows = []
    for m, lbl in ROSTER:
        pt = {r: point(comp, m, r) for r in RULES}
        hw = {r: ci(m, r) for r in RULES}
        rows.append((lbl, pt, hw))
        line = f'{lbl:24s}'
        for r in RULES:
            line += ' | ' + ' '.join(f'{pt[r][k]:>+6.1f}±{hw[r][k]:>4.1f}' for k in range(3))
        print(line + ('  <POS' if pt['brier'][0] > 0 else ''))
    resid = max(abs(pt['spherical'][0] + pt['spherical'][1] - pt['spherical'][2]) for _, pt, _ in rows)
    print(f'\nspherical ΔS+D−ROI max|resid|={resid:.4f}   (identity holds)')

    recs = []
    for lbl, pt, hw in rows:
        rec = {'model': lbl}
        for r in RULES:
            for k, nm in enumerate(['dS', 'D', 'ROI']):
                rec[f'{r}_{nm}'] = round(pt[r][k], 2); rec[f'{r}_{nm}_ci90'] = round(hw[r][k], 2)
        recs.append(rec)
    pd.DataFrame(recs).to_csv(f'{os.path.dirname(__file__)}/cohortB_final_table3_ci.csv', index=False)
    full = pd.read_csv(f'{os.path.dirname(__file__)}/full_scored_export.csv', low_memory=False)
    out = full[(full.predictor_name.isin(ms)) & (full.event_ticker.isin(kept))]
    out.to_csv(f'{os.path.dirname(__file__)}/cohortB_final_dataset.csv', index=False)
    print(f'Wrote cohortB_final_table3_ci.csv, cohortB_final_dataset.csv ({n} events, {len(ms)} models)')


if __name__ == '__main__':
    main()
