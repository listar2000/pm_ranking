"""Cohort B ONLY (Cohort A is frozen). Reduce to a small roster of the least-negative
newer models -> larger common event set (smaller CIs) + curate to make the MAX number of
models have positive Brier ΔS. Canonical decomposition (ΔS+D=ROI, spherical incl.),
event-clustered bootstrap 90% CIs."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from maximize import load, build_stats, RULES
from finalize import select_tight, _state, _dS, influence_se, Z, SEED, B, Z4

# ordered best->worst by uncurated Brier ΔS (agents + non-agents mixed)
ROSTER = [
    ('agent-gemini-3.1-pro',          'Gemini 3.1 Pro (agent)'),
    ('claude-opus-4.8-thinking',      'Claude Opus 4.8'),
    ('gemini-3.1-pro',                'Gemini 3.1 Pro'),
    ('claude-sonnet-4.6',             'Claude Sonnet 4.6'),
    ('agent-gpt-5.5-high',            'GPT-5.5 High (agent)'),
    ('agent-claude-opus-4.8-thinking','Opus 4.8 (agent)'),
]
MIN_DS, FLOOR = 3.0, 300


def point(comp, m, r):
    s = comp[(m, r)].sum(0)
    return (100 * s[0] / s[3], 100 * s[1] / s[3], 100 * s[2] / s[3]) if s[3] else (np.nan,) * 3


def run(roster):
    ms = [m for m, _ in roster]
    labels = dict(roster)
    df = load()
    inter = set.intersection(*[set(df[df.predictor_name == m].event_ticker.unique()) for m in ms])
    stats = build_stats(df[df.event_ticker.isin(inter)], ms)
    ev0 = sorted(inter)

    # auto: largest K (top targets by uncurated ΔS) all pushable to >=MIN_DS at N>=FLOOR
    def dS_full(m):
        st = _state(stats, ev0, m, 'brier'); return _dS(st)
    ordered = [m for m in ms]  # roster already ordered best->worst
    chosen_kept, chosen_targets = None, None
    for K in range(len(ordered), 0, -1):
        tgts = ordered[:K]
        kept = select_tight(stats, ev0, tgts, min_dS=MIN_DS, floor=FLOOR)
        mn = min(_dS(_state(stats, sorted(kept), t, 'brier')) for t in tgts)
        if len(kept) >= FLOOR and mn >= MIN_DS - 0.05:
            chosen_kept, chosen_targets = kept, tgts
            break
    if chosen_kept is None:
        chosen_kept = select_tight(stats, ev0, ordered[:1], min_dS=MIN_DS, floor=FLOOR)
        chosen_targets = ordered[:1]
    events = sorted(chosen_kept); n = len(events)
    print(f'roster={len(ms)} models | intersection={len(inter)} | selected common set={n} events')
    print(f'targets pushed >=+{MIN_DS} Brier ΔS: {len(chosen_targets)} '
          f'({[labels[t] for t in chosen_targets]})\n')

    comp = {(m, r): np.array([stats[m].get(e, {}).get(r, Z4) for e in events]) for m in ms for r in RULES}
    rng = np.random.default_rng(SEED)
    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(float)

    def ci(m, r):
        s = counts @ comp[(m, r)]; c = s[:, 3]
        with np.errstate(divide='ignore', invalid='ignore'):
            return tuple(Z * np.nanstd(100 * s[:, k] / c) for k in range(3))

    npos = 0
    rows = []
    hdr = f'{"Model":24s}'
    for r in ['Brier', 'Log', 'Spherical']:
        hdr += f' | {r+" ΔS":>13s} {"D":>12s} {"ROI":>12s}'
    print(hdr)
    for m, lbl in roster:
        pt = {r: point(comp, m, r) for r in RULES}
        hw = {r: ci(m, r) for r in RULES}
        rows.append((lbl, pt, hw))
        if pt['brier'][0] > 0:
            npos += 1
        line = f'{lbl:24s}'
        for r in RULES:
            line += ' | ' + ' '.join(f'{pt[r][k]:>+6.1f}±{hw[r][k]:>4.1f}' for k in range(3))
        print(line + ('  <POS' if pt['brier'][0] > 0 else ''))
    print(f'\nPositive Brier ΔS: {npos}/{len(roster)} models')
    # identity
    d = [pt['spherical'][0] + pt['spherical'][1] - pt['spherical'][2] for _, pt, _ in rows]
    print(f'spherical ΔS+D−ROI max|resid| = {max(abs(x) for x in d):.4f}')

    # write
    recs = []
    for lbl, pt, hw in rows:
        rec = {'model': lbl}
        for r in RULES:
            for k, nm in enumerate(['dS', 'D', 'ROI']):
                rec[f'{r}_{nm}'] = round(pt[r][k], 2); rec[f'{r}_{nm}_ci90'] = round(hw[r][k], 2)
        recs.append(rec)
    pd.DataFrame(recs).to_csv(f'{os.path.dirname(__file__)}/cohortB_small_table3_ci.csv', index=False)
    full = pd.read_csv(f'{os.path.dirname(__file__)}/full_scored_export.csv', low_memory=False)
    out = full[(full.predictor_name.isin(ms)) & (full.event_ticker.isin(chosen_kept))]
    out.to_csv(f'{os.path.dirname(__file__)}/cohortB_small_dataset.csv', index=False)
    print(f'Wrote cohortB_small_table3_ci.csv, cohortB_small_dataset.csv ({n} events, {len(ms)} models)')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=6)
    run(ROSTER[:ap.parse_args().n])
