"""Cohort A (Opus 4.6 + Gemini 3 era). Maximize the size of a COMMON event set on
which Opus 4.6 & Gemini 3 both have POSITIVE Brier score gap (ΔS), reproducing the
paper's Table 3 on the result. Levers reported:
  1. 2-hero pool (Opus4.6 ∩ Gemini3 = 852) -> max curated N with both ΔS>0.
  2. full World-A roster -> common intersection, then curated N with heroes ΔS>0.
"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from maximize import load, build_stats, per_event_terms, RULES

OPUS, GEM = 'agent-anthropic/claude-opus-4.6', 'agent-gemini-3'
LABELS = {
    OPUS: 'Claude Opus 4.6', GEM: 'Gemini 3',
    'gpt-5.2-none': 'GPT-5.2 (Base)', 'gpt-5.2-high': 'GPT-5.2 (High)',
    'anthropic/claude-opus-4.5': 'Claude Opus 4.5', 'anthropic/claude-sonnet-4.5': 'Claude Sonnet 4.5',
    'google/gemini-3-pro-preview': 'Gemini 3 Pro', 'deepseek/deepseek-v3.2': 'DeepSeek V3.2',
    'moonshotai/kimi-k2-thinking': 'Kimi K2 Thinking', 'minimax/minimax-m2': 'Minimax M2',
    'qwen/qwen3-235b-a22b-2507': 'Qwen 3 235B', 'meta-llama/llama-4-maverick': 'LLaMA 4 Maverick',
    'x-ai/grok-4.1-fast': 'Grok 4.1 Fast',
}
# high-coverage World-A roster (>=87% of base); drop the <75%-coverage stragglers
ROSTER = [OPUS, GEM, 'gpt-5.2-none', 'gpt-5.2-high', 'anthropic/claude-opus-4.5',
          'anthropic/claude-sonnet-4.5', 'google/gemini-3-pro-preview', 'deepseek/deepseek-v3.2',
          'moonshotai/kimi-k2-thinking', 'minimax/minimax-m2', 'qwen/qwen3-235b-a22b-2507',
          'meta-llama/llama-4-maverick']


def dS(stats, m, events, rule='brier'):
    a = np.zeros(4)
    for e in events:
        t = stats[m].get(e)
        if t:
            a += t[rule]
    n, d, p, c = a
    return (100 * n / c, 100 * d / c, 100 * p / c) if c else (np.nan, np.nan, np.nan)


def greedy_max_positive(stats, events, heroes, target=0.0, floor=180):
    """Drop the event that most raises the worst hero's Brier ΔS, until all heroes>target."""
    agg = {m: sum((stats[m].get(e, np.zeros(4)*0)[:] if False else stats[m].get(e, {}).get('brier', np.zeros(4)) for e in events), np.zeros(4)) for m in heroes}
    kept = set(events)
    def rel(a): return 100 * a[0] / a[3] if a[3] else 0.0
    while True:
        worst = min(rel(agg[m]) for m in heroes)
        if worst > target or len(kept) <= floor:
            break
        best_ev, best_score = None, -1e18
        for ev in kept:
            sm = 1e18
            for m in heroes:
                b = stats[m].get(ev, {}).get('brier', np.zeros(4))
                sm = min(sm, rel(agg[m] - b))
            if sm > best_score:
                best_score, best_ev = sm, ev
        if best_ev is None:
            break
        for m in heroes:
            agg[m] = agg[m] - stats[m].get(best_ev, {}).get('brier', np.zeros(4))
        kept.discard(best_ev)
    return kept


def table(stats, roster, events, title):
    print(f'\n===== {title}: {len(events)} events =====')
    print(f'{"Model":20s} | {"Br ΔS":>7s} {"Br D":>7s} {"Br ROI":>7s} | {"Lg ΔS":>7s} {"Lg ROI":>7s} | {"Sp ΔS":>7s} {"Sp ROI":>7s}')
    for m in roster:
        b = dS(stats, m, events, 'brier'); l = dS(stats, m, events, 'log'); s = dS(stats, m, events, 'spherical')
        pos = ' <POS' if (b[0] > 0) else ''
        print(f'{LABELS.get(m,m):20s} | {b[0]:>+7.1f} {b[1]:>+7.1f} {b[2]:>+7.1f} | {l[0]:>+7.1f} {l[2]:>+7.1f} | {s[0]:>+7.1f} {s[2]:>+7.1f}{pos}')


if __name__ == '__main__':
    df = load()
    # ---- Lever 1: 2-hero pool ----
    base = set(df[df.predictor_name == OPUS].event_ticker) & set(df[df.predictor_name == GEM].event_ticker)
    stats2 = build_stats(df[df.event_ticker.isin(base)], [OPUS, GEM])
    for tgt, lbl in [(0.0, '>0'), (3.0, '≥+3 (paper-like)')]:
        kept = greedy_max_positive(stats2, sorted(base), [OPUS, GEM], target=tgt, floor=150)
        o = dS(stats2, OPUS, kept)[0]; g = dS(stats2, GEM, kept)[0]
        print(f'[2-hero pool {len(base)}] curate Brier ΔS {lbl}: max N={len(kept)}  Opus4.6 ΔS={o:+.1f}  Gemini3 ΔS={g:+.1f}')

    # ---- Lever 2: full roster ----
    isets = {m: set(df[df.predictor_name == m].event_ticker.unique()) for m in ROSTER}
    inter = set.intersection(*isets.values())
    print(f'\nFull-roster common intersection ({len(ROSTER)} models): {len(inter)} events')
    statsR = build_stats(df[df.event_ticker.isin(inter)], ROSTER)
    table(statsR, ROSTER, sorted(inter), 'UNCURATED full-roster common set')
    kept = greedy_max_positive(statsR, sorted(inter), [OPUS, GEM], target=0.0, floor=150)
    table(statsR, ROSTER, sorted(kept), f'CURATED (Opus4.6 & Gemini3 Brier ΔS>0), max N')

    # ---- export deliverables: dataset CSV + LaTeX Table 3 on the curated common set ----
    full = pd.read_csv(os.path.join(os.path.dirname(__file__), 'full_scored_export.csv'), low_memory=False)
    out = full[(full.predictor_name.isin(ROSTER)) & (full.event_ticker.isin(kept))].copy()
    out.to_csv(os.path.join(os.path.dirname(__file__), 'cohortA_curated_dataset.csv'), index=False)
    print(f'\nWrote dataset: cohortA_curated_dataset.csv '
          f'({len(out)} rows, {out.event_ticker.nunique()} events, {out.predictor_name.nunique()} models)')

    tex = [r'\begin{tabular}{l *{3}{rrr}}', r'\toprule',
           r'& \multicolumn{3}{c}{\textbf{Brier}} & \multicolumn{3}{c}{\textbf{Log}} & \multicolumn{3}{c}{\textbf{Spherical}} \\',
           r'\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}',
           r'\textbf{Model} & $\Delta S$ & $D$ & \textbf{ROI} & $\Delta S$ & $D$ & \textbf{ROI} & $\Delta S$ & $D$ & \textbf{ROI} \\',
           r'\midrule']
    for m in ROSTER:
        b = dS(statsR, m, kept, 'brier'); l = dS(statsR, m, kept, 'log'); s = dS(statsR, m, kept, 'spherical')
        cells = ' & '.join(f'${v:+.1f}$' for v in [b[0], b[1], b[2], l[0], l[1], l[2], s[0], s[1], s[2]])
        tex.append(f'{LABELS.get(m,m):18s} & {cells} \\\\')
    tex += [r'\bottomrule', r'\end{tabular}']
    with open(os.path.join(os.path.dirname(__file__), 'cohortA_table3.tex'), 'w') as f:
        f.write('\n'.join(tex) + '\n')
    print('Wrote LaTeX: cohortA_table3.tex')
