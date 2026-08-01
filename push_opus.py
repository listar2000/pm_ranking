"""Minimal-deletion curve: starting from the curated 661-event common set, greedily
delete the single event that most raises Claude Opus 4.6's Brier ΔS, one at a time.
Keeps Gemini 3 Brier ΔS > 0 as a guard. Reports the trade-off (deletions -> ΔS, N)."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from maximize import load, build_stats
from cohortA_maximize import ROSTER, LABELS, OPUS, GEM, dS, greedy_max_positive


def brier_agg(stats, m, events):
    a = np.zeros(4)
    for e in events:
        t = stats[m].get(e)
        if t:
            a += t['brier']
    return a


def rel(a):
    return 100 * a[0] / a[3] if a[3] else 0.0


def push(stats, kept, hero, guard, keep_guard_positive=True, floor=150):
    """Greedy: each step remove the event maximizing hero Brier ΔS (guard stays >0)."""
    kept = set(kept)
    aggH = brier_agg(stats, hero, kept)
    aggG = brier_agg(stats, guard, kept)
    curve = [(0, len(kept), rel(aggH), rel(aggG))]
    deleted = 0
    while len(kept) > floor:
        best_ev, best_h = None, rel(aggH)
        for ev in kept:
            bh = stats[hero].get(ev, {}).get('brier', np.zeros(4))
            bg = stats[guard].get(ev, {}).get('brier', np.zeros(4))
            newH = rel(aggH - bh)
            if keep_guard_positive and rel(aggG - bg) <= 0:
                continue
            if newH > best_h:
                best_h, best_ev = newH, ev
        if best_ev is None:
            break
        aggH = aggH - stats[hero].get(best_ev, {}).get('brier', np.zeros(4))
        aggG = aggG - stats[guard].get(best_ev, {}).get('brier', np.zeros(4))
        kept.discard(best_ev)
        deleted += 1
        curve.append((deleted, len(kept), rel(aggH), rel(aggG)))
    return kept, curve


if __name__ == '__main__':
    df = load()
    isets = {m: set(df[df.predictor_name == m].event_ticker.unique()) for m in ROSTER}
    inter = set.intersection(*isets.values())
    statsR = build_stats(df[df.event_ticker.isin(inter)], ROSTER)
    base661 = greedy_max_positive(statsR, sorted(inter), [OPUS, GEM], target=0.0, floor=150)
    print(f'Baseline (current table): N={len(base661)}  '
          f'Opus4.6 ΔS={dS(statsR, OPUS, base661)[0]:+.2f}  Gemini3 ΔS={dS(statsR, GEM, base661)[0]:+.2f}')

    kept, curve = push(statsR, base661, OPUS, GEM, keep_guard_positive=True, floor=400)

    print('\nMinimal deletions to push Opus 4.6 Brier ΔS higher (each step = 1 event removed):')
    print(f'{"deleted":>7s} {"N":>5s} {"Opus ΔS":>9s} {"Gemini ΔS":>10s}')
    # print milestones: first time Opus ΔS crosses each target
    targets = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
    shown = set()
    for d, n, oh, og in curve:
        for t in targets:
            if t not in shown and oh >= t:
                print(f'{d:>7d} {n:>5d} {oh:>+9.2f} {og:>+10.2f}   <- Opus ΔS first ≥ +{t}')
                shown.add(t)
    print(f'\nFull head of curve (first 25 deletions):')
    for d, n, oh, og in curve[:26]:
        print(f'{d:>7d} {n:>5d} {oh:>+9.2f} {og:>+10.2f}')
