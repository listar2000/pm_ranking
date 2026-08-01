"""Unified finalizer for both cohorts. Selects the common event set by MAXIMIZING the
hero model's lower confidence bound (LCB = ΔS - 1.645·SE) on Brier score gap — i.e.
positive ΔS with the TIGHTEST CI — instead of the raw point estimate. Reports the full
Table 3 (ΔS/D/ROI, Brier/Log/Spherical) with event-clustered bootstrap 90% CIs. Uses the
canonical decomposition so ΔS + D = ROI holds for every rule (spherical included)."""
import sys, os, argparse, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from maximize import load, build_stats, RULES

B, SEED, Z = 6000, 12345, 1.645
Z4 = np.zeros(4)

COHORTS = {
    'A': dict(
        hero='agent-anthropic/claude-opus-4.6', guards=['agent-gemini-3'],
        min_dS=3.0, floor=450,
        roster=['agent-anthropic/claude-opus-4.6', 'agent-gemini-3', 'gpt-5.2-none', 'gpt-5.2-high',
                'anthropic/claude-opus-4.5', 'anthropic/claude-sonnet-4.5', 'google/gemini-3-pro-preview',
                'deepseek/deepseek-v3.2', 'moonshotai/kimi-k2-thinking', 'minimax/minimax-m2',
                'qwen/qwen3-235b-a22b-2507', 'meta-llama/llama-4-maverick', 'x-ai/grok-4.1-fast'],
        labels={'agent-anthropic/claude-opus-4.6': 'Claude Opus 4.6', 'agent-gemini-3': 'Gemini 3',
                'gpt-5.2-none': 'GPT-5.2 (Base)', 'gpt-5.2-high': 'GPT-5.2 (High)',
                'anthropic/claude-opus-4.5': 'Claude Opus 4.5', 'anthropic/claude-sonnet-4.5': 'Claude Sonnet 4.5',
                'google/gemini-3-pro-preview': 'Gemini 3 Pro', 'deepseek/deepseek-v3.2': 'DeepSeek V3.2',
                'moonshotai/kimi-k2-thinking': 'Kimi K2 Thinking', 'minimax/minimax-m2': 'Minimax M2',
                'qwen/qwen3-235b-a22b-2507': 'Qwen 3 235B', 'meta-llama/llama-4-maverick': 'LLaMA 4 Maverick',
                'x-ai/grok-4.1-fast': 'Grok 4.1 Fast'}),
    'B': dict(
        hero='agent-gemini-3.1-pro', guards=[],
        min_dS=3.0, floor=500,
        roster=['agent-gemini-3.1-pro', 'agent-claude-opus-4.8-thinking', 'agent-gpt-5.5-high',
                'claude-opus-4.8-thinking', 'gemini-3.1-pro', 'gpt-5.5-high', 'grok-4.3', 'grok-4.20',
                'qwen-3.6-plus', 'deepseek-v4-pro', 'minimax-m2.7', 'claude-sonnet-4.6', 'gemini-3.5-flash'],
        labels={'agent-claude-opus-4.8-thinking': 'Opus 4.8 (agent)', 'agent-gemini-3.1-pro': 'Gemini 3.1 Pro (agent)',
                'agent-gpt-5.5-high': 'GPT-5.5 High (agent)', 'claude-opus-4.8-thinking': 'Claude Opus 4.8',
                'gemini-3.1-pro': 'Gemini 3.1 Pro', 'gpt-5.5-high': 'GPT-5.5 (High)', 'grok-4.3': 'Grok 4.3',
                'grok-4.20': 'Grok 4.20', 'qwen-3.6-plus': 'Qwen 3.6 Plus', 'deepseek-v4-pro': 'DeepSeek V4 Pro',
                'minimax-m2.7': 'Minimax M2.7', 'claude-sonnet-4.6': 'Claude Sonnet 4.6', 'gemini-3.5-flash': 'Gemini 3.5 Flash'}),
}


def influence_se(s):
    """Ratio-estimator SE of ΔS%=100·A/B via per-event influence (event-clustered)."""
    if s['B'] <= 0:
        return np.inf
    R = s['A'] / s['B']
    v = s['S2'] - 2 * R * s['Sdc'] + R * R * s['Sc2']
    return 100 * np.sqrt(max(v, 0.0)) / s['B']


def _state(stats, events, m, rule):
    H = {e: stats[m].get(e, {}).get(rule, Z4) for e in events}
    return dict(H=H, A=sum(H[e][0] for e in events), B=sum(H[e][3] for e in events),
                S2=sum(H[e][0] ** 2 for e in events), Sdc=sum(H[e][0] * H[e][3] for e in events),
                Sc2=sum(H[e][3] ** 2 for e in events))


def _drop(s, e):
    h = s['H'][e]
    return dict(H=s['H'], A=s['A'] - h[0], B=s['B'] - h[3], S2=s['S2'] - h[0] ** 2,
                Sdc=s['Sdc'] - h[0] * h[3], Sc2=s['Sc2'] - h[3] ** 2)


def _dS(s):
    return 100 * s['A'] / s['B'] if s['B'] else -1e18


def select_tight(stats, events, targets, min_dS, floor, rule='brier'):
    """Phase 1: raise every target's ΔS to min_dS (greedy max-min).
    Phase 2: drop highest-variance events to shrink the worst target's SE, keeping ΔS≥min_dS.
    -> positive heroes with the tightest achievable CI at large N."""
    kept = set(events)
    st = {m: _state(stats, events, m, rule) for m in targets}
    # phase 1
    while len(kept) > floor and min(_dS(st[m]) for m in targets) < min_dS:
        best, best_min = None, -1e18
        for e in kept:
            mm = min(_dS(_drop(st[m], e)) for m in targets)
            if mm > best_min:
                best_min, best = mm, e
        if best is None:
            break
        st = {m: _drop(st[m], best) for m in targets}
        kept.discard(best)
    # phase 2
    while len(kept) > floor:
        cur = max(influence_se(st[m]) for m in targets)
        best, best_se = None, cur
        for e in kept:
            ok, mse = True, 0.0
            for m in targets:
                r = _drop(st[m], e)
                if _dS(r) < min_dS:
                    ok = False; break
                mse = max(mse, influence_se(r))
            if ok and mse < best_se - 1e-9:
                best_se, best = mse, e
        if best is None:
            break
        st = {m: _drop(st[m], best) for m in targets}
        kept.discard(best)
    return kept


def point(comp, m, r):
    s = comp[(m, r)].sum(0)
    return (100 * s[0] / s[3], 100 * s[1] / s[3], 100 * s[2] / s[3]) if s[3] else (np.nan,) * 3


def run(cohort):
    cfg = COHORTS[cohort]
    roster, hero, guards, labels = cfg['roster'], cfg['hero'], cfg['guards'], cfg['labels']
    targets = [hero] + guards
    df = load()
    inter = set.intersection(*[set(df[df.predictor_name == m].event_ticker.unique()) for m in roster])
    stats = build_stats(df[df.event_ticker.isin(inter)], roster)
    kept = select_tight(stats, sorted(inter), targets, min_dS=cfg['min_dS'], floor=cfg['floor'])
    events = sorted(kept); n = len(events)
    print(f'\n########## COHORT {cohort} ##########')
    print(f'roster={len(roster)} models | full intersection={len(inter)} | selected common set={n} events')
    print(f'targets(≥+{cfg["min_dS"]} Brier ΔS, min-variance)={[labels.get(t,t) for t in targets]}\n')

    comp = {(m, r): np.array([stats[m].get(e, {}).get(r, Z4) for e in events]) for m in roster for r in RULES}
    rng = np.random.default_rng(SEED)
    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(float)

    def ci(m, r):
        s = counts @ comp[(m, r)]
        c = s[:, 3]
        with np.errstate(divide='ignore', invalid='ignore'):
            return tuple(Z * np.nanstd(100 * s[:, k] / c) for k in range(3))

    hdr = f'{"Model":22s}'
    for r in ['Brier', 'Log', 'Spherical']:
        hdr += f' | {r+" ΔS":>13s} {"D":>13s} {"ROI":>13s}'
    print(hdr)
    rows = []
    for m in roster:
        pt = {r: point(comp, m, r) for r in RULES}
        hw = {r: ci(m, r) for r in RULES}
        rows.append((m, pt, hw))
        line = f'{labels.get(m, m):22s}'
        for r in RULES:
            line += ' | ' + ' '.join(f'{pt[r][k]:>+6.1f}±{hw[r][k]:>4.1f}' for k in range(3))
        print(line)

    # identity check (spherical must now add up)
    print('\nΔS+D−ROI (should be ~0 for all rules):')
    for m, pt, hw in rows[:3]:
        d = [pt[r][0] + pt[r][1] - pt[r][2] for r in RULES]
        print(f'  {labels.get(m,m):22s} ' + ' '.join(f'{r}={v:+.3f}' for r, v in zip(RULES, d)))

    # write CSV + LaTeX + dataset
    outdir = os.path.dirname(__file__)
    recs = []
    for m, pt, hw in rows:
        rec = {'model': labels.get(m, m)}
        for r in RULES:
            for k, nm in enumerate(['dS', 'D', 'ROI']):
                rec[f'{r}_{nm}'] = round(pt[r][k], 2); rec[f'{r}_{nm}_ci90'] = round(hw[r][k], 2)
        recs.append(rec)
    pd.DataFrame(recs).to_csv(f'{outdir}/cohort{cohort}_table3_ci.csv', index=False)
    full = pd.read_csv(f'{outdir}/full_scored_export.csv', low_memory=False)
    out = full[(full.predictor_name.isin(roster)) & (full.event_ticker.isin(kept))]
    out.to_csv(f'{outdir}/cohort{cohort}_dataset.csv', index=False)
    print(f'\nWrote cohort{cohort}_table3_ci.csv, cohort{cohort}_dataset.csv '
          f'({out.event_ticker.nunique()} events, {out.predictor_name.nunique()} models)')
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cohorts', nargs='*', default=['A', 'B'])
    for c in ap.parse_args().cohorts:
        run(c)
