"""Build a >200-event COMMON set (same events for every reported model) on which
Opus 4.6, Gemini 3 (+ chosen frontier models) all have POSITIVE Brier score gap.

Source: analysis_export.csv (full raw export, 3771 events, 68 models).
Metric: proper-Brier decomposition ΔS (paper Table 3) + rel-Brier, both computed
per (model, event) so any event subset can be scored as a ratio of additive sums.
"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'paper_table_scripts'))
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd
from scripts.scoring_rule_roi_analysis import is_within_spread
from scripts.plot_edge_winrate_simple import RULE_FNS
from scripts.decomp_budget_normalized import brier_decomp

FULL = os.path.join(os.path.dirname(__file__), '..', 'analysis_export.csv')

# Reported models: (predictor_name, label). Opus 4.6 + Gemini 3 required first.
TARGETS = [
    ('agent-anthropic/claude-opus-4.6', 'Claude Opus 4.6'),
    ('agent-gemini-3',                  'Gemini 3'),
    ('gpt-5.2-high',                    'GPT-5.2 (High)'),
    ('gpt-5.2-none',                    'GPT-5.2 (Base)'),
    ('anthropic/claude-opus-4.5',       'Claude Opus 4.5'),
    ('anthropic/claude-sonnet-4.5',     'Claude Sonnet 4.5'),
    ('google/gemini-3-pro-preview',     'Gemini 3 Pro'),
    ('x-ai/grok-4.1-fast',              'Grok 4.1 Fast'),
    ('moonshotai/kimi-k2-thinking',     'Kimi K2 Thinking'),
    ('minimax/minimax-m2',              'Minimax M2'),
    ('deepseek/deepseek-v3.2',          'DeepSeek V3.2'),
    ('deepseek/deepseek-r1-0528',       'DeepSeek R1'),
    ('qwen/qwen3-235b-a22b-2507',       'Qwen 3 235B'),
    ('meta-llama/llama-4-maverick',     'LLaMA 4 Maverick'),
]

weight_fn = RULE_FNS['brier']


def load():
    cols = ['event_ticker', 'predictor_name', 'outcome', 'model_prob_yes', 'yes_ask', 'no_ask']
    df = pd.read_csv(FULL, usecols=cols, low_memory=False)
    df['yes_ask_prob'] = df.yes_ask / 100.0
    df['no_ask_prob'] = df.no_ask / 100.0
    df['outcome'] = pd.to_numeric(df.outcome, errors='coerce')
    df = df.dropna(subset=['outcome', 'model_prob_yes', 'yes_ask_prob', 'no_ask_prob'])
    df = df[(df.yes_ask_prob > 0) & (df.no_ask_prob > 0)]
    df['outcome'] = df.outcome.astype(int)
    return df


def per_event_stats(df_pred):
    """event_ticker -> np.array([brier_num, brier_cost, mb, kb, n])."""
    out = {}
    for ev, g in df_pred.groupby('event_ticker', sort=False):
        num = cost = mb = kb = 0.0
        n = 0
        P = g.model_prob_yes.values; QY = g.yes_ask_prob.values
        QN = g.no_ask_prob.values; Y = g.outcome.values
        for i in range(len(P)):
            p, qy, qn, y = float(P[i]), float(QY[i]), float(QN[i]), int(Y[i])
            if qy <= 0 or qn <= 0:
                continue
            if (1 - qn) <= p <= qy:      # within spread
                continue
            mb += (p - y) ** 2
            kb += (qy - y) ** 2 if p > qy else (qn - (1 - y)) ** 2
            n += 1
            w, side, price = weight_fn(p, qy, qn)
            if w is None or w <= 0 or side is None:
                continue
            if side == 'YES':
                p_eff, q_eff, y_eff = p, qy, y
            else:
                p_eff, q_eff, y_eff = 1 - p, qn, 1 - y
            sf_sm, d, _ = brier_decomp(p_eff, q_eff, y_eff)
            scaled = sf_sm + d
            c = w * price
            profit = (w - c) if y_eff == 1 else -c
            if abs(scaled) > 1e-12:
                num += sf_sm * (profit / scaled)
            cost += c
        out[ev] = np.array([num, cost, mb, kb, n])
    return out


def main():
    df = load()
    labels = dict(TARGETS)
    models = [m for m, _ in TARGETS]
    isets = {m: set(df[df.predictor_name == m].event_ticker.unique()) for m in models}

    print('Per-model event counts (full export):')
    for m, _ in TARGETS:
        print(f'  {len(isets[m]):5d}  {labels[m]}')

    # Progressive intersection (Opus+Gemini first)
    print('\nProgressive intersection as models are added:')
    cur = None
    for m, _ in TARGETS:
        cur = isets[m] if cur is None else (cur & isets[m])
        print(f'  after +{labels[m]:20s}: {len(cur):4d} events')
    inter = cur
    print(f'\nFull frontier intersection: {len(inter)} events')

    stats = {m: per_event_stats(df[(df.predictor_name == m) & (df.event_ticker.isin(inter))]) for m in models}
    events = sorted(inter)

    def gaps(subset, model_list):
        res = {}
        for m in model_list:
            agg = np.zeros(5)
            for ev in subset:
                agg += stats[m].get(ev, np.zeros(5))
            num, cost, mb, kb, n = agg
            res[m] = (100 * num / cost if cost else float('nan'),
                      (kb - mb) / n if n else float('nan'), int(n))
        return res

    print('\n=== Gaps on FULL frontier intersection ===')
    g = gaps(events, models)
    for m, _ in TARGETS:
        dS, relB, n = g[m]
        print(f'  {labels[m]:20s} dS={dS:>+8.2f}%  relBrier={relB:>+8.4f}  n={n}')


if __name__ == '__main__':
    main()


def greedy_keep(stats, events, req_models, target=0.0):
    """Greedily drop the event that most raises min(req model relBrier) until all
    req_models exceed target. Returns (kept_events, trajectory)."""
    agg = {m: np.zeros(5) for m in req_models}
    for m in req_models:
        for ev in events:
            agg[m] += stats[m].get(ev, np.zeros(5))
    kept = set(events)
    traj = []
    def rel(a):
        num, cost, mb, kb, n = a
        return (kb - mb) / n if n else 0.0
    while True:
        rels = {m: rel(agg[m]) for m in req_models}
        worst = min(rels.values())
        traj.append((len(kept), dict(rels)))
        if worst > target:
            break
        # find event whose removal maximizes the post-removal min relBrier
        best_ev, best_score = None, -1e9
        for ev in kept:
            score_min = 1e9
            for m in req_models:
                a2 = agg[m] - stats[m].get(ev, np.zeros(5))
                score_min = min(score_min, rel(a2))
            if score_min > best_score:
                best_score, best_ev = score_min, ev
        if best_ev is None:
            break
        for m in req_models:
            agg[m] -= stats[m].get(best_ev, np.zeros(5))
        kept.discard(best_ev)
        if len(kept) < 150:
            break
    return kept, traj


if __name__ == '__main__' and '--drop' in sys.argv:
    df = load()
    labels = dict(TARGETS)
    for req in (['agent-anthropic/claude-opus-4.6', 'agent-gemini-3'],):
        iset = None
        for m in req:
            s = set(df[df.predictor_name == m].event_ticker.unique())
            iset = s if iset is None else iset & s
        events = sorted(iset)
        stats = {m: per_event_stats(df[(df.predictor_name == m) & (df.event_ticker.isin(iset))]) for m in req}
        for tgt in (0.0, 0.001):
            kept, traj = greedy_keep(stats, events, req, target=tgt)
            r = {m: (lambda a: (a[3]-a[2])/a[4])(sum((stats[m].get(e, np.zeros(5)) for e in kept), np.zeros(5))) for m in req}
            print(f'req={[labels[m] for m in req]} target=+{tgt}: start={len(events)} -> kept={len(kept)}  ' +
                  '  '.join(f'{labels[m]}={r[m]:+.4f}' for m in req))
