"""Explore per-model score gap on the common event intersection of the
standardized 159-event CSV. Goal: find a common event set > 200 with positive
Brier score gap for a chosen set of frontier models."""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'paper_table_scripts'))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from scripts.scoring_rule_roi_analysis import load_csv, is_within_spread
from scripts.plot_edge_winrate_simple import RULE_FNS
from scripts.decomp_budget_normalized import brier_decomp

CSV = os.path.join(ROOT, 'neurips', 'standardized_159_events.csv')

# All frontier-ish models present in the CSV (predictor_name -> label)
ALL_LABELS = {
    'agent-anthropic/claude-opus-4.6': 'Claude Opus 4.6',
    'agent-gemini-3': 'Gemini 3',
    'gpt-5.2-high': 'GPT-5.2 (High)',
    'gpt-5.2-none': 'GPT-5.2 (Base)',
    'anthropic/claude-opus-4.5': 'Claude Opus 4.5',
    'anthropic/claude-sonnet-4.5': 'Claude Sonnet 4.5',
    'x-ai/grok-4.1-fast': 'Grok 4.1 Fast',
    'x-ai/grok-4': 'Grok 4',
    'moonshotai/kimi-k2-thinking': 'Kimi K2 Thinking',
    'minimax/minimax-m2': 'Minimax M2',
    'deepseek/deepseek-v3.2': 'DeepSeek V3.2',
    'deepseek/deepseek-r1-0528': 'DeepSeek R1',
    'qwen/qwen3-235b-a22b-2507': 'Qwen 3 235B',
    'meta-llama/llama-4-maverick': 'LLaMA 4 Maverick',
    'gpt-5.1-high': 'GPT-5.1 (High)',
}


def per_event_stats(df_pred):
    """Return dict event_ticker -> (brier_num, brier_cost, mb_sum, kb_sum, n)
    brier_num/brier_cost are the decomposition-ΔS numerator & cost (proper Brier
    strategy). mb/kb/n are the rel-Brier sums (model brier, market brier, count)."""
    weight_fn = RULE_FNS['brier']
    out = {}
    for ev, g in df_pred.groupby('event_ticker', sort=False):
        num = cost = mb = kb = 0.0
        n = 0
        for _, r in g.iterrows():
            p, qy, qn, y = r['model_prob_yes'], r['yes_ask_prob'], r['no_ask_prob'], r['outcome']
            if pd.isna(p) or pd.isna(qy) or pd.isna(qn) or pd.isna(y):
                continue
            if qy <= 0 or qn <= 0:
                continue
            o = {'p': float(p), 'q_yes': float(qy), 'q_no': float(qn), 'actual': int(y)}
            if is_within_spread(o):
                continue
            y = int(y)
            # rel-Brier (find_shared_drops metric)
            mb += (p - y) ** 2
            kb += (qy - y) ** 2 if p > qy else (qn - (1 - y)) ** 2
            n += 1
            # decomposition-ΔS proper-Brier strategy
            w, side, price = weight_fn(p, qy, qn)
            if w is None or w <= 0 or side is None:
                continue
            if side == 'YES':
                p_eff, q_eff, y_eff = p, qy, y
            else:
                p_eff, q_eff, y_eff = 1 - p, qn, 1 - y
            sf_sm_unit, d_unit, _ = brier_decomp(p_eff, q_eff, y_eff)
            scaled_unit = sf_sm_unit + d_unit
            c = w * price
            is_win = (y_eff == 1)
            profit = (w - c) if is_win else -c
            if abs(scaled_unit) > 1e-12:
                scale = profit / scaled_unit
                num += sf_sm_unit * scale
            cost += c
        out[ev] = (num, cost, mb, kb, n)
    return out


def main():
    df = load_csv(CSV)
    df['outcome'] = pd.to_numeric(df['outcome'], errors='coerce')
    dv = df.dropna(subset=['outcome', 'model_prob_yes', 'yes_ask_prob', 'no_ask_prob'])
    dv = dv[(dv.yes_ask_prob > 0) & (dv.no_ask_prob > 0)]

    models = list(ALL_LABELS.keys())
    isets = {m: set(dv[dv.predictor_name == m].event_ticker.unique()) for m in models}
    inter = set.intersection(*isets.values())
    print(f'Full intersection across ALL {len(models)} models: {len(inter)} events')

    # Per-model per-event stats on the intersection
    stats = {}
    for m in models:
        sub = dv[(dv.predictor_name == m) & (dv.event_ticker.isin(inter))]
        stats[m] = per_event_stats(sub)

    events = sorted(inter)

    def gaps_on(subset):
        res = {}
        for m in models:
            num = cost = mb = kb = 0.0
            n = 0
            for ev in subset:
                a, b, c, d, e = stats[m].get(ev, (0, 0, 0, 0, 0))
                num += a; cost += b; mb += c; kb += d; n += e
            dS = 100 * num / cost if cost else float('nan')
            relB = (kb - mb) / n if n else float('nan')
            res[m] = (dS, relB, n)
        return res

    print('\n=== Score gaps on FULL intersection (all events) ===')
    print(f'{"Model":22s} {"BrierDeltaS%":>12s} {"relBrier":>10s} {"nBets":>6s}')
    base = gaps_on(events)
    for m in models:
        dS, relB, n = base[m]
        flag = '' if (dS > 0 and relB > 0) else '  <-- NEG'
        print(f'{ALL_LABELS[m]:22s} {dS:>+12.2f} {relB:>+10.4f} {n:>6d}{flag}')


if __name__ == '__main__':
    main()
