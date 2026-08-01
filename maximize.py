"""What can we maximize on the full export? Ports neurips/roi_decomposition.decompose_model
EXACTLY, but accumulates per (model, event) so any event subset scores as a ratio of
additive sums. Lets us (a) read each model's Brier/Log/Spherical DeltaS/D/ROI on any common
set, and (b) greedily curate a >200-event common set to maximize positive-gap models."""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'paper_table_scripts')))

FULL = os.path.join(os.path.dirname(__file__), 'full_scored_export.csv')
RULES = ['brier', 'log', 'spherical']

# Canonical decomposition (matches decomp_standardized_200.model_rule_decomposition):
# per-market binary scoring, side chosen by each rule's weight fn, then rescaled by
# profit/(ΔS_unit+D_unit) so ΔS + D = ROI holds EXACTLY for every rule (incl. spherical).
from scripts.plot_edge_winrate_simple import RULE_FNS
from scripts.decomp_budget_normalized import brier_decomp, log_decomp, spherical_decomp
DECOMP_FNS = {'brier': brier_decomp, 'log': log_decomp, 'spherical': spherical_decomp}


def per_event_terms(ge):
    """One event's [ΔS_num, D_num, profit, cost] per rule, with ΔS_num+D_num=profit per bet
    (Savage/Bregman rescale). Identical math to the paper's Table 3 generator."""
    p = ge['model_prob_yes'].values.astype(float)
    qy = ge['yes_ask_prob'].values.astype(float)
    qn = ge['no_ask_prob'].values.astype(float)
    y = ge['outcome'].values.astype(int)
    acc = {r: np.zeros(4) for r in RULES}
    for i in range(len(p)):
        pi, qyi, qni, yi = p[i], qy[i], qn[i], int(y[i])
        if qyi <= 0 or qni <= 0:
            continue
        if (1 - qni) <= pi <= qyi:      # within spread -> no bet
            continue
        for r in RULES:
            w, side, price = RULE_FNS[r](pi, qyi, qni)
            if w is None or w <= 0 or side is None:
                continue
            if side == 'YES':
                p_eff, q_eff, y_eff = pi, qyi, yi
            else:
                p_eff, q_eff, y_eff = 1 - pi, qni, 1 - yi
            sf_sm, d, _ = DECOMP_FNS[r](p_eff, q_eff, y_eff)
            scaled = sf_sm + d
            cost = w * price
            profit = (w - cost) if y_eff == 1 else -cost
            if abs(scaled) > 1e-12:
                sc = profit / scaled
                acc[r] += [sf_sm * sc, d * sc, profit, cost]
            else:
                acc[r] += [0.0, profit, profit, cost]
    return acc


def build_stats(df, models):
    """model -> {event -> {rule: np.array([dS,D,profit,cost])}}"""
    stats = {}
    for m in models:
        sub = df[df.predictor_name == m]
        d = {}
        for ev, ge in sub.groupby('event_ticker', sort=False):
            d[ev] = per_event_terms(ge)
        stats[m] = d
    return stats


def gaps(stats, model, events):
    agg = {r: np.zeros(4) for r in RULES}
    for ev in events:
        t = stats[model].get(ev)
        if t:
            for r in RULES:
                agg[r] += t[r]
    out = {}
    for r in RULES:
        dS, D, profit, cost = agg[r]
        out[r] = (100 * dS / cost, 100 * D / cost, 100 * profit / cost) if cost else (np.nan,) * 3
    return out


def load():
    df = pd.read_csv(FULL, usecols=['event_ticker', 'category', 'predictor_name', 'model_prob_yes',
                                     'yes_ask_prob', 'no_ask_prob', 'outcome'], low_memory=False)
    df = df.dropna(subset=['model_prob_yes', 'yes_ask_prob', 'no_ask_prob', 'outcome'])
    df = df[(df.yes_ask_prob > 0) & (df.no_ask_prob > 0)]
    df['outcome'] = df.outcome.astype(int)
    return df


HEROES = ['agent-claude-opus-4.8-thinking', 'agent-anthropic/claude-opus-4.6', 'agent-gemini-3']

if __name__ == '__main__':
    df = load()
    # intersection of the 3 hero models
    isets = {m: set(df[df.predictor_name == m].event_ticker.unique()) for m in HEROES}
    for m in HEROES:
        print(f'{len(isets[m]):5d}  {m}')
    inter = set.intersection(*isets.values())
    print(f'\nHERO intersection (Opus4.8 & Opus4.6 & Gemini3): {len(inter)} events')

    # Every model that fully (>=99%) covers the hero intersection, ranked by Brier ROI
    events = sorted(inter)
    counts = df[df.event_ticker.isin(inter)].groupby('predictor_name').event_ticker.nunique()
    cover = [m for m, c in counts.items() if c >= 0.99 * len(inter)]
    print(f'Models covering >=99% of the hero set: {len(cover)}\n')
    stats = build_stats(df[df.event_ticker.isin(inter)], cover)
    rows = []
    for m in cover:
        g = gaps(stats, m, events)
        rows.append((m, g['brier'][0], g['brier'][2], g['log'][0], g['log'][2], g['spherical'][0], g['spherical'][2]))
    rows.sort(key=lambda r: -r[2])
    print(f'{"model":40s} {"BrΔS":>7s} {"BrROI":>7s} {"LgΔS":>7s} {"LgROI":>7s} {"SpΔS":>7s} {"SpROI":>7s}')
    for m, bds, bro, lds, lro, sds, sro in rows:
        pos = ' POS' if (bds > 0 and bro > 0) else ''
        print(f'{m:40s} {bds:>+7.1f} {bro:>+7.1f} {lds:>+7.1f} {lro:>+7.1f} {sds:>+7.1f} {sro:>+7.1f}{pos}')
