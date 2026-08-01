import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from build_gap_dataset import load, per_event_stats, greedy_keep, TARGETS
import build_gap_dataset as B

df = load()
OPUS, GEM = 'agent-anthropic/claude-opus-4.6', 'agent-gemini-3'

# 1) Opus∩Gemini kept set (target 0.001 decomp-safe)
iset = set(df[df.predictor_name==OPUS].event_ticker) & set(df[df.predictor_name==GEM].event_ticker)
stats2 = {m: per_event_stats(df[(df.predictor_name==m)&(df.event_ticker.isin(iset))]) for m in (OPUS,GEM)}
kept, _ = greedy_keep(stats2, sorted(iset), [OPUS,GEM], target=0.001)
print(f'Opus∩Gemini = {len(iset)}; greedy-kept (relBrier>+0.001) = {len(kept)}')

# 2) On the kept set, relBrier + decomp ΔS for EVERY model with >=95% coverage
def agg_gap(m, events):
    s = per_event_stats(df[(df.predictor_name==m)&(df.event_ticker.isin(events))])
    a = sum((s.get(e,np.zeros(5)) for e in events), np.zeros(5))
    num,cost,mb,kb,n = a
    cov = sum(1 for e in events if e in s and s[e][4]>0)
    return (100*num/cost if cost else float('nan'), (kb-mb)/n if n else float('nan'), int(n), cov)

allm = df.predictor_name.value_counts()
cand = [m for m in allm.index if len(set(df[df.predictor_name==m].event_ticker) & kept) >= 0.98*len(kept)]
print(f'\nModels covering >=98% of the {len(kept)} kept events: {len(cand)}')
print(f'{"model":38s} {"decompDS%":>10s} {"relBrier":>9s} {"nBets":>6s} {"cov":>4s}')
rows=[]
for m in cand:
    dS,rel,n,cov = agg_gap(m, kept)
    rows.append((rel,m,dS,n,cov))
for rel,m,dS,n,cov in sorted(rows, reverse=True):
    pos = 'POS' if (rel>0 and dS>0) else ''
    print(f'{m:38s} {dS:>+10.2f} {rel:>+9.4f} {n:>6d} {cov:>4d}  {pos}')
