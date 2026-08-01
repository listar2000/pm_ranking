import sys, numpy as np
sys.path.insert(0, __file__.rsplit('/',1)[0])
from maximize import load, build_stats, gaps, RULES

df = load()
COHORTS = {
 'A (Opus4.6 era)': ['agent-anthropic/claude-opus-4.6','agent-gemini-3'],
 'B (Opus4.8 era)': ['agent-claude-opus-4.8-thinking','agent-gemini-3.1-pro'],
}
for name, heroes in COHORTS.items():
    isets = {m:set(df[df.predictor_name==m].event_ticker.unique()) for m in heroes}
    inter = set.intersection(*isets.values())
    events = sorted(inter)
    counts = df[df.event_ticker.isin(inter)].groupby('predictor_name').event_ticker.nunique()
    cover = [m for m,c in counts.items() if c >= 0.99*len(inter)]
    stats = build_stats(df[df.event_ticker.isin(inter)], cover)
    rows=[]
    for m in cover:
        g = gaps(stats, m, events)
        rows.append((m, g['brier'][0], g['brier'][2], g['log'][0], g['log'][2]))
    rows.sort(key=lambda r:-r[1])
    npos = sum(1 for r in rows if r[1]>0 and r[2]>0)
    print(f'\n===== COHORT {name}: common set = {len(inter)} events; {len(cover)} models cover it; {npos} POSITIVE (Brier ΔS&ROI) =====')
    print(f'{"model":40s} {"BrΔS":>7s} {"BrROI":>7s} {"LgΔS":>7s} {"LgROI":>7s}')
    for m,bds,bro,lds,lro in rows:
        pos=' <POS' if (bds>0 and bro>0) else ''
        print(f'{m:40s} {bds:>+7.1f} {bro:>+7.1f} {lds:>+7.1f} {lro:>+7.1f}{pos}')
