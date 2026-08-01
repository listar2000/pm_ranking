import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from maximize import load, build_stats, RULES

LABELS = {
 'agent-claude-opus-4.8-thinking':'Opus 4.8 (agent)','agent-gemini-3.1-pro':'Gemini 3.1 Pro (agent)',
 'agent-gpt-5.5-high':'GPT-5.5 High (agent)','claude-opus-4.8-thinking':'Claude Opus 4.8',
 'gemini-3.1-pro':'Gemini 3.1 Pro','gpt-5.5-high':'GPT-5.5 (High)','grok-4.3':'Grok 4.3',
 'grok-4.20':'Grok 4.20','qwen-3.6-plus':'Qwen 3.6 Plus','deepseek-v4-pro':'DeepSeek V4 Pro',
 'minimax-m2.7':'Minimax M2.7','claude-sonnet-4.6':'Claude Sonnet 4.6','gemini-3.5-flash':'Gemini 3.5 Flash',
}
ROSTER = ['agent-claude-opus-4.8-thinking','agent-gemini-3.1-pro','agent-gpt-5.5-high',
 'claude-opus-4.8-thinking','gemini-3.1-pro','gpt-5.5-high','grok-4.3','grok-4.20',
 'qwen-3.6-plus','deepseek-v4-pro','minimax-m2.7','claude-sonnet-4.6','gemini-3.5-flash']

def dS(stats,m,events,rule='brier'):
    a=np.zeros(4)
    for e in events:
        t=stats[m].get(e)
        if t: a+=t[rule]
    n,d,p,c=a
    return (100*n/c,100*d/c,100*p/c) if c else (np.nan,)*3

if __name__=='__main__':
    df=load()
    isets={m:set(df[df.predictor_name==m].event_ticker.unique()) for m in ROSTER}
    inter=set.intersection(*isets.values())
    events=sorted(inter)
    print(f'Cohort B common intersection ({len(ROSTER)} newer models): {len(inter)} events\n')
    stats=build_stats(df[df.event_ticker.isin(inter)],ROSTER)
    rows=[(m,)+dS(stats,m,events,'brier')+dS(stats,m,events,'log')[:1]+dS(stats,m,events,'spherical')[:1] for m in ROSTER]
    rows.sort(key=lambda r:-r[1])
    print(f'{"model":26s} {"BrΔS":>8s} {"BrROI":>8s} {"LgΔS":>8s} {"SpΔS":>8s}')
    for m,bds,bd,bro,lds,sds in rows:
        pos=' <POS' if bds>0 else ''
        print(f'{LABELS.get(m,m):26s} {bds:>+8.1f} {bro:>+8.1f} {lds:>+8.1f} {sds:>+8.1f}{pos}')
