import pandas as pd
df = pd.read_csv('pm_ranking/full_scored_export.csv', usecols=['event_ticker','predictor_name','model_prob_yes','yes_ask_prob','no_ask_prob','outcome'], low_memory=False)
df = df.dropna(subset=['model_prob_yes','yes_ask_prob','no_ask_prob','outcome'])
df = df[(df.yes_ask_prob>0)&(df.no_ask_prob>0)]
NEWER = ['grok-4.3','qwen-3.6-plus','minimax-m2.7','deepseek-v4-pro','gemini-3.1-pro',
 'gemini-3.5-flash','claude-opus-4.8-thinking','gpt-5.5-high','grok-4.20','claude-sonnet-4.6',
 'kimi-k2.6','glm-5.1','thinking-machines-zs-v2','kimi-k3','claude-fable-5','gpt-5.6-sol',
 'agent-gpt-5.5-high','agent-gemini-3.1-pro','agent-claude-opus-4.8-thinking','agent-claude-fable-5',
 'agent-gpt-5.6-sol','foresight-v3']
S = {m:set(df[df.predictor_name==m].event_ticker.unique()) for m in NEWER}
S = {m:s for m,s in S.items() if len(s)>0}
# greedy: start from the pair with largest overlap; add model that keeps intersection largest
import itertools
order = sorted(S, key=lambda m:-len(S[m]))
cur = set(S[order[0]]); seq=[(order[0], len(cur), len(cur))]
remaining = [m for m in order[1:]]
while remaining:
    best=None; bestn=-1
    for m in remaining:
        k=len(cur & S[m])
        if k>bestn: bestn=k; best=m
    cur = cur & S[best]; remaining.remove(best)
    seq.append((best, len(S[best]), len(cur)))
print(f'{\"# added\":>6s} {\"model\":40s} {\"own_n\":>6s} {\"intersection\":>12s}')
for i,(m,own,inter) in enumerate(seq):
    tag = ' [AGENT]' if m.startswith('agent') else ''
    print(f'{i:>6d} {m:40s} {own:>6d} {inter:>12d}{tag}')
