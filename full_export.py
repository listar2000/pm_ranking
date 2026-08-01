"""Full scored export: every predictor x every settled event -> analysis_export
format (one row per prediction-outcome). Faithfully replicates
scripts/export_analysis_csv.build_outcome_rows (median-round dedup, latest market
snapshot <= prediction snapshot_time, one row per outcome) but pre-indexes market
snapshots by event so the full 267K-prediction export runs in minutes, not hours.

Step 1 (--pull): dump raw DB tables to pickle (repeatable, no re-query).
Step 2 (--join): build the scored CSV from the pickles.
"""
import os, sys, json, argparse, pickle, time
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

CACHE = os.path.join(os.path.dirname(__file__), 'db_cache')
os.makedirs(CACHE, exist_ok=True)
OUT = os.path.join(os.path.dirname(__file__), 'full_scored_export.csv')
DB = os.getenv('DATABASE_URL')


def engine():
    return create_engine(DB, connect_args={'options': '-c statement_timeout=1200000'})


def pull(after='2025-01-01', before='2027-01-01'):
    eng = engine()
    with eng.connect() as c:
        print('pull predictions...'); t = time.time()
        preds = pd.read_sql(text("""
            SELECT id AS prediction_id, event_ticker, predictor_name, prediction,
                   submission_id, created_at AS prediction_created_at
            FROM prediction
            WHERE created_at > :a AND created_at < :b
        """), c, params={'a': after, 'b': before})
        print(f'  {len(preds)} rows in {time.time()-t:.0f}s')
        preds.to_pickle(f'{CACHE}/predictions.pkl')

        print('pull events (settled)...'); t = time.time()
        events = pd.read_sql(text("""
            SELECT event_ticker, title AS event_title, category, close_time,
                   market_outcome, markets AS markets_json
            FROM event WHERE market_outcome IS NOT NULL
        """), c)
        print(f'  {len(events)} rows in {time.time()-t:.0f}s')
        events.to_pickle(f'{CACHE}/events.pkl')

        print('pull submissions...'); t = time.time()
        subs = pd.read_sql(text("SELECT id AS submission_id, snapshot_time FROM user_submission"), c)
        print(f'  {len(subs)} rows in {time.time()-t:.0f}s')
        subs.to_pickle(f'{CACHE}/submissions.pkl')

        print('pull markets (chunked by event)...'); t = time.time()
        tickers = sorted(events['event_ticker'].astype(str).unique())
        chunks, BATCH = [], 300
        for i in range(0, len(tickers), BATCH):
            batch = tuple(tickers[i:i+BATCH])
            clause = f"event_ticker = '{batch[0]}'" if len(batch) == 1 else f"event_ticker IN {batch}"
            chunks.append(pd.read_sql(text(f"""
                SELECT event_ticker, market_title, created_at AS market_snapshot_time,
                       yes_ask, yes_bid, no_ask, no_bid, liquidity, volume, last_price, other_info
                FROM market WHERE {clause}
            """), c))
            print(f'  markets {min(i+BATCH,len(tickers))}/{len(tickers)}', end='\r')
        markets = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        print(f'\n  {len(markets)} rows in {time.time()-t:.0f}s')
        markets.to_pickle(f'{CACHE}/markets.pkl')
    print('pull done.')


def parse_json(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (dict, list)):
        return v
    try:
        return json.loads(v)
    except Exception:
        return None


def join():
    preds = pd.read_pickle(f'{CACHE}/predictions.pkl')
    events = pd.read_pickle(f'{CACHE}/events.pkl')
    subs = pd.read_pickle(f'{CACHE}/submissions.pkl')
    markets = pd.read_pickle(f'{CACHE}/markets.pkl')
    print(f'loaded preds={len(preds)} events={len(events)} subs={len(subs)} markets={len(markets)}')

    for df, cols in [(subs, ['snapshot_time']), (markets, ['market_snapshot_time'])]:
        for col in cols:
            df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
    events['close_time'] = pd.to_datetime(events['close_time'], errors='coerce', utc=True)

    valid = set(events['event_ticker'].unique())
    preds = preds[preds['event_ticker'].isin(valid)].copy()
    preds['submission_id'] = preds['submission_id'].astype(str)
    subs['submission_id'] = subs['submission_id'].astype(str)
    preds = preds.merge(subs, on='submission_id', how='inner')
    print(f'after submission join: {len(preds)}')

    # round = submission_count per (event, predictor) by snapshot_time; dedup to median round
    preds = preds.sort_values(['event_ticker', 'predictor_name', 'snapshot_time'])
    preds['submission_count'] = preds.groupby(['event_ticker', 'predictor_name']).cumcount()
    med = (preds.groupby(['predictor_name', 'event_ticker'])['submission_count'].median()
           .reset_index().rename(columns={'submission_count': 'med'}))
    preds = preds.merge(med, on=['predictor_name', 'event_ticker'])
    preds['dist'] = (preds['submission_count'] - preds['med']).abs()
    preds = (preds.sort_values('dist')
             .drop_duplicates(subset=['predictor_name', 'event_ticker'], keep='first')
             .drop(columns=['med', 'dist']))
    print(f'after median-round dedup: {len(preds)} (1 per predictor-event)')

    # Pre-index market snapshots by int64 ns (UTC) to avoid tz-compare issues.
    markets = markets[markets['market_snapshot_time'].notna()].copy()
    markets['snap_ns'] = markets['market_snapshot_time'].astype('int64')
    markets = markets.sort_values(['event_ticker', 'snap_ns'])
    ev_snaptimes = {}   # event -> np.array of sorted unique snapshot ns
    price_map = {}      # (event, snap_ns) -> {market_title: price row}
    for ev, g in markets.groupby('event_ticker'):
        ev_snaptimes[ev] = np.sort(g['snap_ns'].unique())
        for st, gg in g.groupby('snap_ns'):
            d = {}
            for _, m in gg.iterrows():
                if m['market_title'] not in d:
                    d[m['market_title']] = m
            price_map[(ev, int(st))] = d
    print(f'indexed markets for {len(ev_snaptimes)} events')

    events_map = events.set_index('event_ticker').to_dict('index')
    rows = []
    n = 0
    for pred in preds.itertuples(index=False):
        n += 1
        if n % 20000 == 0:
            print(f'  rows built from {n}/{len(preds)} preds -> {len(rows)}', end='\r')
        ticker = pred.event_ticker
        ev = events_map.get(ticker)
        if ev is None:
            continue
        pj = parse_json(pred.prediction)
        mo = parse_json(ev['market_outcome'])
        if not pj or not mo:
            continue
        probs = pj.get('probabilities', []) if isinstance(pj, dict) else []
        if isinstance(probs, list):
            model_probs = {p['market']: p['probability'] for p in probs if isinstance(p, dict) and 'market' in p and 'probability' in p}
        elif isinstance(probs, dict):
            model_probs = {str(k): float(v) for k, v in probs.items()}
        else:
            model_probs = {}
        if not model_probs and isinstance(pj, dict) and 'probabilities' not in pj:
            model_probs = {str(k): float(v) for k, v in pj.items() if isinstance(v, (int, float))}
        if not model_probs:
            continue

        num_outcomes = len(mo)
        event_type = 'binary' if num_outcomes == 2 else ('single' if num_outcomes == 1 else 'multi')
        st = pred.snapshot_time
        times = ev_snaptimes.get(ticker)
        if times is None or len(times) == 0:
            continue
        if pd.notna(st):
            st_ns = pd.Timestamp(st).value
            le = times[times <= st_ns]
            best = le.max() if len(le) else times.max()
        else:
            best = times.max()
        pm = price_map.get((ticker, int(best)), {})

        for outcome_name, outcome_val in mo.items():
            model_p = model_probs.get(outcome_name)
            if model_p is None:
                continue
            m = pm.get(outcome_name)
            if m is None:
                ya = yb = na = nb = liq = vol = lp = None; mt = None
            else:
                ya, yb, na, nb = m['yes_ask'], m['yes_bid'], m['no_ask'], m['no_bid']
                liq, vol, lp = m['liquidity'], m['volume'], m['last_price']
                oi = parse_json(m['other_info']); mt = oi.get('ticker') if isinstance(oi, dict) else None
            rows.append((ticker, ev.get('event_title'), ev.get('category'), ev.get('close_time'),
                         num_outcomes, event_type, outcome_name, mt, outcome_val,
                         pred.predictor_name, str(pred.prediction_id), round(float(model_p), 6),
                         ya, yb, na, nb, liq, vol, lp, st, pred.prediction_created_at, pred.submission_count))
    cols = ['event_ticker', 'event_title', 'category', 'close_time', 'num_outcomes', 'event_type',
            'market_title', 'market_ticker', 'outcome', 'predictor_name', 'prediction_id',
            'model_prob_yes', 'yes_ask', 'yes_bid', 'no_ask', 'no_bid', 'liquidity', 'volume',
            'last_price', 'snapshot_time', 'prediction_created_at', 'round']
    out = pd.DataFrame(rows, columns=cols)
    out['yes_ask_prob'] = out['yes_ask'] / 100.0
    out['no_ask_prob'] = out['no_ask'] / 100.0
    out['yes_bid_prob'] = out['yes_bid'] / 100.0
    out['no_bid_prob'] = out['no_bid'] / 100.0
    out = out.sort_values(['event_ticker', 'predictor_name', 'market_title'])
    out.to_csv(OUT, index=False)
    print(f'\nWrote {len(out)} rows, {out.event_ticker.nunique()} events, '
          f'{out.predictor_name.nunique()} predictors -> {OUT}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pull', action='store_true')
    ap.add_argument('--join', action='store_true')
    ap.add_argument('--after', default='2025-01-01')
    ap.add_argument('--before', default='2027-01-01')
    a = ap.parse_args()
    if a.pull:
        pull(a.after, a.before)
    if a.join:
        join()
