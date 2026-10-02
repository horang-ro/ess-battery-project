"""
features.py — 배터리 1개 = 1행 피처 표 만들기
(notebooks/01_EDA.ipynb Q3·Q4 칸, notebooks/02_modeling.ipynb Step 0 을 정리한 코드)

실행 : python src/features.py   (preprocess.py 다음에 실행)
"""
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'

POLICY_PATTERN = re.compile(r'^([\d.]+)C\((\d+)%\)-([\d.]+)C')


def dq_features(clean, qdlin):
    """ΔQ(V) = 100 번째 − 10 번째 사이클 방전 곡선 (리스트 index 0 = 1 번째 사이클)."""
    rows = []
    for _, r in clean.iterrows():
        L  = qdlin[r['cell_key']]
        dq = np.asarray(L[99]) - np.asarray(L[9])
        rows.append({'cell_key': r['cell_key'], 'batch': r['batch'], 'cycle_life': r['cycle_life'],
                     'dq_min': dq.min(), 'dq_mean': dq.mean(),
                     'dq_var': dq.var(), 'dq_logvar': np.log10(dq.var())})
    return pd.DataFrame(rows)


def parse_policy(p):
    """충전 방식 글자 → C1(1단계 속도), Q1(전환 %), C2(2단계 속도), Cavg_80(0→80% 평균 속도)."""
    m = POLICY_PATTERN.match(p)
    C1, Q1, C2 = float(m[1]), float(m[2]), float(m[3])
    t80 = Q1 / 100 / C1 + max(80 - Q1, 0) / 100 / C2        # 0→80% 충전 시간(시간)
    return pd.Series({'C1': C1, 'Q1': Q1, 'C2': C2, 'Cavg_80': 0.8 / t80})


def policy_features(clean):
    pol = clean[['cell_key', 'batch', 'cycle_life', 'charging_policy']].copy()
    return pol.join(pol['charging_policy'].apply(parse_policy))


def early_feats(g):
    """2~100 번째 사이클 요약. IR 조건은 IR 피처에만 적용 (IR 미측정 배터리도 빠지지 않게)."""
    g  = g.sort_values('cycle')
    ir = g[g['IR'] > 0]
    return pd.Series({
        'QD_2'          : g['QD'].iloc[0],
        'QD_diff'       : g['QD'].iloc[-1] - g['QD'].iloc[0],
        'QD_slope'      : np.polyfit(g['cycle'], g['QD'], 1)[0],
        'IR_2'          : ir['IR'].iloc[0] if len(ir) else np.nan,
        'IR_diff'       : ir['IR'].iloc[-1] - ir['IR'].iloc[0] if len(ir) else np.nan,
        'Tavg_mean'     : g['Tavg'].mean(),
        'Tmax_mean'     : g['Tmax'].mean(),
        'chargetime_med': g['chargetime'].iloc[:5].median(),
    })


def build_features(clean, summary_df, dq_df, pol):
    early = summary_df[summary_df['cycle'].between(2, 100) & summary_df['QD'].between(0.7, 1.2)]
    ef    = early.groupby('cell_key').apply(early_feats).reset_index()
    feat  = (clean[['cell_key', 'batch', 'cycle_life']]
             .merge(ef, on='cell_key')
             .merge(dq_df[['cell_key', 'dq_min', 'dq_mean', 'dq_var', 'dq_logvar']], on='cell_key')
             .merge(pol[['cell_key', 'C1', 'Q1', 'C2', 'Cavg_80']], on='cell_key'))
    feat['log_life'] = np.log10(feat['cycle_life'])

    # 개수 확인 : 정리 후 배치별 배터리 수와 같아야 한다
    assert feat.groupby('batch').size().to_dict() == clean.groupby('batch').size().to_dict(), "배터리 수가 다름!"
    return feat


def main():
    clean      = pd.read_pickle(DATA_DIR / 'cells_clean.pkl')
    summary_df = pd.read_pickle(DATA_DIR / 'summary_df.pkl')
    with open(DATA_DIR / 'qdlin_early.pkl', 'rb') as f:
        qdlin = pickle.load(f)

    dq_df = dq_features(clean, qdlin)
    pol   = policy_features(clean)
    feat  = build_features(clean, summary_df, dq_df, pol)

    dq_df.to_pickle(DATA_DIR / 'dq_features.pkl')
    pol.to_pickle(DATA_DIR / 'policy_features.pkl')
    feat.to_pickle(DATA_DIR / 'features.pkl')
    print("[features] 피처 표 :", feat.groupby('batch').size().to_dict())
    print("[features] IR 비어 있는 배터리 :", feat.loc[feat['IR_2'].isna(), 'cell_key'].tolist())


if __name__ == '__main__':
    main()
