"""
preprocess.py — 원본 .mat 로딩 + 정답(수명)을 믿을 수 없는 배터리 제외
(notebooks/01_EDA.ipynb 의 '배치 비교' 로딩 칸과 Q2-2 칸을 정리한 코드)

실행 : python src/preprocess.py            # 정리 파일이 있으면 로딩은 건너뜀
       python src/preprocess.py --reload   # 원본 .mat 부터 다시 읽음 (수 분 소요)
"""
import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import pickle

ROOT     = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
RAW_DIR  = DATA_DIR / 'archive'

BATCH_FILES = {
    'Batch1': '2017-05-12_batchdata_updated_struct_errorcorrect.mat',
    'Batch2': '2018-02-20_batchdata_updated_struct_errorcorrect.mat',
    'Batch3': '2018-04-12_batchdata_updated_struct_errorcorrect.mat',
}
N_KEEP     = 101    # Qdlin 은 사이클 0~100 번만 보관 (101 번째 이후는 미래 정보)
EOL_CUTOFF = 0.9    # 마지막 용량이 이보다 높으면 수명 끝(0.88Ah) 전에 실험이 멈춘 배터리


# ---------------------------------------------------------------- 1. 원본 로딩
def load_mat(path):
    """MATLAB v7.3(HDF5) 은 mat73, 그 이하는 scipy 로 읽는다."""
    try:
        import mat73
        return mat73.loadmat(str(path))
    except Exception:
        import scipy.io as sio
        return sio.loadmat(str(path), simplify_cells=True)


def to_list_of_dicts(d):
    """mat73 의 dict-of-lists 구조를 list-of-dicts 로 바꾼다."""
    if isinstance(d, dict):
        keys = list(d.keys())
        return [{k: d[k][i] for k in keys} for i in range(len(d[keys[0]]))]
    return d


def load_batch(name, fname):
    """배치 파일 1개 → (배터리 표, 사이클 요약 표, 초기 Qdlin). 원본은 바로 메모리에서 지운다."""
    mat   = load_mat(RAW_DIR / fname)
    cells = to_list_of_dicts(mat['batch'])

    cell_rows, summary_parts, qdlin = [], [], {}
    for i, cell in enumerate(cells):
        key = f"{name}_{i}"
        cl  = cell['cycle_life']
        cl  = float(np.asarray(cl).squeeze()) if cl is not None else np.nan
        policy = str(cell.get('policy_readable') or cell.get('policy') or 'unknown')
        cell_rows.append({'batch': name, 'cell_key': key, 'cell_id': i,
                          'cycle_life': cl, 'charging_policy': policy})

        s = cell['summary']
        n = len(s['QDischarge'])
        part = pd.DataFrame({
            'cycle': np.arange(1, n + 1),
            'QD': s['QDischarge'], 'QC': s['QCharge'], 'IR': s['IR'],
            'Tmax': s['Tmax'], 'Tavg': s['Tavg'], 'Tmin': s['Tmin'],
            'chargetime': s['chargetime'],
        })
        part.insert(0, 'cell_key', key)
        part.insert(0, 'batch', name)
        summary_parts.append(part)

        cyc = cell['cycles']
        qdlin[key] = (list(cyc['Qdlin'][:N_KEEP]) if isinstance(cyc, dict)
                      else [c['Qdlin'] for c in cyc[:N_KEEP]])

    del mat, cells
    gc.collect()
    return pd.DataFrame(cell_rows), pd.concat(summary_parts, ignore_index=True), qdlin


def load_all():
    cells_list, summary_list, qdlin_all = [], [], {}
    for name, fname in BATCH_FILES.items():
        print(f"[load] {name} ...")
        c, s, q = load_batch(name, fname)
        cells_list.append(c); summary_list.append(s); qdlin_all.update(q)
    cells_df   = pd.concat(cells_list, ignore_index=True)
    summary_df = pd.concat(summary_list, ignore_index=True)
    cells_df.to_pickle(DATA_DIR / 'cells_df.pkl')
    summary_df.to_pickle(DATA_DIR / 'summary_df.pkl')
    with open(DATA_DIR / 'qdlin_early.pkl', 'wb') as f:
        pickle.dump(qdlin_all, f)


# ---------------------------------------------------------------- 2. 배터리 정리
def clean_cells(cells_df, summary_df):
    """
    - 수명 값이 없는 배터리 제외
    - 기록 마지막 5사이클 용량 중앙값이 EOL_CUTOFF 보다 높은 배터리 제외
      (수명 끝 0.88Ah 에 닿기 전에 실험이 멈춰, 수명 값이 실제 수명이 아님)
    """
    valid   = cells_df.dropna(subset=['cycle_life'])
    qd_ok   = summary_df[summary_df['QD'].between(0.7, 1.2)]          # 첫 사이클 0값·튀는 값 제외
    last_qd = (qd_ok.sort_values('cycle').groupby('cell_key').tail(5)
                    .groupby('cell_key')['QD'].median().rename('last_QD'))
    valid   = valid.join(last_qd, on='cell_key')
    return valid[valid['last_QD'] <= EOL_CUTOFF].copy()


def main(reload=False):
    raw_ready = all((DATA_DIR / f).exists() for f in ['cells_df.pkl', 'summary_df.pkl', 'qdlin_early.pkl'])
    if reload or not raw_ready:
        load_all()
    else:
        print("[load] 정리 파일이 있어 원본 로딩을 건너뜀 (--reload 로 다시 읽기)")

    cells_df   = pd.read_pickle(DATA_DIR / 'cells_df.pkl')
    summary_df = pd.read_pickle(DATA_DIR / 'summary_df.pkl')
    clean      = clean_cells(cells_df, summary_df)
    clean.to_pickle(DATA_DIR / 'cells_clean.pkl')

    print("[clean] 원본  :", cells_df.groupby('batch').size().to_dict())
    print("[clean] 정리 후:", clean.groupby('batch').size().to_dict())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--reload', action='store_true', help='원본 .mat 부터 다시 읽기')
    main(ap.parse_args().reload)
