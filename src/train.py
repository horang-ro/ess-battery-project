"""
train.py — 데이터 분할 → 실험 → 최종 모델 → Batch 2 테스트(1회) → 성능 표 저장
(notebooks/02_modeling.ipynb Step 1~5 를 정리한 코드)

실행 : python src/train.py   (features.py 다음에 실행)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT         = Path(__file__).resolve().parents[1]
DATA_DIR     = ROOT / 'data'
RESULT_DIR   = ROOT / 'results'
RANDOM_STATE = 42          # 재현성을 위해 모든 무작위 값 고정
TARGET_MAPE  = 9.1         # 원논문 Regression 성능
CORE         = ['dq_logvar']
EXTRAS       = ['QD_slope', 'Cavg_80', 'Tavg_mean']


def mape(life_true, life_pred):
    return np.mean(np.abs(life_pred - life_true) / life_true) * 100


def split_data(feat):
    """Batch 1 → 학습 27 / 검증 9 (수명 3구간 층화), Batch 2 → 테스트 39."""
    b1 = feat[feat['batch'] == 'Batch1'].reset_index(drop=True)
    b2 = feat[feat['batch'] == 'Batch2'].reset_index(drop=True)
    life_bin = pd.qcut(b1['cycle_life'], q=3, labels=['short', 'mid', 'long'])
    train_df, valid_df = train_test_split(b1, test_size=9, stratify=life_bin, random_state=RANDOM_STATE)
    return b1, b2, train_df, valid_df


def evaluate(model, cols, name, train_df, valid_df, policy):
    """Train CV(같은 충전 방식 = 같은 묶음) 와 Valid MAPE. 테스트는 쓰지 않는다."""
    X_tr, y_tr = train_df[cols], train_df['log_life']
    groups  = train_df['cell_key'].map(policy)
    cv_pred = cross_val_predict(model, X_tr, y_tr, cv=GroupKFold(n_splits=5), groups=groups)
    model.fit(X_tr, y_tr)
    return {'실험': name, '피처': ', '.join(cols),
            'Train CV MAPE': round(mape(train_df['cycle_life'], 10 ** cv_pred), 1),
            'Valid MAPE'   : round(mape(valid_df['cycle_life'], 10 ** model.predict(valid_df[cols])), 1)}


def tune(estimator, grid, cols, train_df, policy):
    gs = GridSearchCV(make_pipeline(StandardScaler(), estimator), grid,
                      cv=GroupKFold(n_splits=5), scoring='neg_mean_absolute_error')
    gs.fit(train_df[cols], train_df['log_life'], groups=train_df['cell_key'].map(policy))
    return gs.best_estimator_


def run_experiments(train_df, valid_df, policy):
    linear = lambda: make_pipeline(StandardScaler(), LinearRegression())
    log = [evaluate(linear(), CORE, 'E1 기준선 선형회귀', train_df, valid_df, policy)]
    for extra in EXTRAS:                                        # 보조 피처 하나씩
        log.append(evaluate(linear(), CORE + [extra], f'E2 + {extra}', train_df, valid_df, policy))

    cols_all = CORE + EXTRAS                                    # 규제 모델 + 트리 모델
    models = {
        'E3 Ridge (피처 4개)'       : tune(Ridge(), {'ridge__alpha': np.logspace(-3, 3, 30)}, cols_all, train_df, policy),
        'E4 Lasso (피처 4개)'       : tune(Lasso(max_iter=10000), {'lasso__alpha': np.logspace(-4, 0, 30)}, cols_all, train_df, policy),
        'E5 RandomForest (피처 4개)': tune(RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
                                         {'randomforestregressor__max_depth': [2, 3, 4, None],
                                          'randomforestregressor__min_samples_leaf': [2, 4]}, cols_all, train_df, policy),
    }
    for name, model in models.items():
        log.append(evaluate(model, cols_all, name, train_df, valid_df, policy))
    return pd.DataFrame(log)


def test_final(b1, b2, train_m, valid_m):
    """최종 모델(E1)을 Batch 1 전체로 학습한 뒤 Batch 2 를 한 번만 예측한다."""
    final = make_pipeline(StandardScaler(), LinearRegression())
    final.fit(b1[CORE], b1['log_life'])
    pred = 10 ** final.predict(b2[CORE])

    test_m = round(mape(b2['cycle_life'], pred), 1)
    perf = pd.DataFrame({'MAPE (%)': [train_m, valid_m, test_m,
                                      round(valid_m - train_m, 1), round(test_m - valid_m, 1),
                                      round(test_m - TARGET_MAPE, 1)]},
                        index=['Train (Batch 1 CV)', 'Valid (Batch 1 Hold-out)', 'Test (Batch 2)',
                               'Gap (Train-Valid)', 'Gap (Valid-Test)', 'Gap (Target-Test)'])
    extra = {'RMSE': np.sqrt(mean_squared_error(b2['cycle_life'], pred)),
             'R2'  : r2_score(b2['cycle_life'], pred),
             'bias': np.mean((pred - b2['cycle_life']) / b2['cycle_life']) * 100}
    return perf, extra


def main():
    feat   = pd.read_pickle(DATA_DIR / 'features.pkl')
    policy = pd.read_pickle(DATA_DIR / 'policy_features.pkl').set_index('cell_key')['charging_policy']

    b1, b2, train_df, valid_df = split_data(feat)
    print(f"[split] train {len(train_df)} / valid {len(valid_df)} / test {len(b2)}")

    exp_log = run_experiments(train_df, valid_df, policy)
    print("\n[experiments]\n", exp_log.to_string(index=False))

    e1 = exp_log.iloc[0]
    perf, extra = test_final(b1, b2, e1['Train CV MAPE'], e1['Valid MAPE'])
    print("\n[performance]\n", perf.to_string())
    print(f"보조 지표 : RMSE {extra['RMSE']:.0f} 사이클, R² {extra['R2']:.2f}, 평균 오차 방향 {extra['bias']:+.1f}%")

    RESULT_DIR.mkdir(exist_ok=True)
    exp_log.to_csv(RESULT_DIR / 'experiment_log.csv', index=False)
    perf.to_csv(RESULT_DIR / 'model_performance.csv')


if __name__ == '__main__':
    main()
