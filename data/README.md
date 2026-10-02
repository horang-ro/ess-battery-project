# data

원본 데이터는 용량이 커서(약 7.8GB) GitHub에 올리지 않았다.

## 1. 원본 받기
- 출처 : MIT-Stanford Battery Dataset (Severson et al., Nature Energy 2019)
- 다운로드 : https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle
- 압축을 풀어 아래 3개 파일을 `data/archive/`에 넣는다.

| 파일 | 배치 | 용도 |
|---|---|---|
| `2017-05-12_batchdata_updated_struct_errorcorrect.mat` | Batch 1 | 학습 |
| `2018-02-20_batchdata_updated_struct_errorcorrect.mat` | Batch 2 | 테스트 |
| `2018-04-12_batchdata_updated_struct_errorcorrect.mat` | Batch 3 | EDA 비교용 (모델 평가에는 사용 안 함) |

## 2. 정리 파일 만들기
`python src/preprocess.py`와 `python src/features.py`를 실행하면 아래 파일이 이 폴더에 만들어진다. (노트북 `01_EDA.ipynb`로도 만들 수 있다.)

| 파일 | 내용 |
|---|---|
| `cells_df.pkl` | 배터리별 수명, 충전 방식 |
| `summary_df.pkl` | 사이클별 요약 기록 (방전 용량, 내부 저항, 온도, 충전 시간) |
| `qdlin_early.pkl` | 0~100번째 사이클의 방전 곡선(Qdlin) |
| `cells_clean.pkl` | 정답을 믿을 수 없는 20개를 뺀 배터리 목록 (119개) |
| `dq_features.pkl`, `policy_features.pkl` | ΔQ 피처, 충전 방식 숫자 피처 |
| `features.pkl` | 모델 입력 피처 표 (`src/features.py` 또는 `02_modeling` Step 0) |
