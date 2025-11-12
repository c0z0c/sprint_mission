"""
리더보드 재출용 script.py
운수종사자_분석_B.ipynb에서 학습한 모델로 test 데이터 예측 및 제출 파일 생성
"""

import os
import json
import joblib
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb

# 경고 무시
warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def apply_domain_weights_A(df):
    """A검사 도메인 가중치 적용"""
    weights = {
        "A1": 0.15, "A2": 0.15, "A3": 0.15,
        "A4": 0.10, "A5": 0.10, "A6": 0.08,
        "A7": 0.08, "A8": 0.05, "A9": 0.14
    }
    
    df_weighted = pd.DataFrame(index=df.index)
    for col in df.columns:
        for group, weight in weights.items():
            if col.startswith(group):
                df_weighted[f"{col}_w"] = df[col] * weight
                break
    return df_weighted


def apply_domain_weights_B(df):
    """B검사 도메인 가중치 적용"""
    weights = {
        "B1": 0.15, "B2": 0.15, "B3": 0.10,
        "B4": 0.15, "B5": 0.15, "B6": 0.05,
        "B7": 0.10, "B8": 0.08, "B9": 0.07
    }
    
    df_weighted = pd.DataFrame(index=df.index)
    for col in df.columns:
        for group, weight in weights.items():
            if col.startswith(group):
                df_weighted[f"{col}_w"] = df[col] * weight
                break
    return df_weighted


def load_and_predict(test_jsonl_path, model_path, apply_weights_func, test_type):
    """
    테스트 데이터를 로드하고 모델로 예측
    
    Args:
        test_jsonl_path: 테스트 데이터 경로
        model_path: 모델 파일 경로
        apply_weights_func: 도메인 가중치 적용 함수
        test_type: "A" 또는 "B"
    
    Returns:
        test_ids, predictions
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Test {test_type} 예측")
    logger.info(f"{'='*80}")
    
    # 모델 로드
    if not Path(model_path).exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    checkpoint = joblib.load(model_path)
    model = checkpoint['model']
    features = model.feature_name()
    best_iteration = checkpoint.get('best_iteration', None)
    
    logger.info(f"✓ 모델 로드: {Path(model_path).name}")
    logger.info(f"  피처 개수: {len(features)}")
    logger.info(f"  Best Iteration: {best_iteration}")
    logger.info(f"  Best AUC: {checkpoint.get('best_score', 'N/A')}")
    
    # 데이터 로드
    if not Path(test_jsonl_path).exists():
        raise FileNotFoundError(f"테스트 데이터를 찾을 수 없습니다: {test_jsonl_path}")
    
    df_test = pd.read_json(test_jsonl_path, lines=True)
    test_ids = df_test['Test_id_idx'].values
    
    logger.info(f"✓ 테스트 데이터 로드: {len(df_test):,}개")
    
    # 전처리
    exclude_cols = ['Test_id_idx', 'PrimaryKey', 'Test', 'TestDate']
    df_clean = df_test.drop(columns=[c for c in exclude_cols if c in df_test.columns])
    
    # 범주형 변수 인코딩
    for col in df_clean.select_dtypes(include=['object']).columns:
        df_clean[col] = pd.Categorical(df_clean[col]).codes
    df_clean = df_clean.fillna(0)
    
    # 도메인 가중치 적용
    X_test = apply_weights_func(df_clean)
    
    # 피처 정렬 (모델 학습 시와 동일한 순서)
    for feat in features:
        if feat not in X_test.columns:
            X_test[feat] = 0.0
    X_test = X_test[features]
    
    logger.info(f"✓ 전처리 완료: {X_test.shape}")
    
    # 예측
    if best_iteration is not None:
        y_pred = model.predict(X_test.values, num_iteration=best_iteration)
    else:
        y_pred = model.predict(X_test.values)
    
    logger.info(f"✓ 예측 완료")
    logger.info(f"  확률 범위: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    logger.info(f"  확률 평균: {y_pred.mean():.4f}")
    logger.info(f"  확률 중앙값: {np.median(y_pred):.4f}")
    
    return test_ids, y_pred


def find_file(base_patterns, filename):
    """여러 경로 패턴에서 파일을 찾음"""
    for pattern in base_patterns:
        p = Path(pattern) / filename
        if p.exists():
            logger.info(f"  ✓ 파일 발견: {p}")
            return p
    raise FileNotFoundError(f"{filename} 파일을 찾을 수 없습니다. 시도한 경로: {[str(Path(p)/filename) for p in base_patterns]}")


def main():
    """메인 실행 함수"""
    # ========================================================================
    # 경로 설정 (멀티 경로 탐색)
    # ========================================================================
    # 우선순위별 경로 패턴
    BASE_PATTERNS = [
        "./data/test",      # 대회 기본: ./data/test/A.csv
        "./test",           # 간소화 버전
        "./data",           # 로컬 테스트용
        "../data/test",     # 상위 디렉토리
        "../test",
        "."                 # 현재 디렉토리
    ]
    
    MODEL_DIR = Path("./model")
    OUT_DIR   = Path("./output")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*80}")
    logger.info("리더보드 제출 파일 생성 (test.csv 기반 A/B 분리)")
    logger.info(f"{'='*80}")
    logger.info(f"모델 디렉토리: {MODEL_DIR}")
    logger.info(f"출력 디렉토리: {OUT_DIR}")
    
    # ========================================================================
    # 파일 로드 (멀티 경로 자동 탐색)
    # ========================================================================
    logger.info("\n파일 탐색 중...")
    
    # test.csv 로드 (메타데이터: Test_id + Test 컬럼)
    test_csv = None
    for pattern in ["./data", "./data/test", "./test", ".", "../data"]:
        p = Path(pattern) / "test.csv"
        if p.exists():
            test_csv = p
            logger.info(f"  ✓ test.csv: {test_csv}")
            break
    
    if test_csv is None:
        raise FileNotFoundError("test.csv 파일을 찾을 수 없습니다.")
    
    # A.csv, B.csv 로드
    A_csv = find_file(BASE_PATTERNS, "A.csv")
    B_csv = find_file(BASE_PATTERNS, "B.csv")
    
    # sample_submission.csv 로드 (순서 기준)
    sample_csv = None
    for pattern in ["./data", "./data/test", "./test", ".", "../data"]:
        p = Path(pattern) / "sample_submission.csv"
        if p.exists():
            sample_csv = p
            logger.info(f"  ✓ sample_submission.csv: {sample_csv}")
            break
    
    # 모델 파일
    model_A_path = MODEL_DIR / "lgbm_A.pkl"
    model_B_path = MODEL_DIR / "lgbm_B.pkl"
    
    if not model_A_path.exists():
        raise FileNotFoundError(f"모델 파일 없음: {model_A_path}")
    if not model_B_path.exists():
        raise FileNotFoundError(f"모델 파일 없음: {model_B_path}")
    
    logger.info(f"  ✓ 모델 A: {model_A_path}")
    logger.info(f"  ✓ 모델 B: {model_B_path}")
    
    # ========================================================================
    # 데이터 로드
    # ========================================================================
    logger.info("\n데이터 로드 중...")
    meta = pd.read_csv(test_csv)
    Araw = pd.read_csv(A_csv)
    Braw = pd.read_csv(B_csv)
    logger.info(f"  meta: {len(meta):,}개")
    logger.info(f"  Araw: {len(Araw):,}개")
    logger.info(f"  Braw: {len(Braw):,}개")
    
    # test.csv에서 A/B 분리
    A_meta = meta.loc[meta["Test"] == "A", ["Test_id", "Test"]].merge(Araw, on="Test_id", how="left")
    B_meta = meta.loc[meta["Test"] == "B", ["Test_id", "Test"]].merge(Braw, on="Test_id", how="left")
    logger.info(f"  A 매핑: {len(A_meta):,}개")
    logger.info(f"  B 매핑: {len(B_meta):,}개")
    
    # ========================================================================
    # 예측
    # ========================================================================
    logger.info("\n예측 실행 중...")
    
    # 모델 로드
    checkpoint_A = joblib.load(model_A_path)
    checkpoint_B = joblib.load(model_B_path)
    model_A = checkpoint_A['model']
    model_B = checkpoint_B['model']
    features_A = model_A.feature_name()
    features_B = model_B.feature_name()
    best_iter_A = checkpoint_A.get('best_iteration', None)
    best_iter_B = checkpoint_B.get('best_iteration', None)
    
    logger.info(f"  모델 A: {len(features_A)}개 피처")
    logger.info(f"  모델 B: {len(features_B)}개 피처")
    
    # A 예측
    test_ids_A, y_pred_A = predict_from_df(
        A_meta, model_A, features_A, apply_domain_weights_A, best_iter_A
    )
    logger.info(f"  ✓ A 예측 완료: {len(y_pred_A):,}개 (평균 {y_pred_A.mean():.4f})")
    
    # B 예측
    test_ids_B, y_pred_B = predict_from_df(
        B_meta, model_B, features_B, apply_domain_weights_B, best_iter_B
    )
    logger.info(f"  ✓ B 예측 완료: {len(y_pred_B):,}개 (평균 {y_pred_B.mean():.4f})")
    
    # ========================================================================
    # 제출 파일 생성
    # ========================================================================
    logger.info("\n제출 파일 생성 중...")
    
    # A+B 확률 병합
    probs = pd.concat([
        pd.DataFrame({"Test_id": test_ids_A, "prob": y_pred_A}),
        pd.DataFrame({"Test_id": test_ids_B, "prob": y_pred_B})
    ], axis=0, ignore_index=True)
    
    # sample_submission 기준으로 순서 맞춤
    if sample_csv is not None:
        sample = pd.read_csv(sample_csv)
        out = sample.merge(probs, on="Test_id", how="left")
        out["Label"] = out["prob"].astype(float).fillna(0.0)
        out = out.drop(columns=["prob"])
    else:
        # sample이 없으면 meta 순서 기준
        out = meta[["Test_id"]].merge(probs, on="Test_id", how="left")
        out["Label"] = out["prob"].astype(float).fillna(0.0)
        out = out.drop(columns=["prob"])
    
    output_path = OUT_DIR / "submission.csv"
    out.to_csv(output_path, index=False)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✓ 제출 파일 저장 완료")
    logger.info(f"{'='*80}")
    logger.info(f"파일: {output_path}")
    logger.info(f"크기: {os.path.getsize(output_path) / 1024:.2f} KB")
    logger.info(f"행 개수: {len(out):,}개")
    logger.info(f"Label 평균: {out['Label'].mean():.4f}")
    logger.info(f"{'='*80}")
    
    return output_path


if __name__ == "__main__":
    try:
        output_path = main()
        logger.info(f"\n성공적으로 완료되었습니다: {output_path}")
    except Exception as e:
        logger.error(f"\n오류 발생: {e}", exc_info=True)
        raise
