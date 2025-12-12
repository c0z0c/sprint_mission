# -*- coding: utf-8 -*-
"""MNIST 예측 히스토리 저장소 API - 클래스 기반 설계

이 모듈은 예측 기록을 저장, 조회, 관리하는 기능을 제공합니다.
"""

import datetime
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from helper_dev_utils import get_auto_logger

logger = get_auto_logger(log_level=logging.DEBUG)
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.history.HistoryRecord import HistoryRecord
from src.history.HistoryManager import HistoryManager
from src.history.FileHistoryManager import FileHistoryManager

logger = get_auto_logger(log_level=logging.DEBUG)
# ============================================================================
# 더미 데이터 생성
# ============================================================================


def generate_dummy_history_data(count: int = 5) -> List[Dict]:
    """더미 히스토리 데이터를 생성합니다.

    Args:
        count: 생성할 레코드 개수

    Returns:
        더미 히스토리 레코드 리스트
    """
    dummy_records = []

    for i in range(count):
        # 랜덤 확률 생성
        logits = np.random.randn(10)
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)

        predicted_label = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_label])

        # 더미 이미지
        canvas_image = np.random.randint(0, 255, (200, 200, 4), dtype=np.uint8)
        preprocessed_image = np.random.randint(0, 255, (28, 28), dtype=np.uint8)

        # 타임스탬프 (현재부터 i분 전)
        timestamp = (datetime.datetime.now() - datetime.timedelta(minutes=i)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        dummy_records.append(
            {
                "id": i + 1,
                "canvas_image": canvas_image,
                "preprocessed_image": preprocessed_image,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "probabilities": probabilities,
                "timestamp": timestamp,
                "notes": f"테스트 레코드 #{i + 1}" if i % 2 == 0 else None,
            }
        )

    return dummy_records


# ============================================================================
# 사용 예시 및 테스트
# ============================================================================


def main():
    """사용 예시"""
    logger.debug("=" * 60)
    logger.debug("히스토리 매니저 API 테스트")
    logger.debug("=" * 60)

    # 1. 메모리 기반 히스토리 매니저
    logger.debug("\n[1] 메모리 기반 히스토리 매니저 테스트")
    manager = HistoryManager(max_records=10)

    # 더미 데이터 추가
    for i in range(5):
        logits = np.random.randn(10)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        label = int(np.argmax(probs))
        conf = float(probs[label])

        canvas = np.random.randint(0, 255, (200, 200, 4), dtype=np.uint8)
        preprocessed = np.random.randint(0, 255, (28, 28), dtype=np.uint8)

        manager.add_record(canvas, preprocessed, label, conf, probs)

    logger.debug(f"총 레코드 수: {len(manager)}")
    logger.debug(f"매니저 정보: {manager}")

    # 2. 통계 조회
    logger.debug("\n[2] 통계 조회")
    stats = manager.get_statistics()
    logger.debug(f"통계: {stats}")

    # 3. 레이블별 조회
    logger.debug("\n[3] 레이블별 조회")
    for label in range(10):
        records = manager.get_records_by_label(label)
        if records:
            logger.debug(f"레이블 {label}: {len(records)}개")

    # 4. 신뢰도 범위별 조회
    logger.debug("\n[4] 신뢰도 범위별 조회 (0.5 이상)")
    high_conf_records = manager.get_records_by_confidence_range(min_confidence=0.5)
    logger.debug(f"신뢰도 0.5 이상: {len(high_conf_records)}개")

    # 5. 파일 기반 히스토리 매니저
    logger.debug("\n[5] 파일 기반 히스토리 매니저 테스트")
    file_manager = FileHistoryManager(save_dir="./test_history", max_records=10)

    # 더미 데이터 추가
    canvas = np.random.randint(0, 255, (200, 200, 4), dtype=np.uint8)
    preprocessed = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
    logits = np.random.randn(10)
    probs = np.exp(logits) / np.sum(np.exp(logits))

    record = file_manager.add_record(
        canvas,
        preprocessed,
        predicted_label=7,
        confidence=0.95,
        probabilities=probs,
        notes="파일 저장 테스트",
    )

    logger.debug(f"레코드 저장 완료: ID={record.record_id}")
    logger.debug(f"저장 디렉토리: {file_manager.save_dir}")

    # 6. 더미 히스토리 데이터 생성
    logger.debug("\n[6] 더미 히스토리 데이터 생성")
    dummy_data = generate_dummy_history_data(count=3)
    logger.debug(f"더미 데이터 {len(dummy_data)}개 생성 완료")

    for data in dummy_data:
        logger.debug(
            f"  - ID={data['id']}, 레이블={data['predicted_label']}, "
            f"신뢰도={data['confidence']:.2%}"
        )

    logger.debug("\n" + "=" * 60)
    logger.debug("테스트 완료!")
    logger.debug("=" * 60)


if __name__ == "__main__":
    main()
