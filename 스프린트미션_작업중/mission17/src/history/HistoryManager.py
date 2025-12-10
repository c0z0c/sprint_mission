# -*- coding: utf-8 -*-
"""MNIST 예측 히스토리 저장소 API - 클래스 기반 설계

이 모듈은 예측 기록을 저장, 조회, 관리하는 기능을 제공합니다.
"""

import datetime
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from helper_dev_utils import get_auto_logger
logger = get_auto_logger()

from .HistoryRecord import HistoryRecord

# ============================================================================
# 히스토리 매니저 클래스
# ============================================================================

class HistoryManager:
    """예측 히스토리를 관리하는 클래스

    메모리 기반 히스토리 저장 및 관리 기능을 제공합니다.
    """

    def __init__(self, max_records: int = 100):
        """
        Args:
            max_records: 최대 저장 가능한 레코드 수 (초과 시 오래된 것부터 삭제)
        """
        self.max_records = max_records
        self._records: List[HistoryRecord] = []
        self._counter: int = 0

    def add_record(
        self,
        canvas_image: np.ndarray,
        preprocessed_image: np.ndarray,
        predicted_label: int,
        confidence: float,
        probabilities: np.ndarray,
        image_hash: str,
        notes: Optional[str] = None
    ) -> HistoryRecord:
        """새로운 예측 기록을 추가합니다.

        Args:
            canvas_image: 원본 캔버스 이미지
            preprocessed_image: 전처리된 이미지
            predicted_label: 예측된 레이블
            confidence: 신뢰도
            probabilities: 확률 배열
            image_hash: 이미지 해시
            notes: 추가 메모

        Returns:
            생성된 히스토리 레코드
        """
        self._counter += 1
        timestamp = datetime.datetime.now().isoformat()

        record = HistoryRecord(
            record_id=self._counter,
            canvas_image=canvas_image,
            preprocessed_image=preprocessed_image,
            predicted_label=predicted_label,
            confidence=confidence,
            probabilities=probabilities,
            timestamp=timestamp,
            image_hash=image_hash,
            notes=notes
        )

        self._records.append(record)

        # 최대 레코드 수 초과 시 오래된 것 삭제
        if len(self._records) > self.max_records:
            self._records.pop(0)

        return record

    def get_all_records(self, reverse: bool = True) -> List[HistoryRecord]:
        """모든 히스토리 레코드를 반환합니다.

        Args:
            reverse: True일 경우 최신 순으로 정렬

        Returns:
            히스토리 레코드 리스트
        """
        if reverse:
            return list(reversed(self._records))
        return self._records.copy()

    def get_record_by_id(self, record_id: int) -> Optional[HistoryRecord]:
        """특정 ID의 레코드를 반환합니다.

        Args:
            record_id: 레코드 ID

        Returns:
            해당 ID의 레코드 (없으면 None)
        """
        for record in self._records:
            if record.record_id == record_id:
                return record
        return None

    def get_records_by_label(
        self,
        label: int,
        reverse: bool = True
    ) -> List[HistoryRecord]:
        """특정 예측 레이블을 가진 레코드들을 반환합니다.

        Args:
            label: 예측 레이블 (0-9)
            reverse: True일 경우 최신 순으로 정렬

        Returns:
            해당 레이블의 레코드 리스트
        """
        filtered = [r for r in self._records if r.predicted_label == label]
        if reverse:
            return list(reversed(filtered))
        return filtered

    def get_records_by_confidence_range(
        self,
        min_confidence: float = 0.0,
        max_confidence: float = 1.0,
        reverse: bool = True
    ) -> List[HistoryRecord]:
        """신뢰도 범위 내의 레코드들을 반환합니다.

        Args:
            min_confidence: 최소 신뢰도
            max_confidence: 최대 신뢰도
            reverse: True일 경우 최신 순으로 정렬

        Returns:
            해당 범위의 레코드 리스트
        """
        filtered = [
            r for r in self._records
            if min_confidence <= r.confidence <= max_confidence
        ]
        if reverse:
            return list(reversed(filtered))
        return filtered

    def delete_record(self, record_id: int) -> bool:
        """특정 ID의 레코드를 삭제합니다.

        Args:
            record_id: 삭제할 레코드 ID

        Returns:
            삭제 성공 여부
        """
        for i, record in enumerate(self._records):
            if record.record_id == record_id:
                self._records.pop(i)
                return True
        return False

    def clear_all(self) -> None:
        """모든 히스토리를 삭제합니다."""
        self._records.clear()
        self._counter = 0

    def get_statistics(self) -> Dict:
        """히스토리 통계를 반환합니다.

        Returns:
            통계 정보 딕셔너리
        """
        if not self._records:
            return {
                "total_count": 0,
                "label_distribution": {},
                "avg_confidence": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0
            }

        # 레이블 분포
        label_counts = {}
        for record in self._records:
            label = record.predicted_label
            label_counts[label] = label_counts.get(label, 0) + 1

        # 신뢰도 통계
        confidences = [r.confidence for r in self._records]

        return {
            "total_count": len(self._records),
            "label_distribution": label_counts,
            "avg_confidence": np.mean(confidences),
            "min_confidence": np.min(confidences),
            "max_confidence": np.max(confidences)
        }

    def __len__(self) -> int:
        """현재 저장된 레코드 수를 반환합니다."""
        return len(self._records)

    def __repr__(self) -> str:
        """객체의 문자열 표현을 반환합니다."""
        return f"HistoryManager(records={len(self._records)}, max={self.max_records})"
