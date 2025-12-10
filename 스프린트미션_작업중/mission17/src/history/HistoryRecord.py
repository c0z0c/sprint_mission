# -*- coding: utf-8 -*-
"""MNIST 예측 히스토리 저장소 API - 클래스 기반 설계

이 모듈은 예측 기록을 저장, 조회, 관리하는 기능을 제공합니다.
"""

import datetime
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from helper_dev_utils import get_auto_logger
logger = get_auto_logger()


# ============================================================================
# 히스토리 레코드 데이터 클래스
# ============================================================================


@dataclass
class HistoryRecord:
    """예측 히스토리 레코드를 담는 데이터 클래스

    Attributes:
        record_id: 고유 레코드 ID
        canvas_image: 원본 캔버스 이미지 (numpy 배열)
        preprocessed_image: 전처리된 28x28 이미지 (numpy 배열)
        predicted_label: 예측된 숫자 (0-9)
        confidence: 신뢰도 (0.0 ~ 1.0)
        probabilities: 각 클래스별 확률 배열
        timestamp: 예측 시각 (ISO 8601 형식 문자열)
        image_hash: 전처리된 이미지의 SHA256 해시 (중복 방지용)
        notes: 추가 메모 (선택적)
    """
    record_id: int
    canvas_image: np.ndarray
    preprocessed_image: np.ndarray
    predicted_label: int
    confidence: float
    probabilities: np.ndarray
    timestamp: str
    image_hash: str
    notes: Optional[str] = None

    def to_dict(self) -> Dict:
        """레코드를 딕셔너리로 변환합니다 (numpy 배열 제외).

        Returns:
            직렬화 가능한 딕셔너리
        """
        return {
            "record_id": self.record_id,
            "predicted_label": self.predicted_label,
            "confidence": self.confidence,
            "probabilities": self.probabilities.tolist(),
            "timestamp": self.timestamp,
            "image_hash": self.image_hash,
            "notes": self.notes
        }

    def to_streamlit_dict(self) -> Dict:
        """Streamlit 표시용 딕셔너리로 변환합니다.

        Returns:
            Streamlit UI에 표시할 딕셔너리
        """
        return {
            "id": self.record_id,
            "canvas_image": self.canvas_image,
            "preprocessed_image": self.preprocessed_image,
            "predicted_label": self.predicted_label,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "timestamp": self.timestamp,
            "image_hash": self.image_hash,
            "notes": self.notes
        }

    @staticmethod
    def compute_image_hash(preprocessed_image: np.ndarray, use_bbox_resize: bool = True) -> str:
        """전처리된 이미지의 SHA256 해시를 계산합니다.

        Args:
            preprocessed_image: 전처리된 28x28 이미지 (numpy 배열)

        Returns:
            SHA256 해시 문자열 (16진수)
        """
        image_bytes = preprocessed_image.tobytes()
        # 전처리 방식을 해시에 포함
        combined = image_bytes + bytes([use_bbox_resize])
        return hashlib.sha256(combined).hexdigest()