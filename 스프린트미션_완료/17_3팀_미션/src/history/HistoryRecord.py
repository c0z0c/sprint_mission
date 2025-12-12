# -*- coding: utf-8 -*-
"""MNIST 예측 히스토리 저장소 API - 레코드 데이터 클래스

이 모듈은 단일 예측 기록을 표현하는 HistoryRecord 데이터 클래스를 정의합니다.

주요 기능:
    - 예측 결과 및 이미지 정보 저장
    - 딕셔너리 변환 (JSON 직렬화용)
    - 이미지 해시 계산 (중복 방지)
"""

import datetime
import hashlib
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from helper_dev_utils import get_auto_logger
from PIL import Image

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

logger = get_auto_logger(log_level=logging.DEBUG)


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
            "notes": self.notes,
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
            "notes": self.notes,
        }

    @staticmethod
    def compute_image_hash(
        preprocessed_image: np.ndarray, use_bbox_resize: bool = True
    ) -> str:
        """전처리된 이미지의 SHA256 해시를 계산합니다.

        동일한 이미지가 여러 번 예측되는 것을 방지하기 위해
        이미지의 바이트 데이터와 전처리 방식을 조합하여 해시를 생성합니다.

        Args:
            preprocessed_image: 전처리된 28x28 이미지 (numpy 배열)
            use_bbox_resize: 바운딩 박스 리사이즈 사용 여부

        Returns:
            SHA256 해시 문자열 (16진수, 64자)
        """
        image_bytes = preprocessed_image.tobytes()
        # 전처리 방식을 해시에 포함
        combined = image_bytes + bytes([use_bbox_resize])
        return hashlib.sha256(combined).hexdigest()
