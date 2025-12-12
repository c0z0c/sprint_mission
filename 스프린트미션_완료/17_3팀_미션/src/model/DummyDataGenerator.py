# -*- coding: utf-8 -*-
"""MNIST ONNX 모델링 API

이 모듈은 MNIST 숫자 예측을 위한 ONNX 모델 관리, 이미지 전처리, 추론 기능을 제공합니다.
"""

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import requests
from helper_dev_utils import get_auto_logger
from PIL import Image

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.model.DataClass import PredictionResult

logger = get_auto_logger(log_level=logging.DEBUG)

# ============================================================================
# 더미 데이터 생성 유틸리티
# ============================================================================


class DummyDataGenerator:
    """테스트용 더미 데이터를 생성하는 클래스"""

    @staticmethod
    def generate_dummy_canvas(
        size: Tuple[int, int] = (200, 200), digit: Optional[int] = None
    ) -> np.ndarray:
        """더미 캔버스 이미지를 생성합니다.

        Args:
            size: 캔버스 크기 (width, height)
            digit: 그릴 숫자 (None일 경우 랜덤)

        Returns:
            RGBA 캔버스 이미지
        """
        # 흰색 배경 생성
        canvas = np.ones((size[1], size[0], 4), dtype=np.uint8) * 255

        # 더미로 랜덤 선 그리기
        num_strokes = np.random.randint(3, 8)
        for _ in range(num_strokes):
            x1 = np.random.randint(50, size[0] - 50)
            y1 = np.random.randint(50, size[1] - 50)
            x2 = x1 + np.random.randint(-30, 30)
            y2 = y1 + np.random.randint(-30, 30)

            cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 0, 255), thickness=5)

        return canvas

    @staticmethod
    def generate_dummy_prediction() -> PredictionResult:
        """더미 예측 결과를 생성합니다.

        Returns:
            더미 예측 결과
        """
        # 랜덤 로그값 생성 후 softmax 적용
        logits = np.random.randn(10)
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)

        predicted_label = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_label])

        # 더미 전처리 이미지 (28x28 랜덤)
        preprocessed_image = np.random.randint(0, 255, (28, 28), dtype=np.uint8)

        return PredictionResult(
            predicted_label=predicted_label,
            confidence=confidence,
            probabilities=probabilities,
            preprocessed_image=preprocessed_image,
        )
