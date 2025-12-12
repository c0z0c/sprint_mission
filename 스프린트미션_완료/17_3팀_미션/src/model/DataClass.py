# -*- coding: utf-8 -*-
"""MNIST ONNX 모델링 API - 데이터 클래스 정의

이 모듈은 MNIST 예측에 사용되는 주요 데이터 클래스들을 정의합니다.

주요 클래스:
    - PredictionResult: 모델 예측 결과를 담는 데이터 클래스
    - ModelConfig: ONNX 모델 설정 정보를 담는 데이터 클래스

사용 예:
    config = ModelConfig()
    result = PredictionResult(predicted_label=7, confidence=0.95, ...)
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


logger = get_auto_logger(log_level=logging.DEBUG)


# ============================================================================
# 데이터 클래스 정의
# ============================================================================


@dataclass
class PredictionResult:
    """예측 결과를 담는 데이터 클래스

    Attributes:
        predicted_label: 예측된 숫자 (0-9)
        confidence: 최고 확률값 (0.0 ~ 1.0)
        probabilities: 각 클래스(0-9)별 확률 배열
        preprocessed_image: 전처리된 28x28 이미지 (선택적)
    """

    predicted_label: int
    confidence: float
    probabilities: np.ndarray
    preprocessed_image: Optional[np.ndarray] = None


@dataclass
class ModelConfig:
    """모델 설정을 담는 데이터 클래스

    Attributes:
        model_url: ONNX 모델 다운로드 URL
        model_name: 모델 파일명
        cache_dir: 모델 캐시 디렉토리 경로
        input_shape: 모델 입력 형태 (batch, channel, height, width)
        num_classes: 분류 클래스 개수
    """

    model_url: str = (
        "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.onnx"
    )
    model_name: str = "mnist-12.onnx"
    cache_dir: str = "./models"
    input_shape: Tuple[int, int, int, int] = (1, 1, 28, 28)
    num_classes: int = 10

    def __hash__(self):
        """Make ModelConfig hashable for Streamlit caching"""
        return hash(
            (
                self.model_url,
                self.model_name,
                self.cache_dir,
                self.input_shape,
                self.num_classes,
            )
        )
