# -*- coding: utf-8 -*-
"""MNIST ONNX 모델링 API - 통합 파이프라인

이 모듈은 MNIST 예측의 전체 과정을 통합 관리하는 MNISTPipeline 클래스를 제공합니다.

주요 기능:
    - 모델 다운로드 및 로딩 자동화
    - 이미지 전처리 및 추론 일괄 처리
    - 단일 인터페이스로 예측 수행

파이프라인 구조:
    MNISTPipeline
    ├── ModelDownloader: 모델 다운로드
    ├── ImagePreprocessor: 이미지 전처리
    └── ONNXPredictor: 모델 추론

사용 예:
    pipeline = MNISTPipeline(config)
    pipeline.initialize()
    result = pipeline.predict(canvas_image)
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


from src.model.DataClass import ModelConfig, PredictionResult
from src.model.DummyDataGenerator import DummyDataGenerator
from src.model.ImagePreprocessor import ImagePreprocessor
from src.model.ModelDownloader import ModelDownloader
from src.model.ONNXPredictor import ONNXPredictor

logger = get_auto_logger(log_level=logging.DEBUG)

# ============================================================================
# 통합 MNIST 예측 파이프라인 클래스
# ============================================================================


class MNISTPipeline:
    """MNIST 숫자 예측을 위한 통합 파이프라인 클래스

    모델 다운로드, 로딩, 전처리, 추론을 하나의 인터페이스로 제공합니다.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Args:
            config: 모델 설정 (None일 경우 기본 설정 사용)
        """
        self.config = config or ModelConfig()
        self.downloader = ModelDownloader(self.config)
        self.preprocessor = ImagePreprocessor(target_size=self.config.input_shape[2:])
        self._predictor: Optional[ONNXPredictor] = None

    def initialize(self, force_download: bool = False) -> None:
        """파이프라인을 초기화합니다 (모델 다운로드 및 로드).

        Args:
            force_download: True일 경우 모델을 강제로 재다운로드
        """
        # 모델 다운로드
        model_path = self.downloader.download(force=force_download)

        # 모델 로드
        self._predictor = ONNXPredictor(model_path)

    @property
    def predictor(self) -> ONNXPredictor:
        """예측기 객체를 반환합니다.

        Returns:
            ONNXPredictor 객체

        Raises:
            RuntimeError: 파이프라인이 초기화되지 않은 경우
        """
        if self._predictor is None:
            raise RuntimeError(
                "파이프라인이 초기화되지 않았습니다. "
                "initialize() 메서드를 먼저 호출하세요."
            )
        return self._predictor

    def preprocess_only(
        self, canvas_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """캔버스 이미지를 전처리만 수행합니다 (추론 없이).

        Args:
            canvas_image: 캔버스 이미지 (RGBA 또는 RGB)

        Returns:
            model_input: 모델 입력용 배열 (1, 1, 28, 28)
            display_image: 표시용 28x28 이미지
        """
        return self.preprocessor.preprocess(canvas_image)

    def predict(
        self, canvas_image: np.ndarray, use_bbox_resize: bool = True
    ) -> PredictionResult:
        """캔버스 이미지로부터 숫자를 예측합니다.

        Args:
            canvas_image: 캔버스 이미지 (RGBA 또는 RGB)

        Returns:
            예측 결과 객체 (전처리된 이미지 포함)
        """
        # 이미지 전처리
        model_input, display_image = self.preprocessor.preprocess(
            canvas_image, use_bbox_resize=use_bbox_resize
        )

        # 추론 수행
        result = self.predictor.predict(model_input)

        # 전처리 이미지 추가
        result.preprocessed_image = display_image

        return result
