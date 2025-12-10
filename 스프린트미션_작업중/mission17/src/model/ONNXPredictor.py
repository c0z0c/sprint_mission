# -*- coding: utf-8 -*-
"""MNIST ONNX 모델링 API - 클래스 기반 설계

이 모듈은 MNIST 숫자 예측을 위한 ONNX 모델 관리, 이미지 전처리, 추론 기능을 제공합니다.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import requests
from PIL import Image
from helper_dev_utils import get_auto_logger
logger = get_auto_logger()

from .DataClass import PredictionResult

# ============================================================================
# ONNX 모델 추론 클래스
# ============================================================================

class ONNXPredictor:
    """ONNX Runtime을 사용한 모델 추론 클래스

    ONNX 모델을 로드하고 추론을 수행합니다.
    """

    def __init__(self, model_path: Path, providers: Optional[List[str]] = None):
        """
        Args:
            model_path: ONNX 모델 파일 경로
            providers: ONNX Runtime 실행 프로바이더 리스트
                      (기본값: ['CPUExecutionProvider'])
        """
        self.model_path = model_path

        if providers is None:
            providers = ['CPUExecutionProvider']

        # ONNX Runtime 세션 생성
        self.session = ort.InferenceSession(
            str(model_path),
            providers=providers
        )

        # 모델 입력/출력 정보 확인
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        logger.debug(f"모델 로드 완료: {model_path}")
        logger.debug(f"입력 이름: {self.input_name}")
        logger.debug(f"출력 이름: {self.output_name}")

    def predict(self, input_data: np.ndarray) -> PredictionResult:
        """이미지를 입력받아 예측을 수행합니다.

        Args:
            input_data: 전처리된 이미지 (1, 1, 28, 28), float32

        Returns:
            예측 결과 객체
        """
        # 추론 실행
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_data}
        )

        # 출력 형태: [1, 10] (로그 확률 또는 로짓)
        logits = outputs[0][0]

        # Softmax 적용하여 확률로 변환
        probabilities = self._softmax(logits)

        # 예측 레이블 및 신뢰도
        predicted_label = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_label])

        return PredictionResult(
            predicted_label=predicted_label,
            confidence=confidence,
            probabilities=probabilities
        )

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax 함수를 적용합니다.

        Args:
            x: 입력 배열 (로짓)

        Returns:
            확률 분포 (합이 1)
        """
        # 수치 안정성을 위해 최댓값을 뺌
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

