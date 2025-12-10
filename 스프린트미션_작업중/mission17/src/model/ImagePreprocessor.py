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
from helper_utils import get_auto_logger
logger = get_auto_logger()

# ============================================================================
# 이미지 전처리 클래스
# ============================================================================


class ImagePreprocessor:
    """MNIST 모델을 위한 이미지 전처리 클래스

    캔버스 이미지를 ONNX 모델 입력 형식(1x1x28x28, float32)으로 변환합니다.
    """

    def __init__(self, target_size: Tuple[int, int] = (28, 28)):
        """
        Args:
            target_size: 목표 이미지 크기 (height, width)
        """
        self.target_size = target_size

    def preprocess(self, canvas_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """캔버스 이미지를 ONNX 모델 입력 형식으로 전처리합니다.

        처리 단계:
        1. RGBA/RGB -> 그레이스케일 변환
        2. 28x28 크기로 리사이즈
        3. 색상 반전 (검은 선/흰 배경 -> 흰 숫자/검은 배경)
        4. 정규화 (0~255 -> 0.0~1.0)
        5. 형태 변경 (28, 28) -> (1, 1, 28, 28)

        Args:
            canvas_image: 캔버스 이미지 (RGBA 또는 RGB)

        Returns:
            model_input: 모델 입력용 배열 (1, 1, 28, 28), float32
            display_image: 표시용 28x28 이미지
        """
        # 1. 그레이스케일 변환
        grayscale = self._to_grayscale(canvas_image)

        # 2. 크기 조정 (28x28)
        resized = self._resize(grayscale, self.target_size)

        # 3. 색상 반전 (MNIST는 흰색 숫자/검은색 배경을 기대)
        inverted = self._invert(resized)

        # 4. 정규화 (0~255 -> 0.0~1.0)
        normalized = self._normalize(inverted)

        # 5. 형태 변경 (1, 1, 28, 28)
        model_input = normalized.reshape(1, 1, 28, 28).astype(np.float32)

        # 표시용 이미지 (28x28)
        display_image = inverted

        return model_input, display_image

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """이미지를 그레이스케일로 변환합니다.

        Args:
            image: RGBA 또는 RGB 이미지

        Returns:
            그레이스케일 이미지
        """
        if len(image.shape) == 2:
            # 이미 그레이스케일
            return image

        if image.shape[2] == 4:
            # RGBA -> RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # RGB -> 그레이스케일
        grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return grayscale

    def _resize(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """이미지 크기를 조정합니다.

        Args:
            image: 입력 이미지
            target_size: 목표 크기 (height, width)

        Returns:
            리사이즈된 이미지
        """
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        return resized

    def _invert(self, image: np.ndarray) -> np.ndarray:
        """이미지 색상을 반전합니다.

        Args:
            image: 그레이스케일 이미지

        Returns:
            반전된 이미지
        """
        inverted = 255 - image
        return inverted

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """픽셀 값을 0.0~1.0으로 정규화합니다.

        Args:
            image: 입력 이미지 (0~255)

        Returns:
            정규화된 이미지 (0.0~1.0)
        """
        normalized = image.astype(np.float32) / 255.0
        return normalized
