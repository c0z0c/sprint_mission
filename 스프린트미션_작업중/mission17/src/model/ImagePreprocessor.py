# -*- coding: utf-8 -*-
"""MNIST ONNX 모델링 API - 클래스 기반 설계

이 모듈은 MNIST 숫자 예측을 위한 ONNX 모델 관리, 이미지 전처리, 추론 기능을 제공합니다.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import cv2
import numpy as np
import onnxruntime as ort
import requests
from PIL import Image
from scipy import ndimage
from helper_dev_utils import get_auto_logger
logger = get_auto_logger(log_level=logging.DEBUG)

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
        2. 중심 정렬 (바운딩 박스 추출 후 정사각형으로 크롭 및 중앙 배치)
        3. 28x28 크기로 리사이즈
        4. 색상 반전 (검은 선/흰 배경 -> 흰 숫자/검은 배경)
        5. 정규화 (0~255 -> 0.0~1.0)
        6. 형태 변경 (28, 28) -> (1, 1, 28, 28)

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

        # 4. 중심 정렬 (원본 크기에서 수행)
        inverted = self._center_align(inverted)

        # 5. 정규화 (0~255 -> 0.0~1.0)
        normalized = self._normalize(inverted)

        # 6. 형태 변경 (1, 1, 28, 28)
        model_input = normalized.reshape(1, 1, 28, 28).astype(np.float32)

        # 표시용 이미지 (28x28)
        display_image = inverted

        return model_input, display_image
    
    def _center_align(self, inverted: np.ndarray) -> np.ndarray:
        """
        반전 된 값이다
        중심 정렬 하자

        MNIST 학습 데이터와 동일하게 숫자를 이미지 중앙에 배치하여 인식률을 향상시킵니다.
        질량 중심(center of mass)을 계산하여 이미지 중심으로 이동시킵니다.

        Args:
            inverted: 그레이스케일 이미지 (흰 숫자/검은 배경)

        Returns:
            중심 정렬된 이미지 (원본과 동일한 크기)
        """
        # 빈 이미지 체크 (모든 픽셀이 0인 경우)
        if np.sum(inverted) == 0:
            logger.debug("빈 이미지 감지: 중심 정렬 스킵")
            return inverted
        
        # 질량 중심 계산 (픽셀 값이 가중치로 작용)
        cy, cx = ndimage.center_of_mass(inverted)
        
        # NaN 체크 (예외적인 경우)
        if np.isnan(cy) or np.isnan(cx):
            logger.debug(f"질량 중심 계산 실패 (NaN): 원본 반환")
            return inverted
        
        # 이미지 중심 좌표 (28x28의 경우 13.5, 13.5)
        rows, cols = inverted.shape
        center_y = rows / 2.0
        center_x = cols / 2.0
        
        # 시프트 벡터 계산 (이미지 중심 - 질량 중심)
        shift_y = center_y - cy
        shift_x = center_x - cx
        
        # # 시프트 벡터 크기 제한 (너무 큰 이동 방지)
        # max_shift = 5.0
        # shift_y = np.clip(shift_y, -max_shift, max_shift)
        # shift_x = np.clip(shift_x, -max_shift, max_shift)
        # logger.debug(f"질량 중심: ({cy:.2f}, {cx:.2f}), 시프트: ({shift_y:.2f}, {shift_x:.2f})")
        # # 이미지 시프트 (mode='constant', cval=0: 검은 배경 유지)
        # result = ndimage.shift(inverted, [shift_y, shift_x], mode='constant', cval=0)

        # 시프트 벡터 계산 (이미지 중심 - 질량 중심)
        shift_y = center_y - cy
        shift_x = center_x - cx
        
        logger.debug(f"질량 중심: ({cy:.2f}, {cx:.2f}), 시프트: ({shift_y:.2f}, {shift_x:.2f})")
        
        # 이미지 시프트 (mode='constant', cval=0: 검은 배경 유지)
        result = ndimage.shift(inverted, [shift_y, shift_x], mode='constant', cval=0)
        
        return result

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
