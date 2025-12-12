# -*- coding: utf-8 -*-
"""MNIST ONNX 모델링 API - 이미지 전처리

이 모듈은 캔버스 이미지를 ONNX 모델 입력 형식으로 변환하는 ImagePreprocessor 클래스를 제공합니다.

주요 기능:
    - 그레이스케일 변환 및 색상 반전
    - 바운딩 박스 기반 리사이즈 (비율 유지)
    - 직접 리사이즈 (전체 캔버스)
    - 정규화 및 shape 변환 (1, 1, 28, 28)

전처리 과정:
    1. RGBA/RGB → 그레이스케일
    2. 색상 반전 (검은 선 → 흰 숫자)
    3. 바운딩 박스 추출 및 리사이즈
    4. 28x28 캔버스 중앙 배치
    5. 정규화 (0~255 → 0.0~1.0)
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
# 이미지 전처리 클래스
# ============================================================================


class ImagePreprocessor:
    """MNIST 모델을 위한 이미지 전처리 클래스

    캔버스 이미지를 ONNX 모델 입력 형식(1x1x28x28, float32)으로 변환합니다.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (28, 28),
        target_content_size: int = 19,
        bbox_threshold: int = 10,
    ):
        """
        Args:
            target_size: 목표 이미지 크기 (height, width) - 최종 캔버스 크기
            target_content_size: 바운딩 박스 리사이즈 시 최대 크기 (24*0.8 = 19)
            bbox_threshold: 바운딩 박스 계산 시 픽셀 임계값 (배경 노이즈 제거)
        """
        self.target_size = target_size
        self.target_content_size = target_content_size
        self.bbox_threshold = bbox_threshold

    def preprocess(
        self, canvas_image: np.ndarray, use_bbox_resize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """캔버스 이미지를 ONNX 모델 입력 형식으로 전처리합니다.

        처리 단계:
        1. RGBA/RGB -> 그레이스케일 변환
        2. 색상 반전 (검은 선/흰 배경 -> 흰 숫자/검은 배경)
        3-A. (use_bbox_resize=True) 바운딩 박스 추출 → 비율 유지 리사이즈 → 중앙 배치
        3-B. (use_bbox_resize=False) 전체 이미지 직접 리사이즈 (28x28)
        4. 정규화 (0~255 -> 0.0~1.0)
        5. 형태 변경 (28, 28) -> (1, 1, 28, 28)

        Args:
            canvas_image: 캔버스 이미지 (RGBA 또는 RGB)
            use_bbox_resize: True=바운딩 박스 기반 리사이즈, False=직접 리사이즈

        Returns:
            model_input: 모델 입력용 배열 (1, 1, 28, 28), float32
            display_image: 표시용 28x28 이미지
        """
        # 1. 그레이스케일 변환
        grayscale = self._to_grayscale(canvas_image)

        # 2. 색상 반전 (MNIST는 흰색 숫자/검은색 배경을 기대)
        inverted = self._invert(grayscale)

        # 3. 전처리 방식 분기
        logger.debug(f"use_bbox_resize: {use_bbox_resize}")

        if use_bbox_resize:
            # 3-A. 바운딩 박스 기반 리사이즈
            bbox = self._get_bounding_box(inverted)

            if bbox is None:
                # 빈 캔버스: 28x28 검은 이미지 반환
                empty_canvas = np.zeros(self.target_size, dtype=np.uint8)
                normalized = self._normalize(empty_canvas)
                model_input = normalized.reshape(1, 1, 28, 28).astype(np.float32)
                return model_input, empty_canvas

            # 비율 유지 리사이즈
            resized = self._resize_with_aspect_ratio(inverted, bbox)
            # 28x28 캔버스 중앙 배치
            final_image = self._place_on_canvas(resized)
        else:
            # 3-B. 직접 리사이즈 (200x200 -> 28x28)
            final_image = self._resize_to_content_size(inverted)

        # 4. 정규화 (0~255 -> 0.0~1.0)
        normalized = self._normalize(final_image)

        # 7. 형태 변경 (1, 1, 28, 28)
        model_input = normalized.reshape(1, 1, 28, 28).astype(np.float32)

        # 표시용 이미지 (28x28)
        display_image = final_image

        return model_input, display_image

    def _get_bounding_box(
        self, image: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """이미지에서 0이 아닌 픽셀의 바운딩 박스를 계산합니다.

        Args:
            image: 반전된 그레이스케일 이미지 (흰 숫자/검은 배경)

        Returns:
            (y_min, y_max, x_min, x_max) 또는 빈 이미지 시 None
        """
        # 임계값 이상인 픽셀 위치 추출 (배경 노이즈 제거)
        mask = image > self.bbox_threshold
        rows, cols = np.where(mask)

        # 빈 이미지 체크
        if len(rows) == 0:
            logger.debug("바운딩 박스 없음: 빈 이미지")
            return None

        # 바운딩 박스 좌표 계산
        y_min, y_max = rows.min(), rows.max()
        x_min, x_max = cols.min(), cols.max()

        logger.debug(
            f"바운딩 박스: y[{y_min}:{y_max}], x[{x_min}:{x_max}], 크기: ({y_max-y_min+1}, {x_max-x_min+1})"
        )

        return (y_min, y_max, x_min, x_max)

    def _resize_with_aspect_ratio(
        self, image: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """바운딩 박스 영역을 크롭 후 비율을 유지하며 리사이즈합니다.

        Args:
            image: 입력 이미지
            bbox: 바운딩 박스 (y_min, y_max, x_min, x_max)

        Returns:
            리사이즈된 이미지 (최대 변이 target_content_size)
        """
        y_min, y_max, x_min, x_max = bbox

        # 바운딩 박스 영역 크롭 (+1은 inclusive 범위)
        cropped = image[y_min : y_max + 1, x_min : x_max + 1]
        h, w = cropped.shape[:2]

        logger.debug(f"크롭 후 크기: ({h}, {w})")

        # 최대 변 기준 스케일 계산
        max_dim = max(h, w)
        scale = self.target_content_size / max_dim

        # 새 크기 계산 (최소 1픽셀 보장)
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))

        logger.debug(f"리사이즈: ({h}, {w}) -> ({new_h}, {new_w}), 스케일: {scale:.3f}")

        # 리사이즈 (INTER_AREA: 축소 시 품질 우수)
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return resized

    def _resize_to_content_size(self, image: np.ndarray) -> np.ndarray:
        """이미지를 목표 크기로 직접 리사이즈합니다 (바운딩 박스 없이).

        200x200 캔버스를 28x28로 직접 축소하여 배경을 포함한 전체 이미지를 변환합니다.
        바운딩 박스 기반 방식과 성능 비교를 위한 대안 전처리 방식입니다.

        Args:
            image: 입력 이미지 (반전된 그레이스케일)

        Returns:
            리사이즈된 이미지 (target_size)
        """
        resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        logger.debug(f"직접 리사이즈: {image.shape} -> {resized.shape}")
        return resized

    def _place_on_canvas(self, image: np.ndarray) -> np.ndarray:
        """리사이즈된 이미지를 목표 크기 캔버스 중앙에 배치합니다.

        Args:
            image: 리사이즈된 이미지

        Returns:
            중앙 배치된 이미지 (target_size)
        """
        canvas_h, canvas_w = self.target_size
        h, w = image.shape[:2]

        # 크기 검증 (예외 처리)
        if h > canvas_h or w > canvas_w:
            logger.warning(
                f"이미지 크기({h}, {w})가 캔버스({canvas_h}, {canvas_w})보다 큼: 추가 리사이즈"
            )
            # 긴급 리사이즈
            scale = min(canvas_h / h, canvas_w / w)
            new_h = max(1, int(h * scale))
            new_w = max(1, int(w * scale))
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = new_h, new_w

        # 빈 캔버스 생성
        canvas = np.zeros((canvas_h, canvas_w), dtype=image.dtype)

        # 중심 좌표 계산
        offset_y = (canvas_h - h) // 2
        offset_x = (canvas_w - w) // 2

        logger.debug(
            f"캔버스 배치: 오프셋 ({offset_y}, {offset_x}), 이미지 크기 ({h}, {w})"
        )

        # 이미지 배치
        canvas[offset_y : offset_y + h, offset_x : offset_x + w] = image

        return canvas

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

        INTER_AREA 보간법을 사용하여 축소 시 품질을 유지합니다.

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

        캔버스의 검은 선을 흰색으로 변환하여 MNIST 모델 입력 형식에 맞춥니다.
        (MNIST는 검은 배경에 흰 숫자를 학습했습니다)

        Args:
            image: 그레이스케일 이미지

        Returns:
            반전된 이미지
        """
        inverted = 255 - image
        return inverted

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """픽셀 값을 0.0~1.0으로 정규화합니다.

        ONNX 모델의 입력 형식에 맞게 uint8 범위를 float32로 변환합니다.

        Args:
            image: 입력 이미지 (0~255)

        Returns:
            정규화된 이미지 (0.0~1.0, float32)
        """
        normalized = image.astype(np.float32) / 255.0
        return normalized
