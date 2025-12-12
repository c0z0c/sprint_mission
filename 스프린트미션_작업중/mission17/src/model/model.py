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
import sys
from PIL import Image
from helper_dev_utils import get_auto_logger

logger = get_auto_logger(log_level=logging.DEBUG)

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.model.DataClass import PredictionResult
from src.model.DataClass import ModelConfig
from src.model.DummyDataGenerator import DummyDataGenerator
from src.model.ImagePreprocessor import ImagePreprocessor
from src.model.ModelDownloader import ModelDownloader
from src.model.ONNXPredictor import ONNXPredictor
from src.model.MNISTPipeline import MNISTPipeline

logger = get_auto_logger(log_level=logging.DEBUG)

# ============================================================================
# 사용 예시 및 테스트
# ============================================================================


def main():
    """사용 예시"""
    logger.debug("=" * 60)
    logger.debug("MNIST ONNX 모델링 API 테스트")
    logger.debug("=" * 60)

    # 1. 파이프라인 생성 및 초기화
    logger.debug("\n[1] 파이프라인 초기화 중...")
    pipeline = MNISTPipeline()
    pipeline.initialize()

    # 2. 더미 캔버스 이미지 생성
    logger.debug("\n[2] 더미 캔버스 이미지 생성 중...")
    dummy_canvas = DummyDataGenerator.generate_dummy_canvas()
    logger.debug(f"캔버스 이미지 형태: {dummy_canvas.shape}")

    # 3. 예측 수행
    logger.debug("\n[3] 예측 수행 중...")
    result = pipeline.predict(dummy_canvas)

    logger.debug(f"\n예측 결과:")
    logger.debug(f"  - 예측 숫자: {result.predicted_label}")
    logger.debug(f"  - 신뢰도: {result.confidence:.2%}")
    logger.debug(f"  - 확률 분포: {result.probabilities}")
    logger.debug(f"  - 전처리 이미지 형태: {result.preprocessed_image.shape}")

    # 4. 더미 데이터 생성 테스트
    logger.debug("\n[4] 더미 예측 결과 생성 테스트...")
    dummy_result = DummyDataGenerator.generate_dummy_prediction()
    logger.debug(f"  - 더미 예측 숫자: {dummy_result.predicted_label}")
    logger.debug(f"  - 더미 신뢰도: {dummy_result.confidence:.2%}")

    logger.debug("\n" + "=" * 60)
    logger.debug("테스트 완료!")
    logger.debug("=" * 60)


if __name__ == "__main__":
    main()
