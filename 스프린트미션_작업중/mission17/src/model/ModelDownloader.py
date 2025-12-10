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

from src.model.DataClass import ModelConfig

# ============================================================================
# 모델 다운로더 클래스
# ============================================================================

class ModelDownloader:
    """ONNX 모델 다운로드를 관리하는 클래스

    GitHub ONNX Models 저장소에서 MNIST 모델을 다운로드하고 캐싱합니다.
    """

    def __init__(self, config: ModelConfig):
        """
        Args:
            config: 모델 설정 객체
        """
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.model_path = self.cache_dir / config.model_name

    def download(self, force: bool = False) -> Path:
        """모델 파일을 다운로드합니다.

        Args:
            force: True일 경우 기존 파일이 있어도 재다운로드

        Returns:
            다운로드된 모델 파일 경로

        Raises:
            requests.RequestException: 다운로드 실패 시
        """
        # 캐시 디렉토리 생성
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 이미 파일이 존재하고 force가 False면 기존 파일 반환
        if self.model_path.exists() and not force:
            logger.debug(f"모델 파일이 이미 존재합니다: {self.model_path}")
            return self.model_path

        logger.debug(f"모델 다운로드 중: {self.config.model_url}")

        # 파일 다운로드
        response = requests.get(self.config.model_url, stream=True)
        response.raise_for_status()

        # 파일 저장
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0

        with open(self.model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)

                    # 진행률 표시
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        logger.debug(f"\r다운로드 진행률: {progress:.1f}%", end='')

        logger.debug(f"\n모델 다운로드 완료: {self.model_path}")
        return self.model_path

    def get_model_path(self) -> Optional[Path]:
        """캐시된 모델 파일 경로를 반환합니다.

        Returns:
            모델 파일 경로 (존재하지 않으면 None)
        """
        return self.model_path if self.model_path.exists() else None
