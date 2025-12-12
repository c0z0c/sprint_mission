# -*- coding: utf-8 -*-
"""MNIST ONNX 모델링 API - 클래스 기반 설계

이 모듈은 MNIST 숫자 예측을 위한 ONNX 모델 관리, 이미지 전처리, 추론 기능을 제공합니다.
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import logging
import cv2
import numpy as np
import onnxruntime as ort
import requests
from PIL import Image
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[2]  # mission17/
sys.path.insert(0, str(project_root))

from helper_dev_utils import get_auto_logger
from src.model.DataClass import ModelConfig

logger = get_auto_logger(log_level=logging.DEBUG)

# ============================================================================
# 모델 다운로더 클래스
# ============================================================================

# ONNX 파일 매직 넘버 (처음 8바이트)
ONNX_MAGIC = b"\x08\x03\x12\x02\x1a\x00"  # ONNX protobuf 시그니처
MIN_ONNX_SIZE = 1024  # 최소 1KB


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
        # 절대 경로로 정규화
        self.cache_dir = Path(config.cache_dir).resolve()
        self.model_path = self.cache_dir / config.model_name

    def _validate_onnx_file(self, file_path: Path) -> bool:
        """ONNX 파일 무결성 검증

        Args:
            file_path: 검증할 파일 경로

        Returns:
            검증 성공 여부
        """
        try:
            # 파일 크기 검증
            file_size = file_path.stat().st_size
            if file_size < MIN_ONNX_SIZE:
                logger.error(
                    f"파일 크기 부족: {file_size} bytes < {MIN_ONNX_SIZE} bytes"
                )
                return False

            # ONNX 매직 넘버 검증
            with open(file_path, "rb") as f:
                header = f.read(8)
                # ONNX protobuf는 0x08로 시작
                if not header.startswith(b"\x08"):
                    logger.error(f"ONNX 매직 넘버 불일치: {header[:4].hex()}")
                    return False

            # ONNX Runtime 로드 테스트 (경고는 허용)
            try:
                session = ort.InferenceSession(
                    str(file_path), providers=["CPUExecutionProvider"]
                )
                input_shape = session.get_inputs()[0].shape
                output_shape = session.get_outputs()[0].shape

                logger.debug(
                    f"ONNX 검증 성공: input={input_shape}, output={output_shape}"
                )

            except Exception as ort_error:
                # opset 버전 문제는 경고만 출력 (실제 추론 시 판단)
                error_msg = str(ort_error)
                if "NOT_IMPLEMENTED" in error_msg or "opset" in error_msg.lower():
                    logger.warning(
                        f"ONNX Runtime 호환성 경고 (opset 버전): {error_msg}\n"
                        f"모델 로드를 시도하지만 추론 시 실패할 수 있습니다."
                    )
                    # 파일 자체는 유효하므로 True 반환
                    return True
                else:
                    # 다른 치명적 에러는 실패 처리
                    logger.error(f"ONNX Runtime 로드 실패: {error_msg}")
                    return False

            return True

        except Exception as e:
            logger.error(f"ONNX 검증 실패: {e}")
            return False

    def download(
        self,
        force: bool = False,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        max_retries: int = 3,
    ) -> Path:
        """모델 파일을 다운로드합니다.

        Args:
            force: True일 경우 기존 파일이 있어도 재다운로드
            progress_callback: 진행률 콜백 함수 (progress: float, status: str)
            max_retries: 최대 재시도 횟수

        Returns:
            다운로드된 모델 파일 경로

        Raises:
            requests.RequestException: 다운로드 실패 시
            ValueError: 모델 검증 실패 시
        """
        # 캐시 디렉토리 생성
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"캐시 디렉토리 (절대경로): {self.cache_dir}")

        # 이미 파일이 존재하고 force가 False면 검증 후 반환
        if self.model_path.exists() and not force:
            logger.debug(f"기존 모델 파일 발견: {self.model_path}")
            if self._validate_onnx_file(self.model_path):
                logger.debug("기존 모델 검증 성공")
                return self.model_path
            else:
                logger.warning("기존 모델 검증 실패, 재다운로드 시작")

        # 임시 파일 경로
        tmp_path = self.model_path.with_suffix(".tmp")

        # 재시도 로직
        for attempt in range(1, max_retries + 1):
            try:
                logger.debug(
                    f"모델 다운로드 시도 ({attempt}/{max_retries}): {self.config.model_url}"
                )

                # 파일 다운로드
                response = requests.get(self.config.model_url, stream=True, timeout=30)
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))
                downloaded_size = 0

                # 임시 파일로 다운로드
                with open(tmp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)

                            # 진행률 표시
                            if total_size > 0:
                                progress = downloaded_size / total_size
                                status_msg = f"다운로드 중 ({attempt}/{max_retries}): {progress*100:.1f}%"

                                if progress_callback:
                                    progress_callback(progress, status_msg)

                                if progress % 0.2 < 0.01:  # 20%마다 로깅
                                    logger.debug(status_msg)

                logger.debug(f"다운로드 완료: {tmp_path} ({downloaded_size} bytes)")

                # 파일 검증
                if not self._validate_onnx_file(tmp_path):
                    raise ValueError("다운로드된 파일 검증 실패")

                # 검증 통과 시 최종 경로로 이동
                if self.model_path.exists():
                    self.model_path.unlink()
                tmp_path.rename(self.model_path)

                logger.info(f"모델 다운로드 및 검증 완료: {self.model_path}")
                return self.model_path

            except (requests.RequestException, ValueError, OSError) as e:
                logger.warning(f"다운로드 시도 {attempt} 실패: {e}")

                # 임시 파일 정리
                if tmp_path.exists():
                    tmp_path.unlink()

                if attempt < max_retries:
                    wait_time = 2**attempt  # 지수 백오프: 2, 4, 8초
                    logger.debug(f"{wait_time}초 대기 후 재시도")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(
                        f"모델 다운로드 실패 ({max_retries}회 시도): {e}"
                    ) from e

    def get_model_path(self) -> Optional[Path]:
        """캐시된 모델 파일 경로를 반환합니다.

        Returns:
            모델 파일 경로 (존재하지 않으면 None)
        """
        return self.model_path if self.model_path.exists() else None
