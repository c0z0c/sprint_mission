# -*- coding: utf-8 -*-
"""MNIST 예측 히스토리 저장소 API - 파일 기반 매니저

이 모듈은 예측 기록을 파일 시스템에 영구 저장하는 FileHistoryManager를 제공합니다.

주요 기능:
    - 예측 기록 디스크 저장 (이미지 + 메타데이터)
    - 앱 재시작 후에도 기록 유지
    - 이미지 해시 기반 중복 검색
    - 자동 저장 모드 지원

저장 구조:
    ./history/
        ├── metadata.json          # 레코드 메타데이터
        └── images/
            ├── canvas_1.png       # 원본 캔버스 이미지
            ├── preprocessed_1.png # 전처리 이미지
            └── ...
"""

import datetime
import hashlib
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from helper_dev_utils import get_auto_logger
from PIL import Image

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.history.HistoryManager import HistoryManager
from src.history.HistoryRecord import HistoryRecord

logger = get_auto_logger(log_level=logging.DEBUG)

# ============================================================================
# 파일 기반 히스토리 매니저 클래스
# ============================================================================


class FileHistoryManager(HistoryManager):
    """파일 시스템에 영구 저장하는 히스토리 매니저

    이미지와 메타데이터를 파일로 저장하고 불러옵니다.
    """

    def __init__(
        self,
        save_dir: str = "./history",
        max_records: int = 100,
        auto_save: bool = True,
    ):
        """
        Args:
            save_dir: 저장 디렉토리 경로
            max_records: 최대 저장 가능한 레코드 수
            auto_save: True일 경우 레코드 추가 시 자동 저장
        """
        super().__init__(max_records)
        self.save_dir = Path(save_dir)
        self.auto_save = auto_save

        # 디렉토리 생성
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.save_dir / "images"
        self.images_dir.mkdir(exist_ok=True)

        # 기존 데이터 로드
        self._load_from_disk()

    def add_record(
        self,
        canvas_image: np.ndarray,
        preprocessed_image: np.ndarray,
        predicted_label: int,
        confidence: float,
        probabilities: np.ndarray,
        image_hash: str,
        notes: Optional[str] = None,
    ) -> HistoryRecord:
        """새로운 예측 기록을 추가하고 저장합니다.

        Args:
            canvas_image: 원본 캔버스 이미지
            preprocessed_image: 전처리된 이미지
            predicted_label: 예측된 라벨
            confidence: 신뢰도
            probabilities: 확률 배열
            image_hash: 이미지 해시
            notes: 추가 메모

        Returns:
            생성된 히스토리 레코드
        """
        record = super().add_record(
            canvas_image,
            preprocessed_image,
            predicted_label,
            confidence,
            probabilities,
            image_hash,
            notes,
        )

        if self.auto_save:
            self._save_record_to_disk(record)
            self._save_metadata()

        return record

    def find_by_hash(self, image_hash: str) -> Optional[HistoryRecord]:
        """이미지 해시로 레코드를 검색합니다.

        Args:
            image_hash: 검색할 이미지 해시

        Returns:
            발견된 레코드 또는 None
        """
        for record in self._records:
            if record.image_hash == image_hash:
                return record
        return None

    def _save_record_to_disk(self, record: HistoryRecord) -> None:
        """레코드의 이미지를 디스크에 저장합니다.

        캔버스 이미지와 전처리 이미지를 PNG 형식으로 저장합니다.
        파일명은 "canvas_{record_id}.png", "preprocessed_{record_id}.png" 형식입니다.

        Args:
            record: 저장할 레코드
        """
        record_id = record.record_id

        # 캔버스 이미지 저장
        canvas_path = self.images_dir / f"canvas_{record_id}.png"
        canvas_img = Image.fromarray(record.canvas_image.astype(np.uint8))
        canvas_img.save(canvas_path)

        # 전처리 이미지 저장
        preprocessed_path = self.images_dir / f"preprocessed_{record_id}.png"
        preprocessed_img = Image.fromarray(record.preprocessed_image.astype(np.uint8))
        preprocessed_img.save(preprocessed_path)

    def _save_metadata(self) -> None:
        """메타데이터를 JSON 파일로 저장합니다.

        모든 레코드의 메타정보(레이블, 확률, 타임스탬프 등)를 JSON으로 저장합니다.
        이미지 데이터는 별도 PNG 파일로 저장되며 JSON에는 포함되지 않습니다.
        """
        metadata = {
            "counter": self._counter,
            "records": [r.to_dict() for r in self._records],
        }

        metadata_path = self.save_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def _load_from_disk(self) -> None:
        """디스크에서 저장된 데이터를 로드합니다.

        metadata.json과 이미지 파일들을 읽어서 메모리에 로드합니다.
        앱 시작 시 자동으로 호출되어 이전 세션의 기록을 복원합니다.
        """
        metadata_path = self.save_dir / "metadata.json"

        if not metadata_path.exists():
            return

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self._counter = metadata.get("counter", 0)
        records_data = metadata.get("records", [])

        for record_data in records_data:
            record_id = record_data["record_id"]

            # 이미지 로드
            canvas_path = self.images_dir / f"canvas_{record_id}.png"
            preprocessed_path = self.images_dir / f"preprocessed_{record_id}.png"

            if not canvas_path.exists() or not preprocessed_path.exists():
                continue

            canvas_image = np.array(Image.open(canvas_path))
            preprocessed_image = np.array(Image.open(preprocessed_path))

            record = HistoryRecord(
                record_id=record_data["record_id"],
                canvas_image=canvas_image,
                preprocessed_image=preprocessed_image,
                predicted_label=record_data["predicted_label"],
                confidence=record_data["confidence"],
                probabilities=np.array(record_data["probabilities"]),
                timestamp=record_data["timestamp"],
                image_hash=record_data.get("image_hash", ""),
                notes=record_data.get("notes"),
            )

            self._records.append(record)

    def clear_all(self) -> None:
        """모든 히스토리와 저장된 파일을 삭제합니다.

        메모리의 레코드뿐만 아니라 디스크의 이미지 파일과
        메타데이터 JSON 파일도 함께 삭제합니다.
        """
        super().clear_all()

        # 이미지 파일 삭제
        for img_file in self.images_dir.glob("*.png"):
            img_file.unlink()

        # 메타데이터 파일 삭제
        metadata_path = self.save_dir / "metadata.json"
        if metadata_path.exists():
            metadata_path.unlink()
