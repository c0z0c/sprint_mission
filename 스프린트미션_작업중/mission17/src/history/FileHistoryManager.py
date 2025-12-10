# -*- coding: utf-8 -*-
"""MNIST 예측 히스토리 저장소 API - 클래스 기반 설계

이 모듈은 예측 기록을 저장, 조회, 관리하는 기능을 제공합니다.
"""

import datetime
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from .HistoryRecord import HistoryRecord
from .HistoryManager import HistoryManager

from helper_dev_utils import get_auto_logger
logger = get_auto_logger()

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
        auto_save: bool = True
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
        notes: Optional[str] = None
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
            notes
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
        """메타데이터를 JSON 파일로 저장합니다."""
        metadata = {
            "counter": self._counter,
            "records": [r.to_dict() for r in self._records]
        }

        metadata_path = self.save_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def _load_from_disk(self) -> None:
        """디스크에서 저장된 데이터를 로드합니다."""
        metadata_path = self.save_dir / "metadata.json"

        if not metadata_path.exists():
            return

        with open(metadata_path, 'r', encoding='utf-8') as f:
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
                notes=record_data.get("notes")
            )

            self._records.append(record)

    def clear_all(self) -> None:
        """모든 히스토리와 저장된 파일을 삭제합니다."""
        super().clear_all()

        # 이미지 파일 삭제
        for img_file in self.images_dir.glob("*.png"):
            img_file.unlink()

        # 메타데이터 파일 삭제
        metadata_path = self.save_dir / "metadata.json"
        if metadata_path.exists():
            metadata_path.unlink()

