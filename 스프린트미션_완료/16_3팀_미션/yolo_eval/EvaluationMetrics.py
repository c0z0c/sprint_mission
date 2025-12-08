from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time
from helper_utils.helper_logger import *

_print = print

@dataclass
class EvaluationMetrics:
    """평가 지표를 저장하는 데이터클래스"""
    map50: float = 0.0
    map50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    inference_time: float = 0.0
    model_name: str = ""
    total_images: int = 0
    images_with_gt: int = 0
    images_with_pred: int = 0
    avg_gt_boxes: float = 0.0
    avg_pred_boxes: float = 0.0
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            'model_name': self.model_name,
            'mAP50': self.map50,
            'mAP50-95': self.map50_95,
            'precision': self.precision,
            'recall': self.recall,
            'inference_time_ms': self.inference_time * 1000,
            'total_images': self.total_images,
            'images_with_gt': self.images_with_gt,
            'images_with_pred': self.images_with_pred,
            'avg_gt_boxes': self.avg_gt_boxes,
            'avg_pred_boxes': self.avg_pred_boxes
        }
    
    def print_summary(self):
        """평가 결과 요약 출력"""
        _print("=" * 80)
        _print(f"평가 결과: {self.model_name}")
        _print("=" * 80)
        _print(f"mAP50: {self.map50:.4f}")
        _print(f"mAP50-95: {self.map50_95:.4f}")
        _print(f"정밀도(Precision): {self.precision:.4f}")
        _print(f"재현율(Recall): {self.recall:.4f}")
        _print(f"추론 시간(평균): {self.inference_time * 1000:.2f}ms")
        _print("-" * 80)
        _print(f"테스트된 총 이미지 수: {self.total_images}")
        _print(f"GT 박스가 있는 이미지 수: {self.images_with_gt}")
        _print(f"예측이 있는 이미지 수: {self.images_with_pred}")
        _print(f"이미지당 평균 GT 박스 수: {self.avg_gt_boxes:.2f}")
        _print(f"이미지당 평균 예측 수: {self.avg_pred_boxes:.2f}")
        _print("=" * 80)

logger.info("EvaluationMetrics 클래스 로드 완료")
