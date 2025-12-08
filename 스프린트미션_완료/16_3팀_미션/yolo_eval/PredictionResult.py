from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time
from helper_utils.helper_logger import *


@dataclass
class PredictionResult:
    """개별 이미지 예측 결과"""
    image_name: str
    gt_boxes: List[Dict] = field(default_factory=list)
    pred_boxes: List[Dict] = field(default_factory=list)
    inference_time: float = 0.0
    
    @property
    def gt_count(self) -> int:
        return len(self.gt_boxes)
    
    @property
    def pred_count(self) -> int:
        return len(self.pred_boxes)

logger.info("PredictionResult 클래스 로드 완료")
