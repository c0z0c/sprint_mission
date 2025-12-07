import pandas as pd
import numpy as np
import time
import yaml

try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal
from pathlib import Path
from helper_utils.helper_logger import *
from ultralytics import YOLO

from .EvaluationMetrics import EvaluationMetrics
from .PredictionResult import PredictionResult
from .YOLOEvaluator import YOLOEvaluator

class YOLOEvaluationPipeline:
    """YOLO 모델 평가 파이프라인"""
    
    def __init__(self, yaml_path: str, device: str = 'cuda'):
        """
        Args:
            yaml_path: 데이터셋 YAML 파일 경로
            device: 사용할 디바이스
        """
        self.yaml_path = yaml_path
        self.device = device
        self.evaluators: Dict[str, YOLOEvaluator] = {}
        self.results: Dict[str, EvaluationMetrics] = {}
        
    def add_model(
        self, 
        model_path: str, 
        model_name: Optional[str] = None,
        verbose: bool = False,
        model_type: Literal['yolo_pt', 'torch_state_dict', 'quantized', 'openvino', 'onnx'] = 'yolo_pt',
        model_config: Optional[str] = None
    ) -> 'YOLOEvaluationPipeline':
        """
        평가할 모델 추가
        
        Args:
            model_path: 모델 파일 경로
            model_name: 모델 이름
            verbose: 상세 출력 여부
            model_type: 모델 타입 ('yolo_pt', 'torch_state_dict', 'quantized', 'openvino', 'onnx')
            model_config: 모델 구조 정의 파일 (torch_state_dict 사용 시 필수)

        Returns:
            self (체이닝 지원)

        Example:
            # 1. 기본 YOLO .pt 모델 (기존 방식)
            pipeline.add_model(
                model_path="yolov8m.pt",
                model_name="baseline"
            )

            # 2. PyTorch state_dict .pth 모델
            pipeline.add_model(
                model_path="mission_16_yolo.pth",
                model_name="torch_saved",
                model_type='torch_state_dict',
                model_config='yolov8m.yaml'  # 필수
            )

            # 3. 양자화 모델
            pipeline.add_model(
                model_path="quantized_model.pth",
                model_name="int8_quantized",
                model_type='quantized'
            )

            # 4. OpenVINO 모델
            pipeline.add_model(
                model_path="best_int8_openvino_model",  # 디렉토리 또는 .xml
                model_name="int8_openvino",
                model_type='openvino'
            )

            # 5. ONNX 모델 (FP32/FP16/INT8 QDQ)
            pipeline.add_model(
                model_path="yolov8m_int8_qdq.onnx",
                model_name="int8_onnx",
                model_type='onnx'
            )
        """
        if model_name is None:
            model_name = Path(model_path).stem

        logger.info(f"모델 추가: {model_name} (타입: {model_type})")

        evaluator = YOLOEvaluator(
            model_path=model_path,
            yaml_path=self.yaml_path,
            model_name=model_name,
            device=self.device,
            verbose=verbose,
            model_type=model_type,
            model_config=model_config
        )
        
        self.evaluators[model_name] = evaluator
        return self
    
    def run_validation(
        self,
        model_names: Optional[List[str]] = None,
        **val_kwargs
    ) -> Dict[str, EvaluationMetrics]:
        """
        검증 실행
        
        Args:
            model_names: 평가할 모델 이름 리스트 (None이면 전체)
            **val_kwargs: validate() 메서드에 전달할 인자
            
        Returns:
            모델별 평가 메트릭 딕셔너리
        """
        if model_names is None:
            model_names = list(self.evaluators.keys())
        
        print("=" * 80)
        print("모델 검증 시작")
        print("=" * 80)
        
        for model_name in model_names:
            if model_name not in self.evaluators:
                logger.warning(f"모델을 찾을 수 없음: {model_name}")
                continue
            
            print(f"\n[{model_name}] 검증 중...")
            evaluator = self.evaluators[model_name]
            metrics = evaluator.validate(**val_kwargs)
            self.results[model_name] = metrics
            
        return self.results
    
    def run_predictions(
        self,
        model_names: Optional[List[str]] = None,
        max_images: Optional[int] = None,
        conf: float = 0.25
    ) -> Dict[str, List[PredictionResult]]:
        """
        예측 실행
        
        Args:
            model_names: 평가할 모델 이름 리스트 (None이면 전체)
            max_images: 최대 이미지 수
            conf: 신뢰도 임계값
            
        Returns:
            모델별 예측 결과 딕셔너리
        """
        if model_names is None:
            model_names = list(self.evaluators.keys())
        
        predictions = {}
        
        for model_name in model_names:
            if model_name not in self.evaluators:
                logger.warning(f"모델을 찾을 수 없음: {model_name}")
                continue
            
            print(f"[{model_name}] 예측 중...")
            evaluator = self.evaluators[model_name]
            preds = evaluator.predict_on_images(max_images=max_images, conf=conf)
            predictions[model_name] = preds
            
            # 통계 계산
            evaluator.compute_prediction_stats()
        
        return predictions
    
    
    def run_full_evaluation(
        self,
        model_names: Optional[List[str]] = None,
        val_kwargs: Optional[Dict] = None,
        pred_max_images: Optional[int] = None,
        pred_conf: float = 0.25
    ) -> Dict[str, EvaluationMetrics]:
        """
        전체 평가 실행 (검증 + 예측)
        
        Args:
            model_names: 평가할 모델 이름 리스트
            val_kwargs: 검증 인자
            pred_max_images: 예측 최대 이미지 수
            pred_conf: 예측 신뢰도 임계값
            
        Returns:
            모델별 평가 메트릭 딕셔너리
        """
        if val_kwargs is None:
            val_kwargs = {}
        
        # 검증 실행
        self.run_validation(model_names=model_names, **val_kwargs)
        
        # 예측 실행
        self.run_predictions(
            model_names=model_names, 
            max_images=pred_max_images,
            conf=pred_conf
        )
        
        return self.results
    
    def print_summary(self, model_names: Optional[List[str]] = None):
        """평가 결과 요약 출력"""
        if model_names is None:
            model_names = list(self.evaluators.keys())
        
        for model_name in model_names:
            evaluator = self.evaluators.get(model_name)
            if evaluator and evaluator.metrics:
                evaluator.metrics.print_summary()
    
    def get_comparison_dataframe(self) -> pd.DataFrame:
        """모델 비교 DataFrame 생성"""
        if not self.results:
            raise ValueError("평가 결과가 없습니다.")
        
        data = []
        for model_name, metrics in self.results.items():
            row = metrics.to_dict()
            data.append(row)
        
        df = pd.DataFrame(data)
        return df.sort_values('mAP50-95', ascending=False)
    
    def save_results(self, output_path: str):
        """평가 결과 저장"""
        df = self.get_comparison_dataframe()
        
        # CSV 저장
        csv_path = output_path if output_path.endswith('.csv') else f"{output_path}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"결과 저장: {csv_path}")
        
        # JSON 저장
        json_path = csv_path.replace('.csv', '.json')
        df.to_json(json_path, orient='records', indent=2, force_ascii=False)
        logger.info(f"결과 저장: {json_path}")
        
        return csv_path, json_path

logger.info("YOLOEvaluationPipeline 클래스 로드 완료")
