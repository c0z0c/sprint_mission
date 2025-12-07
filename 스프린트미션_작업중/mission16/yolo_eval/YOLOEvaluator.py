import os
import pandas as pd
import numpy as np
import time
import yaml
import torch

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
from .EvaluationMetrics import EvaluationMetrics
from .QuantizedModelWrapper import QuantizedModelWrapper

class YOLOEvaluator:
    """YOLO 모델 평가 클래스"""
    
    def __init__(
        self, 
        model_path: str,
        yaml_path: str,
        model_name: Optional[str] = None,
        device: str = 'cuda',
        verbose: bool = False,
        model_type: Literal['yolo_pt', 'torch_state_dict', 'quantized', 'openvino', 'onnx'] = 'yolo_pt',
        model_config: Optional[str] = None
    ):
        """
        Args:
            model_path: YOLO 모델 파일 경로
            yaml_path: 데이터셋 YAML 파일 경로
            model_name: 모델 이름 (None이면 파일명 사용)
            device: 사용할 디바이스 ('cuda' or 'cpu')
            verbose: 상세 출력 여부
            model_type: 모델 타입
                - 'yolo_pt': Ultralytics YOLO .pt 파일 (기본값)
                - 'torch_state_dict': PyTorch state_dict .pth 파일
                - 'quantized': 양자화된 모델 (torch.load로 직접 로드)
                - 'openvino': OpenVINO IR 형식 (.xml 파일 또는 디렉토리)
                - 'onnx': ONNX 모델 (.onnx 파일, FP32/FP16/INT8 QDQ 지원)
            model_config: 모델 구조 정의 파일 경로 (torch_state_dict 사용 시 필수, 예: 'yolov8m.yaml')
        """
        self.model_path = model_path
        self.yaml_path = yaml_path
        self.model_name = model_name or Path(model_path).stem
        self.device = device
        self.verbose = verbose
        self.model_type = model_type
        self.model_config = model_config
        
        # 모델 로드
        self._load_model()
        
        # 데이터셋 경로 로드
        self._load_dataset_paths()
        
        # 결과 저장
        self.predictions: List[PredictionResult] = []
        self.metrics: Optional[EvaluationMetrics] = None
        
    def _load_model(self):
        """YOLO 모델 로드 (타입별 처리)"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없음: {self.model_path}")

        logger.info(f"모델 로딩 중 (타입: {self.model_type}): {self.model_path}")

        if self.model_type == 'yolo_pt':
            # 기본 Ultralytics YOLO .pt 파일
            self.model = YOLO(self.model_path)
            
        elif self.model_type == 'onnx':
            # ONNX 모델은 QuantizedModelWrapper로 처리
            logger.info("ONNX 모델을 QuantizedModelWrapper로 로딩")
            
            model_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)
            logger.info(f"ONNX 모델 크기: {model_size_mb:.2f} MB")
            
            # QuantizedModelWrapper로 래핑 (ONNX Runtime 사용)
            self.model = QuantizedModelWrapper(
                quantized_model=self.model_path,
                yaml_path=self.yaml_path,
                device=self.device,
                verbose=self.verbose,
                model_type='onnx'
            )
            
        elif self.model_type == 'torch_state_dict':
            # PyTorch state_dict .pth 파일
            if not self.model_config:
                raise ValueError("torch_state_dict 타입은 model_config 파라미터가 필수입니다.")

            # state_dict 로드 (PyTorch 2.6+ 호환성을 위해 weights_only=False)
            loaded_obj = torch.load(self.model_path, map_location=self.device, weights_only=False)

            # 타입 확인: state_dict인지 전체 모델 객체인지 판별
            if isinstance(loaded_obj, dict):
                # state_dict인 경우 (정상 케이스)
                logger.info(f"state_dict 형식 감지. 모델 구조 로드: {self.model_config}")
                # 1. 빈 YOLO 모델 인스턴스 생성 (구조 정의)
                yolo_instance = YOLO(self.model_config)

                # 2. 모델에 가중치 적용
                yolo_instance.model.load_state_dict(loaded_obj)  # type: ignore
                yolo_instance.model.eval()  # type: ignore

                # 3. YOLO 인스턴스의 ckpt_path를 None으로 설정하여 .val() 호출 시 경로 체크 우회
                yolo_instance.ckpt_path = None
                yolo_instance.model_name = self.model_name

                self.model = yolo_instance
                logger.info("state_dict 로드 및 적용 완료")
            else:
                # 전체 모델 객체인 경우
                logger.warning(
                    f"torch_state_dict 타입으로 지정했지만 전체 모델 객체가 감지되었습니다. "
                    f"model_type='quantized' 사용을 권장합니다."
                )
                loaded_obj.eval()
                # YOLO wrapper로 감싸기
                self.model = QuantizedModelWrapper(
                    quantized_model=loaded_obj,
                    yaml_path=self.yaml_path,
                    device=self.device,
                    verbose=self.verbose
                )
                logger.info("전체 모델 객체로 로드 및 래핑 완료")
            
        elif self.model_type == 'quantized':
            # 양자화된 모델 로드 (PyTorch 2.6+ 호환성을 위해 weights_only=False)
            loaded_obj = torch.load(self.model_path, map_location=self.device, weights_only=False)

            # 타입 확인: state_dict인지 전체 모델 객체인지 판별
            if isinstance(loaded_obj, dict):
                # state_dict인 경우: model_config가 필요함
                if not self.model_config:
                    raise ValueError(
                        "양자화 모델이 state_dict 형식으로 저장되어 있습니다. "
                        "model_config 파라미터로 모델 구조 파일(예: 'yolov8m.yaml')을 지정해야 합니다."
                    )

                logger.info(f"state_dict 형식 감지. 모델 구조 로드: {self.model_config}")
                # 모델 구조 생성
                yolo_instance = YOLO(self.model_config)

                # state_dict 적용
                yolo_instance.model.load_state_dict(loaded_obj)  # type: ignore
                quantized_model = yolo_instance.model
                quantized_model.eval()  # type: ignore
                logger.info("state_dict 로드 및 적용 완료")
            else:
                # 전체 모델 객체인 경우
                quantized_model = loaded_obj
                quantized_model.eval()
                logger.info("전체 모델 객체 로드 완료")

            # YOLO 래퍼로 감싸기 (ultralytics API 호환성)
            self.model = QuantizedModelWrapper(
                quantized_model=quantized_model,
                yaml_path=self.yaml_path,
                device=self.device,
                verbose=self.verbose
            )
            logger.info("양자화 모델 래핑 완료")

        elif self.model_type == 'openvino':
            # OpenVINO IR 형식 모델
            # .xml 파일이 직접 전달된 경우 부모 디렉토리로 변경
            if self.model_path.endswith('.xml') and '_openvino_model' in self.model_path:
                original_path = self.model_path
                self.model_path = str(Path(self.model_path).parent)
                logger.info(f"OpenVINO .xml 파일 감지. 디렉토리로 변경: {original_path} -> {self.model_path}")

            # Ultralytics YOLO로 로드 (OpenVINO 자동 감지)
            self.model = YOLO(self.model_path)
            logger.info("OpenVINO 모델 로드 완료")

        else:
            raise ValueError(f"지원하지 않는 model_type: {self.model_type}")

        logger.info(f"모델 로드 완료: {self.model_name}")
        
    def _load_dataset_paths(self):
        """YAML 파일에서 데이터셋 경로 로드"""
        with open(self.yaml_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        self.base_path = yaml_config['path']
        self.test_images_path = os.path.join(self.base_path, yaml_config['test'])
        self.test_labels_path = os.path.join(self.base_path, 'labels', 'test')
        
        logger.info(f"데이터셋 경로: {self.base_path}")
        logger.info(f"테스트 이미지: {self.test_images_path}")
        logger.info(f"테스트 라벨: {self.test_labels_path}")
        
        
    def _get_tqdm_kwargs(self) -> Dict:
        """Widget 오류를 방지하는 안전한 tqdm 설정"""
        return {
            'disable': False,
            'leave': True,
            'file': sys.stdout,
            'ascii': True,  # ASCII 문자만 사용
            'dynamic_ncols': False,
    #        'ncols': 80  # 고정 폭
        }        
        
    def validate(
        self, 
        split: str = 'test',
        imgsz: int = 640,
        batch: int = 16,
        conf: float = 0.25,
        iou: float = 0.75,
        project: Optional[str] = None,
        name: Optional[str] = None,
        save: bool = True
    ) -> EvaluationMetrics:
        """
        모델 검증 실행
        
        Args:
            split: 데이터 분할 ('train', 'val', 'test')
            imgsz: 이미지 크기
            batch: 배치 크기
            conf: 신뢰도 임계값
            iou: IoU 임계값
            project: 결과 저장 기본 디렉터리 (None이면 'runs' 사용)
            name: 실험 이름/하위 폴더 (None이면 자동 생성)
            save: 결과 저장 여부 (False 시 디스크 저장 안 함)
            
        Returns:
            EvaluationMetrics 객체
            
        Example:
            >>> # 저장 비활성화
            >>> evaluator.validate(save=False)
            >>> 
            >>> # 커스텀 경로에 저장: results/exp1/ 에 저장됨
            >>> evaluator.validate(project='results', name='exp1')
        """
        logger.info(f"모델 검증 시작: split={split}, imgsz={imgsz}, batch={batch}")
        logger.info(f"모델 검증 시작: self.model_type={self.model_type}")
        logger.info(f"모델 검증 시작: self.model_config={self.model_config}")

        start_time = time.time()

        # YOLO validation 실행
        # torch_state_dict, quantized는 model 객체를 직접 전달
        # yolo_pt, onnx, openvino는 YOLO 객체로 직접 실행
        if self.model_type in ['torch_state_dict', 'quantized']:
            metrics = self.model.val(
                data=self.yaml_path,
                split=split,
                imgsz=imgsz,
                batch=batch,
                conf=conf,
                iou=iou,
                project=project,
                name=name,
                save=save,
                verbose=self.verbose,
                model=self.model.model  # 모델 객체 직접 전달
            )
        else:
            # yolo_pt, onnx, openvino는 파일 경로 사용 (정상 동작)
            metrics = self.model.val(
                data=self.yaml_path,
                split=split,
                imgsz=imgsz,
                batch=batch,
                conf=conf,
                iou=iou,
                project=project,
                name=name,
                save=save,
                verbose=self.verbose
            )
        
        inference_time = (time.time() - start_time) / len(self._get_image_files())
        
        # 메트릭 객체 생성
        eval_metrics = EvaluationMetrics(
            map50=float(metrics.box.map50),
            map50_95=float(metrics.box.map),
            precision=float(metrics.box.mp),
            recall=float(metrics.box.mr),
            inference_time=inference_time,
            model_name=self.model_name
        )
        
        self.metrics = eval_metrics
        logger.info("모델 검증 완료")
        
        return eval_metrics
    
    def predict_on_images(
        self, 
        max_images: Optional[int] = None,
        conf: float = 0.25
    ) -> List[PredictionResult]:
        """
        테스트 이미지에 대한 예측 실행
        
        Args:
            max_images: 최대 이미지 수 (None이면 전체)
            conf: 신뢰도 임계값
            
        Returns:
            PredictionResult 리스트
        """
        logger.info(f"이미지 예측 시작: max_images={max_images}")
        
        image_files = self._get_image_files()
        if max_images:
            image_files = image_files[:max_images]
        
        predictions = []
        pbar = tqdm(image_files, desc="이미지 예측 중", **self._get_tqdm_kwargs())
        
        import time
        for img_file in pbar:
            img_path = os.path.join(self.test_images_path, img_file)
            image_name = os.path.splitext(img_file)[0]
            
            # Ground Truth 로드
            gt_boxes = self._load_ground_truth(image_name)
            
            # 예측 실행
            start_time = time.time()
            results = self.model(img_path, conf=conf, verbose=False)
            inference_time = time.time() - start_time
            
            # 예측 결과 파싱
            pred_boxes = self._parse_predictions(results)
            
            # 결과 저장
            pred_result = PredictionResult(
                image_name=image_name,
                gt_boxes=gt_boxes,
                pred_boxes=pred_boxes,
                inference_time=inference_time
            )
            predictions.append(pred_result)
        
        self.predictions = predictions
        logger.info(f"예측 완료: {len(predictions)}개 이미지")
        
        return predictions
    
    def _get_image_files(self) -> List[str]:
        """이미지 파일 목록 가져오기"""
        return [f for f in os.listdir(self.test_images_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    def _load_ground_truth(self, image_name: str) -> List[Dict]:
        """Ground Truth 라벨 로드"""
        label_path = os.path.join(self.test_labels_path, f"{image_name}.txt")
        gt_boxes = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        gt_boxes.append({
                            'class': int(parts[0]),
                            'x_center': float(parts[1]),
                            'y_center': float(parts[2]),
                            'width': float(parts[3]),
                            'height': float(parts[4])
                        })
        
        return gt_boxes
    
    def _parse_predictions(self, results) -> List[Dict]:
        """예측 결과 파싱"""
        pred_boxes = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                pred_boxes.append({
                    'class': int(boxes.cls[i].item()),
                    'conf': float(boxes.conf[i].item()),
                    'xyxy': boxes.xyxy[i].cpu().numpy().tolist()
                })
        
        return pred_boxes
    
    def compute_prediction_stats(self) -> EvaluationMetrics:
        """예측 통계 계산"""
        if not self.predictions:
            raise ValueError("예측 결과가 없습니다. predict_on_images()를 먼저 실행하세요.")
        
        total_images = len(self.predictions)
        images_with_gt = sum(1 for p in self.predictions if p.gt_count > 0)
        images_with_pred = sum(1 for p in self.predictions if p.pred_count > 0)
        avg_gt_boxes = sum(p.gt_count for p in self.predictions) / total_images
        avg_pred_boxes = sum(p.pred_count for p in self.predictions) / total_images
        avg_inference_time = sum(p.inference_time for p in self.predictions) / total_images
        
        # 기존 metrics 업데이트 또는 새로 생성
        if self.metrics:
            self.metrics.total_images = total_images
            self.metrics.images_with_gt = images_with_gt
            self.metrics.images_with_pred = images_with_pred
            self.metrics.avg_gt_boxes = avg_gt_boxes
            self.metrics.avg_pred_boxes = avg_pred_boxes
            self.metrics.inference_time = avg_inference_time
        else:
            self.metrics = EvaluationMetrics(
                model_name=self.model_name,
                total_images=total_images,
                images_with_gt=images_with_gt,
                images_with_pred=images_with_pred,
                avg_gt_boxes=avg_gt_boxes,
                avg_pred_boxes=avg_pred_boxes,
                inference_time=avg_inference_time
            )
        
        return self.metrics
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """예측 결과를 DataFrame으로 반환"""
        if not self.predictions:
            raise ValueError("예측 결과가 없습니다.")
        
        data = []
        for pred in self.predictions:
            data.append({
                'image_name': pred.image_name,
                'gt_count': pred.gt_count,
                'pred_count': pred.pred_count,
                'inference_time_ms': pred.inference_time * 1000
            })
        
        return pd.DataFrame(data)

logger.info("YOLOEvaluator 클래스 로드 완료")
