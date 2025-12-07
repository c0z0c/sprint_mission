"""
YOLO API와 호환되는 양자화된 PyTorch 모델 래퍼

간단 설명:
 이 모듈은 양자화된(torch.quantization 적용) PyTorch detection 모델을
 Ultralytics YOLO 스타일 인터페이스(.__call__, .val)로 래핑합니다.
 주로 양자화된 모델을 기존 YOLO 기반 평가/추론 파이프라인에서
 재사용할 때 사용합니다.

입력(주요 생성자 인자):
 - quantized_model: torch.nn.Module, 양자화된 모델(추론 모드 가능)
 - yaml_path: str 또는 Path, 데이터셋 설정 YAML 경로 (필수: path, nc, names)
 - device: 'cuda' 또는 'cpu' (기본 'cuda')
 - verbose: bool (로그 상세화)

출력/동작:
 - __call__(source, conf, iou): 이미지 경로/리스트를 받아 ultralytics.engine.results.Results 리스트 반환
 - val(...): 간단한 검증 루프를 수행하여 QuantizedMetricsWrapper 반환
 - predict(...)는 __call__의 별칭

YAML 예시 (필수 필드):
 path: /path/to/dataset
 nc: 3
 names: ['person', 'car', 'bike']

제약 및 주의사항:
 - .val()는 간단 구현으로 정확한 mAP 계산을 수행하지 않습니다.
 - ultralytics 라이브러리의 일부 기능(non_max_suppression, Results 등)에 의존합니다.
 - 모델 출력 형식은 [batch, boxes, features(=5+nc)] 형태를 권장합니다.
 - 변경 없이 문서화만 수정하십시오(기능 변경 없음).

간단 사용 예시:
 from QuantizedModelWrapper import QuantizedModelWrapper
 wrapper = QuantizedModelWrapper(quantized_model, 'data.yaml', device='cpu')
 results = wrapper('image.jpg', conf=0.25)

"""

import os
import torch
import yaml
import numpy as np
import cv2
from types import SimpleNamespace
from typing import Dict, List, Optional, Union, Literal
from pathlib import Path
from helper_utils.helper_logger import *

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("onnxruntime를 찾을 수 없습니다. ONNX 모델은 지원되지 않습니다.")

# Global NMS function cache
_non_max_suppression = None

def _get_nms_function():
    """
    NMS 함수를 다양한 경로에서 임포트 시도 (캐싱됨)

    Returns:
        non_max_suppression function or None
    """
    global _non_max_suppression

    if _non_max_suppression is not None:
        return _non_max_suppression

    import_attempts = [
        # Priority 1: Correct path for ultralytics 8.3.220+ (VERIFIED WORKING)
        ('ultralytics.utils.nms', 'non_max_suppression', lambda m: m),

        # Priority 2: Alternative attempt
        ('ultralytics.utils', 'ops', lambda m: m.non_max_suppression),

        # Priority 3: Legacy paths (backward compatibility)
        ('ultralytics.utils.ops', 'non_max_suppression', lambda m: m),
        ('ultralytics.ops', 'non_max_suppression', lambda m: m),
        ('ultralytics.models.yolo.detect.predict.nms', 'non_max_suppression', lambda m: m),
    ]

    for module_path, attr_name, extractor in import_attempts:
        try:
            module = __import__(module_path, fromlist=[attr_name])
            func = extractor(getattr(module, attr_name))
            _non_max_suppression = func
            logger.info(f"NMS 함수 임포트 성공: {module_path}.{attr_name}")
            return _non_max_suppression
        except (ImportError, AttributeError) as e:
            logger.debug(f"NMS 임포트 실패 ({module_path}): {e}")
            continue

    logger.error(
        "NMS 함수를 모든 경로에서 찾을 수 없습니다.\n"
        "시도한 경로:\n"
        "  1. ultralytics.utils.nms.non_max_suppression (8.3.220+)\n"
        "  2. ultralytics.utils.ops.non_max_suppression\n"
        "  3. ultralytics.ops.non_max_suppression (legacy)\n"
        "  4. ultralytics.models.yolo.detect.predict.nms (old)\n"
        "ultralytics 버전을 확인하거나 pip install --upgrade ultralytics를 시도하세요."
    )
    return None

class QuantizedModelWrapper:
    """
    양자화된 PyTorch 모델을 Ultralytics YOLO API 호환 인터페이스로 래핑하는 클래스.

    이 래퍼는 .val() 및 __call__() 메서드를 구현하여 YOLO 인터페이스와 일치시키므로,
    양자화된 모델을 YOLOEvaluator에서 원활하게 사용할 수 있습니다.
    """

    def __init__(self, quantized_model, yaml_path, device='cuda', verbose=False, 
                 model_type: Literal['pytorch', 'onnx'] = 'pytorch'):
        """
        양자화 모델 래퍼 초기화

        Args:
            quantized_model: PyTorch 양자화 모델 (torch.nn.Module) 또는 ONNX 모델 경로 (str)
            yaml_path: 데이터셋 YAML 설정 파일 경로
            device: 추론을 실행할 디바이스
            verbose: 상세 로깅 활성화
            model_type: 'pytorch' (기본값) 또는 'onnx'
        """
        self.model_type = model_type
        self.yaml_path = yaml_path
        self.device = device
        self.verbose = verbose
        self.names = None  # 클래스 이름 딕셔너리 {0: 'cat', 1: 'dog'}
        self.nc = None     # 클래스 수
        self.onnx_session = None
        self.input_name = None
        self.output_names = None
        
        # 모델 로드 (타입별 처리)
        if model_type == 'onnx':
            if not ONNX_AVAILABLE:
                raise ImportError("ONNX 모델을 사용하려면 'pip install onnxruntime'를 실행하세요.")
            self._load_onnx_model(quantized_model)
            self.model = None  # ONNX는 PyTorch 모델 없음
        else:
            self.model = quantized_model

        # 데이터셋 설정 로드
        self._load_dataset_config()

        # 모델 호환성 검증
        if model_type == 'pytorch':
            self._validate_model()
        else:
            self._validate_onnx_model()

        logger.info(f"QuantizedModelWrapper 초기화 완료 (타입: {model_type})")

    def _load_dataset_config(self):
        """YAML 파일에서 클래스 이름 및 설정 로드"""
        if not os.path.exists(self.yaml_path):
            raise FileNotFoundError(f"데이터셋 YAML을 찾을 수 없음: {self.yaml_path}")

        with open(self.yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        required_keys = ['path', 'nc', 'names']
        missing = [k for k in required_keys if k not in config]
        if missing:
            raise ValueError(f"데이터셋 YAML에 필수 키가 누락됨: {missing}")

        self.nc = config['nc']
        self.names = config['names']

        # names가 리스트인 경우 딕셔너리로 변환
        if isinstance(self.names, list):
            self.names = {i: name for i, name in enumerate(self.names)}
    
    def _load_onnx_model(self, model_path: str):
        """ONNX Runtime 세션 초기화"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX 모델 파일을 찾을 수 없음: {model_path}")
        
        logger.info(f"ONNX 모델 로딩: {model_path}")
        
        # ONNX Runtime 세션 옵션 설정
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3  # 에러만 출력 (0=VERBOSE, 3=ERROR)
        
        # Execution Provider 설정 (CPU 우선)
        providers = ['CPUExecutionProvider']
        if self.device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
        
        # ONNX 세션 생성
        self.onnx_session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )
        
        # 입출력 정보 추출
        self.input_name = self.onnx_session.get_inputs()[0].name
        self.output_names = [output.name for output in self.onnx_session.get_outputs()]
        
        input_shape = self.onnx_session.get_inputs()[0].shape
        logger.info(f"ONNX 입력: {self.input_name}, shape={input_shape}")
        logger.info(f"ONNX 출력: {self.output_names}")
        logger.info(f"Execution Provider: {self.onnx_session.get_providers()}")
    
    def _validate_onnx_model(self):
        """ONNX 모델 검증 (더미 입력으로 테스트)"""
        try:
            dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
            outputs = self.onnx_session.run(self.output_names, {self.input_name: dummy_input})
            
            output_shape = outputs[0].shape
            logger.info(f"ONNX 모델 검증 성공. 출력 shape: {output_shape}")
            
            if len(output_shape) != 3:
                logger.warning(
                    f"ONNX 출력 형태가 예상과 다름: {output_shape}. "
                    f"예상: (batch, num_predictions, 5+nc)"
                )
        except Exception as e:
            logger.error(f"ONNX 모델 검증 실패: {e}")
            raise

        logger.info(f"데이터셋 설정 로드 완료: {self.nc}개 클래스")

    def _validate_model(self):
        """
        양자화 모델이 호환되는지 검증
        더미 입력으로 순전파를 테스트하여 출력 형식이 올바른지 확인
        """
        try:
            dummy_input = torch.randn(1, 3, 640, 640).to(self.device)
            with torch.no_grad():
                output = self.model(dummy_input)

            # 예상 출력 형태: [batch, num_predictions, 5+nc]
            if isinstance(output, (tuple, list)):
                output = output[0]  # 첫 번째 출력 사용

            if not isinstance(output, torch.Tensor):
                raise ValueError(f"모델 출력은 텐서여야 하지만 {type(output)}를 받음")

            if len(output.shape) != 3:
                logger.warning(
                    f"모델 출력 형태가 예상과 다름: {output.shape}. "
                    f"예상: [batch, boxes, features]"
                )

            logger.info(f"모델 검증 통과. 출력 형태: {output.shape}")

        except Exception as e:
            raise RuntimeError(
                f"양자화 모델 검증 실패: {str(e)}\n"
                f"모델 아키텍처가 YOLO 출력 형식과 일치하는지 확인하세요."
            )

    def val(self, data=None, split='test', imgsz=640, batch=16,
            conf=0.25, iou=0.75, project=None, name=None,
            save=True, verbose=None, **kwargs):
        """
        Ultralytics 유틸리티를 사용하여 검증 실행

        Args:
            data: 데이터셋 YAML 경로 (None이면 self.yaml_path 사용)
            split: 데이터 분할 ('train', 'val', 'test')
            imgsz: 이미지 크기
            batch: 배치 크기
            conf: 신뢰도 임계값
            iou: IoU 임계값
            project: 결과 저장 디렉터리
            name: 실험 이름
            save: 결과 저장 여부
            verbose: 상세 출력 여부

        Returns:
            QuantizedMetricsWrapper (DetMetrics 형식과 호환)
        """
        # NMS 함수 가져오기
        non_max_suppression = _get_nms_function()
        if non_max_suppression is None:
            logger.error("NMS 함수를 찾을 수 없어 검증을 중단합니다.")
            return QuantizedMetricsWrapper({
                'map50': 0.0,
                'map50_95': 0.0,
                'precision': 0.0,
                'recall': 0.0
            })

        logger.info(f"양자화 모델 검증 시작: split={split}, imgsz={imgsz}, batch={batch}")

        # YAML 경로 설정
        yaml_file = data or self.yaml_path
        with open(yaml_file, 'r', encoding='utf-8') as f:
            data_dict = yaml.safe_load(f)

        # 데이터셋 경로 구성
        base_path = data_dict['path']
        split_path = data_dict.get(split, 'test')
        dataset_path = os.path.join(base_path, split_path)

        logger.info(f"데이터셋 경로: {dataset_path}")

        # YOLODataset 생성 - hyp 파라미터 대신 기본값 사용
        try:
            from ultralytics.data.dataset import YOLODataset
            from ultralytics.cfg import get_cfg
            from ultralytics.utils import DEFAULT_CFG

            # 기본 하이퍼파라미터 생성
            try:
                hyp = DEFAULT_CFG
            except:
                hyp = get_cfg()

            dataset = YOLODataset(
                img_path=dataset_path,
                imgsz=imgsz,
                batch_size=batch,
                augment=False,
                hyp=hyp,  # None 대신 기본값 사용
                rect=True,
                cache=False,
                single_cls=False,
                stride=32,
                pad=0.5,
                prefix=f'{split}: ',
                task='detect',
                classes=None,
                data=data_dict
            )
        except Exception as e:
            logger.error(f"데이터셋 로드 실패: {e}")
            # 간단한 메트릭 반환
            return QuantizedMetricsWrapper({
                'map50': 0.0,
                'map50_95': 0.0,
                'precision': 0.0,
                'recall': 0.0
            })

        # 간단한 검증 루프 (완전한 mAP 계산은 복잡하므로 기본 구현)
        if self.model_type == 'pytorch':
            self.model.eval()
        
        total_correct = 0
        total_predictions = 0
        total_targets = 0

        with torch.no_grad():
            for i in range(min(len(dataset), 100)):  # 샘플링
                try:
                    batch_data = dataset[i]
                    img = batch_data['img']

                    # 모델 타입에 따라 추론 실행
                    if self.model_type == 'onnx':
                        # ONNX Runtime 추론
                        if not isinstance(img, np.ndarray):
                            img = img.cpu().numpy() if isinstance(img, torch.Tensor) else np.array(img)
                        
                        img = img[np.newaxis, :].astype(np.float32)
                        onnx_outputs = self.onnx_session.run(self.output_names, {self.input_name: img})
                        predictions = torch.from_numpy(onnx_outputs[0]).to(self.device)
                    else:
                        # PyTorch 추론
                        if not isinstance(img, torch.Tensor):
                            img = torch.from_numpy(img)

                        # unsqueeze로 배치 차원 추가하고, device로 이동
                        img = img.unsqueeze(0).to(self.device)

                        # Dynamic quantization은 float32 입력을 기대
                        if img.dtype != torch.float32:
                            img = img.float()

                        # 순전파
                        predictions = self.model(img)

                    # NMS 적용
                    if isinstance(predictions, (tuple, list)):
                        predictions = predictions[0]

                    # 간단한 카운팅 (실제 mAP 계산은 더 복잡함)
                    predictions = non_max_suppression(
                        predictions,
                        conf_thres=conf,
                        iou_thres=iou,
                        classes=None,
                        agnostic=False,
                        multi_label=False,
                        max_det=300
                    )

                    if predictions and len(predictions[0]) > 0:
                        total_predictions += len(predictions[0])

                except Exception as e:
                    logger.warning(f"검증 중 오류 (인덱스 {i}): {e}")
                    continue

        logger.info(f"검증 완료: {total_predictions}개 예측")

        # 간단한 메트릭 반환 (실제 mAP 계산은 매우 복잡함)
        # 실제 프로덕션에서는 predict_on_images() 사용 권장
        metrics = {
            'map50': 0.0,  # 실제 계산 필요
            'map50_95': 0.0,  # 실제 계산 필요
            'precision': 0.0,  # 실제 계산 필요
            'recall': 0.0  # 실제 계산 필요
        }

        logger.warning(
            "양자화 모델의 .val() 메서드는 기본 구현입니다. "
            "정확한 메트릭을 위해서는 predict_on_images() 및 compute_prediction_stats() 사용을 권장합니다."
        )

        return QuantizedMetricsWrapper(metrics)

    def __call__(self, source, conf=0.25, iou=0.45, verbose=False, **kwargs):
        """
        이미지에 대한 예측 실행

        Args:
            source: 이미지 경로 또는 경로 리스트
            conf: 신뢰도 임계값
            iou: NMS를 위한 IoU 임계값
            verbose: 상세 출력 여부

        Returns:
            Results 객체 리스트 (.boxes 속성 포함)
        """
        # NMS 함수 가져오기
        non_max_suppression = _get_nms_function()
        if non_max_suppression is None:
            logger.error("NMS 함수를 찾을 수 없어 예측을 중단합니다.")
            return []

        try:
            from ultralytics.engine.results import Results, Boxes
        except ImportError as e:
            raise ImportError(
                f"Ultralytics 라이브러리 임포트 실패: {e}\n"
                f"pip install ultralytics를 실행하여 설치하세요."
            )

        # 소스를 리스트로 변환
        if isinstance(source, str):
            sources = [source]
        elif isinstance(source, (list, tuple)):
            sources = list(source)
        else:
            sources = [source]

        results = []

        if self.model_type == 'pytorch':
            self.model.eval()
            
        with torch.no_grad():
            for img_path in sources:
                try:
                    # 이미지 로드 및 전처리
                    orig_img = cv2.imread(str(img_path))
                    if orig_img is None:
                        logger.warning(f"이미지 로드 실패: {img_path}")
                        continue

                    img_tensor = self._preprocess_image(orig_img, imgsz=640)
                    
                    # 모델 타입에 따라 추론 실행
                    if self.model_type == 'onnx':
                        # ONNX Runtime 추론
                        img_np = img_tensor.unsqueeze(0).cpu().numpy()
                        onnx_outputs = self.onnx_session.run(self.output_names, {self.input_name: img_np})
                        predictions = torch.from_numpy(onnx_outputs[0]).to(self.device)
                    else:
                        # PyTorch 추론
                        img_tensor = img_tensor.unsqueeze(0).to(self.device)
                        predictions = self.model(img_tensor)

                    # 튜플/리스트인 경우 첫 번째 요소 사용
                    if isinstance(predictions, (tuple, list)):
                        predictions = predictions[0]

                    # NMS 적용
                    detections = non_max_suppression(
                        predictions,
                        conf_thres=conf,
                        iou_thres=iou,
                        classes=None,
                        agnostic=False,
                        multi_label=False,
                        max_det=300
                    )

                    # 첫 번째 배치 아이템 (단일 이미지)
                    det = detections[0] if len(detections) > 0 else torch.zeros((0, 6))

                    # CPU로 이동
                    det = det.cpu()

                    # 디버그: 첫 번째 이미지의 예측 결과 출력
                    if len(results) == 0 and len(det) > 0:
                        logger.info(f"[DEBUG] 첫 번째 이미지 예측:")
                        logger.info(f"  - 이미지 크기: {orig_img.shape}")
                        logger.info(f"  - 예측 박스 수: {len(det)}")
                        for i, box in enumerate(det[:3]):  # 최대 3개만 출력
                            x1, y1, x2, y2, conf, cls = box
                            logger.info(f"  - Box {i}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}], conf={conf:.3f}, class={int(cls)}")

                    # Boxes 객체 생성
                    if len(det) > 0:
                        boxes_obj = Boxes(
                            det,
                            orig_shape=orig_img.shape[:2]
                        )
                    else:
                        # 빈 박스
                        boxes_obj = Boxes(
                            torch.zeros((0, 6)),
                            orig_shape=orig_img.shape[:2]
                        )

                    # Results 객체 생성
                    result = Results(
                        orig_img=orig_img,
                        path=str(img_path),
                        names=self.names,
                        boxes=det
                    )

                    results.append(result)

                except Exception as e:
                    logger.error(f"예측 중 오류 ({img_path}): {e}")
                    # 빈 결과 추가
                    try:
                        empty_result = Results(
                            orig_img=np.zeros((640, 640, 3), dtype=np.uint8),
                            path=str(img_path),
                            names=self.names,
                            boxes=torch.zeros((0, 6))
                        )
                        results.append(empty_result)
                    except:
                        pass

        return results

    def _preprocess_image(self, img, imgsz=640):
        """
        YOLO 모델용 이미지 전처리
        - 레터박스를 사용한 리사이즈
        - BGR을 RGB로 변환
        - [0, 1]로 정규화
        - HWC를 CHW 형식으로 변환

        Args:
            img: OpenCV 이미지 (numpy array)
            imgsz: 대상 이미지 크기

        Returns:
            전처리된 이미지 텐서
        """
        try:
            from ultralytics.data.augment import LetterBox

            # 레터박스 리사이즈
            letterbox = LetterBox(imgsz, auto=True, stride=32)
            img_resized = letterbox(image=img)

        except ImportError:
            # Ultralytics 없이 간단한 리사이즈
            h, w = img.shape[:2]
            scale = imgsz / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img_resized = cv2.resize(img, (new_w, new_h))

            # 패딩 추가
            pad_h = (imgsz - new_h) // 2
            pad_w = (imgsz - new_w) // 2
            img_resized = cv2.copyMakeBorder(
                img_resized, pad_h, imgsz - new_h - pad_h,
                pad_w, imgsz - new_w - pad_w,
                cv2.BORDER_CONSTANT, value=(114, 114, 114)
            )

        # BGR을 RGB로 변환
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # 정규화 및 전치
        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_chw = img_normalized.transpose(2, 0, 1)  # HWC to CHW

        # 텐서로 변환
        img_tensor = torch.from_numpy(img_chw)

        return img_tensor

    def predict(self, source, **kwargs):
        """__call__의 별칭"""
        return self.__call__(source, **kwargs)


class QuantizedMetricsWrapper:
    """
    검증 메트릭을 Ultralytics DetMetrics API에 맞게 래핑
    .box 속성을 제공하며 .map50, .map, .mp, .mr 속성 포함
    """

    def __init__(self, metrics_dict):
        """
        메트릭 래퍼 생성

        Args:
            metrics_dict: 'map50', 'map50_95', 'precision', 'recall' 키를 가진 딕셔너리
        """
        self.box = SimpleNamespace(
            map50=metrics_dict.get('map50', 0.0),
            map=metrics_dict.get('map50_95', 0.0),
            mp=metrics_dict.get('precision', 0.0),
            mr=metrics_dict.get('recall', 0.0)
        )

    def results_dict(self):
        """메트릭을 딕셔너리로 반환"""
        return {
            'metrics/mAP50(B)': self.box.map50,
            'metrics/mAP50-95(B)': self.box.map,
            'metrics/precision(B)': self.box.mp,
            'metrics/recall(B)': self.box.mr
        }


logger.info("QuantizedModelWrapper 모듈 로드 완료")
