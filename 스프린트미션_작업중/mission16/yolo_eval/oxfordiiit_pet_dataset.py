"""Oxford-IIIT Pet Dataset을 YOLO 형식으로 변환하는 모듈.

Kaggle에서 데이터셋을 다운로드하고 YOLO 형식으로 변환한 후,
DataFrame으로 재로드하여 변환 정상 여부를 검증합니다.
"""

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import kagglehub
import pandas as pd
import yaml
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PetDatasetPaths:
    """Pet Dataset 경로 관리 클래스."""
    
    root: Path
    images: Path
    annotations: Path
    trainval_list: Path
    test_list: Path
    
    @classmethod  # 4칸 들여쓰기
    def from_kagglehub_path(cls, kagglehub_path: str) -> "PetDatasetPaths":  # 4칸 들여쓰기
        """Kagglehub 다운로드 경로로부터 PetDatasetPaths 생성.
        
        Args:
            kagglehub_path: kagglehub.dataset_download() 반환값
            
        Returns:
            PetDatasetPaths 인스턴스
        """
        root = Path(kagglehub_path)  # 8칸 들여쓰기
        return cls(  # 8칸 들여쓰기
            root=root,
            images=root / "images" / "images",
            annotations=root / "annotations" / "annotations" / "xmls",
            trainval_list=root / "annotations" / "annotations" / "trainval.txt",
            test_list=root / "annotations" / "annotations" / "test.txt"
        )

def parse_list_file(list_path: Path) -> List[Tuple[str, int, int, int]]:
    """리스트 파일을 파싱하여 이미지명, 클래스, species, breed 정보 추출.
    
    Args:
        list_path: trainval.txt 또는 test.txt 경로
        
    Returns:
        [(image_name, class_id, species, breed), ...] 리스트
    """
    data = []
    with open(list_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                image_name, class_id, species, breed = parts
                data.append((
                    image_name,
                    int(class_id) - 1,  # YOLO는 0부터 시작
                    int(species),
                    int(breed)
                ))
    return sorted(data, key=lambda x: x[0])  # 재현성을 위해 정렬


def parse_xml_annotation(xml_path: Path) -> Dict:
    """XML 어노테이션 파일에서 바운딩 박스 정보 추출.
    
    Args:
        xml_path: XML 어노테이션 파일 경로
        
    Returns:
        {"width": int, "height": int, "boxes": [(xmin, ymin, xmax, ymax), ...]}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    boxes = []
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        boxes.append((xmin, ymin, xmax, ymax))
    
    return {"width": width, "height": height, "boxes": boxes}


def convert_bbox_to_yolo(
    xmin: int, ymin: int, xmax: int, ymax: int,
    img_width: int, img_height: int
) -> Tuple[float, float, float, float]:
    """바운딩 박스를 YOLO 형식으로 변환.
    
    Args:
        xmin, ymin, xmax, ymax: 절대 좌표
        img_width, img_height: 이미지 크기
        
    Returns:
        (x_center, y_center, width, height) - 정규화된 값 (0~1)
    """
    x_center = ((xmin + xmax) / 2) / img_width
    y_center = ((ymin + ymax) / 2) / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    return x_center, y_center, width, height


def create_yolo_dataset(
    paths: PetDatasetPaths,
    output_dir: Path,
    class_names: List[str] = None,
    max_samples_per_split: Tuple[int, int, int] | None = None,
    label_mode: Literal["breed", "species"] = "breed"
) -> Path:
    """Oxford-IIIT Pet Dataset을 YOLO 형식으로 변환.
    
    Args:
        paths: PetDatasetPaths 인스턴스
        output_dir: YOLO 데이터셋 저장 경로
        class_names: 클래스명 리스트 (None이면 자동 생성)
        max_samples_per_split: 각 분할(train/val/test)의 최대 샘플 개수 (None이면 전체)
        label_mode: 라벨링 모드 ("breed": 품종 분류, "species": 개/고양이 분류)
        
    Returns:
        생성된 data.yaml 파일 경로
    """
    logger.info("YOLO 데이터셋 변환 시작")
    
    # 출력 디렉토리 생성
    output_dir = Path(output_dir)
    (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "test").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "test").mkdir(parents=True, exist_ok=True)
    
    # trainval 데이터 파싱
    trainval_data = parse_list_file(paths.trainval_list)
    # test_data = parse_list_file(paths.test_list)
    # trainval을 train/val로 분할 (8:2)
    # split_idx = int(len(trainval_data) * 0.7)
    # train_data = trainval_data[:split_idx]
    # val_data = trainval_data[split_idx:]
    # test_data = trainval_data[split_idx:]
    
    total = len(trainval_data)
    train_idx = int(total * 0.7)
    val_idx = int(total * 0.9)  # 누적 비율 90% -> val까지
    train_data = trainval_data[:train_idx]
    val_data = trainval_data[train_idx:val_idx]
    test_data = trainval_data[val_idx:]    
    
    # 샘플 개수 제한 (테스트용)
    if max_samples_per_split is not None:
        if max_samples_per_split[0] > 0:
            train_data = train_data[:max_samples_per_split[0]]
        if max_samples_per_split[1] > 0:
            val_data = val_data[:max_samples_per_split[1]]
        if max_samples_per_split[2] > 0:
            test_data = test_data[:max_samples_per_split[2]]
        logger.info(f"샘플 개수 제한 적용: 각 분할당 최대 {max_samples_per_split}개")
    
    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # 클래스명 자동 생성
    if class_names is None:
        if label_mode == "species":
            class_names = ["cat", "dog"]  # species: 1=cat, 2=dog -> 0=cat, 1=dog
        else:
            all_classes = set([item[1] for item in trainval_data + test_data])
            class_names = [f"class_{i}" for i in range(max(all_classes) + 1)]
    
    # 데이터셋 변환
    def process_split(data: List[Tuple], split_name: str):
        for image_name, class_id, species, _ in tqdm(data, desc=f"Processing {split_name}"):
            # 이미지 복사
            src_img = paths.images / f"{image_name}.jpg"
            dst_img = output_dir / "images" / split_name / f"{image_name}.jpg"
            
            if not src_img.exists():
                logger.warning(f"이미지 파일 없음: {src_img}")
                continue
            
            # 심볼릭 링크 또는 복사
            try:
                if not dst_img.exists():
                    dst_img.symlink_to(src_img)
            except OSError:
                import shutil
                shutil.copy2(src_img, dst_img)
            
            # XML 파싱
            xml_path = paths.annotations / f"{image_name}.xml"
            if not xml_path.exists():
                logger.warning(f"XML 파일 없음: {xml_path}")
                continue
            
            annotation = parse_xml_annotation(xml_path)
            
            # label_mode에 따라 라벨 결정
            label_id = species - 1 if label_mode == "species" else class_id
            
            # YOLO 라벨 생성
            label_path = output_dir / "labels" / split_name / f"{image_name}.txt"
            with open(label_path, 'w') as f:
                for box in annotation['boxes']:
                    yolo_box = convert_bbox_to_yolo(
                        *box,
                        annotation['width'],
                        annotation['height']
                    )
                    f.write(f"{label_id} {' '.join(map(str, yolo_box))}\n")
    
    process_split(train_data, "train")
    process_split(val_data, "val")
    process_split(test_data, "test")
    
    # data.yaml 생성
    yaml_path = output_dir / "data.yaml"
    yaml_data = {
        "path": str(output_dir.absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    
    logger.info(f"YOLO 데이터셋 생성 완료: {yaml_path}")
    return yaml_path


def yolo_dataset_to_dataframe(yaml_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """YOLO 데이터셋을 DataFrame으로 변환.
    
    Args:
        yaml_path: YOLO data.yaml 파일 경로
        
    Returns:
        (train_df, valid_df, test_df) 튜플
    """
    logger.info("YOLO 데이터셋을 DataFrame으로 로드")
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    base_path = Path(config['path'])
    class_names = config['names']
    
    def load_split(split_name: str) -> pd.DataFrame:
        images_dir = base_path / config[split_name]
        labels_dir = base_path / config[split_name].replace("images", "labels")
        
        data = []
        image_files = sorted(images_dir.glob("*.jpg"))
        
        for img_path in image_files:
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            if not label_path.exists():
                logger.warning(f"라벨 파일 없음: {label_path}")
                continue
            
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x_center, y_center, width, height = map(float, parts)
                        data.append({
                            "image_path": str(img_path),
                            "image_name": img_path.stem,
                            "class_id": int(class_id),
                            "class_name": class_names[int(class_id)],
                            "x_center": x_center,
                            "y_center": y_center,
                            "width": width,
                            "height": height,
                            "split": split_name
                        })
        
        return pd.DataFrame(data)
    
    train_df = load_split("train")
    valid_df = load_split("val")
    test_df = load_split("test")
    
    logger.info(f"DataFrame 로드 완료 - Train: {len(train_df)}, Val: {len(valid_df)}, Test: {len(test_df)}")
    
    return train_df, valid_df, test_df


def validate_conversion(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Dict:
    """YOLO 변환 결과 검증.
    
    Args:
        train_df, valid_df, test_df: 변환된 DataFrame
        
    Returns:
        검증 결과 딕셔너리
    """
    logger.info("변환 결과 검증 시작")
    
    results = {
        "train_samples": len(train_df),
        "valid_samples": len(valid_df),
        "test_samples": len(test_df),
        "total_samples": len(train_df) + len(valid_df) + len(test_df),
        "num_classes": train_df['class_id'].nunique(),
        "class_distribution": {},
        "bbox_validation": {}
    }
    
    # 클래스 분포
    all_df = pd.concat([train_df, valid_df, test_df])
    results["class_distribution"] = all_df['class_name'].value_counts().to_dict()
    
    # 바운딩 박스 검증
    bbox_valid = (
        (all_df['x_center'] >= 0) & (all_df['x_center'] <= 1) &
        (all_df['y_center'] >= 0) & (all_df['y_center'] <= 1) &
        (all_df['width'] > 0) & (all_df['width'] <= 1) &
        (all_df['height'] > 0) & (all_df['height'] <= 1)
    )
    
    results["bbox_validation"] = {
        "valid_count": bbox_valid.sum(),
        "invalid_count": (~bbox_valid).sum(),
        "valid_ratio": bbox_valid.mean()
    }
    
    logger.info(f"검증 완료: {results}")
    return results


def oxfordiit_pet_to_yolo(
    max_samples_per_split: Tuple[int, int, int] | None = None,
    label_mode: Literal["breed", "species"] = "species",
    output_dir: Path | None = None,
) -> Tuple[Path, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """메인 실행 함수.
    
    Args:
        max_samples_per_split: (train, val, test) 각 분할의 최대 샘플 개수. None이면 전체 사용.
        label_mode: 라벨링 모드 ("breed": 품종 분류, "species": 개/고양이 분류)
        
    Returns:
        (yaml_path, train_df, valid_df, test_df, validation_results) 튜플
    """
    # Kaggle 데이터셋 다운로드
    logger.info("Kaggle 데이터셋 다운로드 시작")
    kagglehub_path = kagglehub.dataset_download("devdgohil/the-oxfordiiit-pet-dataset")
    logger.info(f"다운로드 완료: {kagglehub_path}")
    
    # 경로 설정
    paths = PetDatasetPaths.from_kagglehub_path(kagglehub_path)
    if output_dir is None:
        output_dir = Path("./oxfordiiit_pet_yolo")
    
    # YOLO 변환
    yaml_path = create_yolo_dataset(
        paths, output_dir,
        max_samples_per_split=max_samples_per_split,
        label_mode=label_mode
    )
    
    # DataFrame 재로드
    train_df, valid_df, test_df = yolo_dataset_to_dataframe(yaml_path)
    
    # 검증
    validation_results = validate_conversion(train_df, valid_df, test_df)
    
    logger.info("=" * 50)
    logger.info("변환 완료!")
    logger.info(f"YAML 경로: {yaml_path}")
    logger.info(f"검증 결과: {validation_results}")
    logger.info("=" * 50)
    
    return yaml_path, train_df, valid_df, test_df, validation_results


if __name__ == "__main__":
    oxfordiit_pet_to_yolo(max_samples_per_split=(30, 20, 10), label_mode="species", output_dir=Path("./oxford_pet_yolo"))
