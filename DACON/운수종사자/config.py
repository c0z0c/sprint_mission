"""
운수종사자 적성검사 분석 프로젝트 설정
Config for Transport Worker Aptitude Test Analysis Project
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path


@dataclass
class DataConfig:
    """데이터 관련 설정"""
    # 파일 경로
    data_path: str = "/mnt/user-data/uploads/train_A.json"
    output_dir: str = "/mnt/user-data/outputs"
    
    # 데이터 분할 비율
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # 시드
    random_seed: int = 42
    
    # 타겟 컬럼
    target_col: str = "Label"
    
    # 제외할 컬럼 (ID, 날짜 등)
    exclude_cols: List[str] = field(default_factory=lambda: [
        "Test_id_idx", "PrimaryKey", "Test", "TestDate"
    ])
    
    # A검사별 피처 그룹
    feature_groups: Dict[str, List[str]] = field(default_factory=lambda: {
        "A1": ["A1L_fail_rate", "A1L_mean_error", "A1L_fast_fail", "A1L_std",
               "A1R_fail_rate", "A1R_mean_error", "A1R_fast_fail", "A1R_std",
               "A1_direction_diff"],
        "A2": ["A2_constant_fail", "A2_constant_error", "A2_accel_fail", 
               "A2_accel_error", "A2_decel_fail", "A2_decel_error", 
               "A2_total_fail", "A2_mean_error", "A2_std"],
        "A3": ["A3S_correct_rate", "A3S_fail_rate", "A3S_mean_rt",
               "A3B_correct_rate", "A3B_fail_rate", "A3B_mean_rt",
               "A3_valid_correct", "A3_valid_fail_rate", "A3_invalid_correct",
               "A3_invalid_fail_rate", "A3_size_diff", "A3_valid_invalid_diff",
               "A3_total_fail", "A3_mean_rt"],
        "A4": ["A4C_correct_rate", "A4C_fail_rate", "A4C_mean_rt",
               "A4I_correct_rate", "A4I_fail_rate", "A4I_mean_rt",
               "A4_stroop_effect", "A4_total_correct", "A4_total_fail", "A4_mean_rt"],
        "A5": ["A5_no_change_correct", "A5_change_correct", 
               "A5_sensitivity", "A5_total_fail"],
        "A6": ["A6_correct_count", "A6_accuracy", "A6_incorrect_count", "A6_acc_level"],
        "A7": ["A7_correct_count", "A7_accuracy", "A7_incorrect_count", 
               "A7_acc_level", "A7_performance"],
        "A8": ["A8_distortion_score", "A8_consistency_score", "A8_distortion_norm",
               "A8_consistency_norm", "A8_distortion_high", "A8_consistency_good",
               "A8_validity_score", "A8_test_reliable"],
        "A9": ["A9_rt_mean", "A9_rt_std", "A9_rt_cv", "A9_rt_min", "A9_rt_max",
               "A9_rt_q25", "A9_rt_q75", "A9_accuracy", "A9_go_rt", "A9_go_acc",
               "A9_go_miss", "A9_nogo_acc", "A9_nogo_commission", "A9_nogo_rt",
               "A9_cond1_rt", "A9_cond1_acc", "A9_cond2_rt", "A9_cond2_acc",
               "A9_cond3_rt", "A9_cond3_acc", "A9_error_rt", "A9_correct_rt",
               "A9_too_fast_cnt", "A9_too_slow_cnt", "A9_response_rate",
               "A9_no_response_cnt", "A9_false_alarm_rate", "A9_impulsivity_index"]
    })


@dataclass
class ModelConfig:
    """모델 관련 설정"""
    # 모델 타입
    model_type: str = "lightgbm"  # lightgbm, xgboost, random_forest, logistic
    
    # LightGBM 하이퍼파라미터
    lgb_params: Dict = field(default_factory=lambda: {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "max_depth": -1,
        "min_child_samples": 20,
        "verbose": -1,
        "random_state": 42
    })
    
    # XGBoost 하이퍼파라미터
    xgb_params: Dict = field(default_factory=lambda: {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.9,
        "min_child_weight": 1,
        "random_state": 42
    })
    
    # RandomForest 하이퍼파라미터
    rf_params: Dict = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 10,
        "min_samples_leaf": 4,
        "random_state": 42,
        "n_jobs": -1
    })


@dataclass
class TrainingConfig:
    """학습 관련 설정"""
    # 학습 파라미터
    num_boost_round: int = 1000
    early_stopping_rounds: int = 50
    verbose_eval: int = 50
    
    # 교차 검증
    use_cross_validation: bool = False
    n_folds: int = 5
    
    # 클래스 불균형 처리
    handle_imbalance: bool = True
    scale_pos_weight: Optional[float] = None  # None이면 자동 계산


@dataclass
class VisualizationConfig:
    """시각화 관련 설정"""
    # 그래프 저장 설정
    save_plots: bool = True
    plot_format: str = "png"
    dpi: int = 300
    
    # 그래프 스타일
    style: str = "seaborn-v0_8-darkgrid"
    figsize: tuple = (12, 8)
    
    # 한글 폰트 설정
    use_korean_font: bool = True
    font_family: str = "DejaVu Sans"


@dataclass
class ProjectConfig:
    """전체 프로젝트 설정"""
    # 프로젝트 정보
    project_name: str = "운수종사자_적성검사_분석"
    version: str = "1.0.0"
    
    # 하위 설정
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    def __post_init__(self):
        """설정 초기화 후 디렉토리 생성"""
        Path(self.data.output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{self.data.output_dir}/plots").mkdir(exist_ok=True)
        Path(f"{self.data.output_dir}/models").mkdir(exist_ok=True)
        Path(f"{self.data.output_dir}/reports").mkdir(exist_ok=True)
    
    def get_feature_list(self) -> List[str]:
        """모든 피처 리스트 반환"""
        all_features = []
        for features in self.data.feature_groups.values():
            all_features.extend(features)
        return all_features
    
    def print_config(self):
        """설정 출력"""
        print(f"{'='*60}")
        print(f"Project: {self.project_name} v{self.version}")
        print(f"{'='*60}")
        print(f"\n[Data Config]")
        print(f"  - Data path: {self.data.data_path}")
        print(f"  - Train/Val/Test: {self.data.train_ratio}/{self.data.val_ratio}/{self.data.test_ratio}")
        print(f"  - Random seed: {self.data.random_seed}")
        print(f"\n[Model Config]")
        print(f"  - Model type: {self.model.model_type}")
        print(f"\n[Training Config]")
        print(f"  - Num boost round: {self.training.num_boost_round}")
        print(f"  - Early stopping: {self.training.early_stopping_rounds}")
        print(f"  - Cross validation: {self.training.use_cross_validation}")
        print(f"{'='*60}\n")


# 전역 설정 인스턴스
config = ProjectConfig()


if __name__ == "__main__":
    # 설정 테스트
    config.print_config()
    print(f"Total features: {len(config.get_feature_list())}")
    print(f"\nFeature groups:")
    for group_name, features in config.data.feature_groups.items():
        print(f"  {group_name}: {len(features)} features")