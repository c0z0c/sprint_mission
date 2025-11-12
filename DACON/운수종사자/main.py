"""
운수종사자 적성검사 분석 메인 파이프라인
Main Pipeline for Transport Worker Aptitude Test Analysis
"""

from config import ProjectConfig
from data_loader import DataLoader
from data_analyzer import DataAnalyzer
from model_trainer import ModelTrainer
import argparse
from pathlib import Path


class MVPPipeline:
    """MVP 분석 파이프라인"""
    
    def __init__(self, config_path: str = None):
        """
        Args:
            config_path: 설정 파일 경로 (현재는 미사용, 향후 YAML 등 지원)
        """
        self.config = ProjectConfig()
        self.loader = None
        self.analyzer = None
        self.trainer = None
        
    def setup(self):
        """파이프라인 설정"""
        print("\n" + "="*70)
        print("  운수종사자 적성검사 분석 시스템 MVP")
        print("  Transport Worker Aptitude Test Analysis System")
        print("="*70)
        self.config.print_config()
    
    def run_data_loading(self, data_path: str = None):
        """
        1단계: 데이터 로드 및 전처리
        
        Args:
            data_path: 데이터 파일 경로
        """
        print("\n" + "="*70)
        print("STEP 1: DATA LOADING & PREPROCESSING")
        print("="*70)
        
        if data_path:
            self.config.data.data_path = data_path
        
        self.loader = DataLoader(self.config)
        
        # 데이터 로드
        self.loader.load_json_data()
        
        # 전처리
        self.loader.preprocess_data()
        
        # 데이터 분할
        self.loader.split_data(stratify=True)
        
        # 요약 정보
        summary = self.loader.get_data_summary()
        print("\n" + "-"*70)
        print("Data Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        print("-"*70)
        
        # 전처리된 데이터 저장
        self.loader.save_processed_data()
        
        return self.loader
    
    def run_data_analysis(self, generate_full_report: bool = True):
        """
        2단계: 데이터 분석 및 시각화
        
        Args:
            generate_full_report: 전체 리포트 생성 여부
        """
        if self.loader is None:
            raise ValueError("Data not loaded. Please run run_data_loading() first.")
        
        print("\n" + "="*70)
        print("STEP 2: DATA ANALYSIS & VISUALIZATION")
        print("="*70)
        
        self.analyzer = DataAnalyzer(self.config, self.loader)
        
        if generate_full_report:
            self.analyzer.generate_full_report()
        else:
            # 기본 분석만 수행
            self.analyzer.plot_label_distribution()
            self.analyzer.plot_feature_importance_preliminary(top_n=20)
        
        return self.analyzer
    
    def run_model_training(self, model_type: str = None):
        """
        3단계: 모델 학습
        
        Args:
            model_type: 모델 타입 (lightgbm, xgboost, random_forest, logistic)
        """
        if self.loader is None:
            raise ValueError("Data not loaded. Please run run_data_loading() first.")
        
        print("\n" + "="*70)
        print("STEP 3: MODEL TRAINING")
        print("="*70)
        
        if model_type:
            self.config.model.model_type = model_type
        
        self.trainer = ModelTrainer(self.config, self.loader)
        
        # 학습
        self.trainer.train()
        
        # 모델 저장
        self.trainer.save_model()
        
        return self.trainer
    
    def run_model_evaluation(self):
        """
        4단계: 모델 평가
        """
        if self.trainer is None or self.trainer.model is None:
            raise ValueError("Model not trained. Please run run_model_training() first.")
        
        print("\n" + "="*70)
        print("STEP 4: MODEL EVALUATION")
        print("="*70)
        
        # Validation Set 평가
        print("\n[Validation Set Evaluation]")
        X_val, y_val = self.loader.get_X_y(self.loader.val_data)
        self.trainer.evaluate(X_val, y_val, dataset_name="Validation")
        
        # Test Set 평가
        print("\n[Test Set Evaluation]")
        X_test, y_test = self.loader.get_X_y(self.loader.test_data)
        self.trainer.evaluate(X_test, y_test, dataset_name="Test")
        
        # 시각화
        print("\nGenerating evaluation plots...")
        self.trainer.plot_confusion_matrix(dataset_name="test")
        self.trainer.plot_roc_curve(dataset_name="test")
        self.trainer.plot_feature_importance(top_n=20)
        
        return self.trainer.results
    
    def run_full_pipeline(self, 
                          data_path: str = None,
                          model_type: str = None,
                          full_analysis: bool = True):
        """
        전체 파이프라인 실행
        
        Args:
            data_path: 데이터 파일 경로
            model_type: 모델 타입
            full_analysis: 전체 분석 수행 여부
        """
        self.setup()
        
        # 1. 데이터 로드 및 전처리
        self.run_data_loading(data_path)
        
        # 2. 데이터 분석
        self.run_data_analysis(generate_full_report=full_analysis)
        
        # 3. 모델 학습
        self.run_model_training(model_type)
        
        # 4. 모델 평가
        results = self.run_model_evaluation()
        
        # 최종 요약
        self.print_final_summary(results)
        
        return results
    
    def print_final_summary(self, results: dict):
        """최종 결과 요약 출력"""
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        
        # 데이터 정보
        summary = self.loader.get_data_summary()
        print("\n[Data Information]")
        print(f"  Total Samples: {summary['total_samples']}")
        print(f"  Train/Val/Test: {summary['train_samples']}/{summary['val_samples']}/{summary['test_samples']}")
        print(f"  Features: {summary['num_features']}")
        print(f"  Label Distribution: {summary['label_distribution']}")
        
        # 모델 성능
        print("\n[Model Performance]")
        print(f"  Model Type: {self.config.model.model_type.upper()}")
        
        if 'test_results' in results:
            test_metrics = results['test_results']['metrics']
            print("\n  Test Set Metrics:")
            print(f"    Accuracy:  {test_metrics['accuracy']:.4f}")
            print(f"    Precision: {test_metrics['precision']:.4f}")
            print(f"    Recall:    {test_metrics['recall']:.4f}")
            print(f"    F1-Score:  {test_metrics['f1']:.4f}")
            print(f"    AUC:       {test_metrics['auc']:.4f}")
        
        # 출력 파일 위치
        print("\n[Output Files]")
        print(f"  Processed Data: {self.config.data.output_dir}/")
        print(f"  Plots: {self.config.data.output_dir}/plots/")
        print(f"  Models: {self.config.data.output_dir}/models/")
        
        print("\n" + "="*70)
        print("✓ Pipeline Completed Successfully!")
        print("="*70 + "\n")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="운수종사자 적성검사 분석 시스템 MVP"
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default="/mnt/user-data/uploads/train_A.json",
        help='데이터 파일 경로'
    )
    
    parser.add_argument(
        '--model-type',
        type=str,
        default='lightgbm',
        choices=['lightgbm', 'xgboost', 'random_forest', 'logistic'],
        help='사용할 모델 타입'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='빠른 분석 모드 (전체 시각화 생략)'
    )
    
    parser.add_argument(
        '--step',
        type=str,
        choices=['load', 'analyze', 'train', 'evaluate', 'all'],
        default='all',
        help='실행할 단계'
    )
    
    args = parser.parse_args()
    
    # 파이프라인 실행
    pipeline = MVPPipeline()
    
    if args.step == 'all':
        pipeline.run_full_pipeline(
            data_path=args.data_path,
            model_type=args.model_type,
            full_analysis=not args.quick
        )
    elif args.step == 'load':
        pipeline.setup()
        pipeline.run_data_loading(args.data_path)
    elif args.step == 'analyze':
        pipeline.setup()
        pipeline.run_data_loading(args.data_path)
        pipeline.run_data_analysis(generate_full_report=not args.quick)
    elif args.step == 'train':
        pipeline.setup()
        pipeline.run_data_loading(args.data_path)
        pipeline.run_model_training(args.model_type)
    elif args.step == 'evaluate':
        pipeline.setup()
        pipeline.run_data_loading(args.data_path)
        pipeline.run_model_training(args.model_type)
        pipeline.run_model_evaluation()


if __name__ == "__main__":
    # 커맨드라인 인자가 없을 경우 기본 실행
    import sys
    if len(sys.argv) == 1:
        print("Running with default settings...")
        pipeline = MVPPipeline()
        try:
            pipeline.run_full_pipeline(
                data_path="/mnt/user-data/uploads/train_A.json",
                model_type="lightgbm",
                full_analysis=False  # 빠른 테스트를 위해
            )
        except Exception as e:
            print(f"\nError: {e}")
            print("\nUsage examples:")
            print("  python main.py --data-path /path/to/data.json")
            print("  python main.py --model-type xgboost")
            print("  python main.py --quick")
            print("  python main.py --step analyze")
    else:
        main()