"""
간단한 파이프라인 테스트 스크립트
Quick Pipeline Test Script
"""

import sys
sys.path.append('/home/claude/transport_worker_analysis')

from config import ProjectConfig
from data_loader import DataLoader
from data_analyzer import DataAnalyzer
from model_trainer import ModelTrainer

def test_pipeline():
    """파이프라인 간단 테스트"""
    
    print("\n" + "="*70)
    print("  운수종사자 적성검사 분석 시스템 - Quick Test")
    print("="*70 + "\n")
    
    try:
        # 1. 설정
        print("1. Loading configuration...")
        config = ProjectConfig()
        print("   ✓ Configuration loaded")
        
        # 2. 데이터 로드
        print("\n2. Loading and preprocessing data...")
        loader = DataLoader(config)
        
        try:
            df = loader.load_json_data("/mnt/user-data/uploads/train_A.json")
            print(f"   ✓ Data loaded: {df.shape}")
        except FileNotFoundError:
            print("   ✗ train_A.json not found")
            print("   → Please ensure the data file exists at:")
            print("     /mnt/user-data/uploads/train_A.json")
            return False
        
        # 3. 전처리
        df = loader.preprocess_data()
        print(f"   ✓ Preprocessed: {df.shape}")
        
        # 4. 데이터 분할
        train_df, val_df, test_df = loader.split_data()
        print(f"   ✓ Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        # 5. 간단한 분석
        print("\n3. Running basic analysis...")
        analyzer = DataAnalyzer(config, loader)
        analyzer.plot_label_distribution()
        print("   ✓ Label distribution plotted")
        
        # 6. 모델 학습 (빠른 테스트용)
        print("\n4. Training model (quick mode)...")
        config.training.num_boost_round = 100  # 빠른 테스트
        config.training.early_stopping_rounds = 20
        
        trainer = ModelTrainer(config, loader)
        trainer.train()
        print("   ✓ Model trained")
        
        # 7. 평가
        print("\n5. Evaluating model...")
        results = trainer.evaluate(dataset_name="Test")
        print("   ✓ Model evaluated")
        
        # 8. 결과 요약
        print("\n" + "="*70)
        print("Test Results Summary:")
        print("="*70)
        for metric, value in results['metrics'].items():
            print(f"  {metric:12s}: {value:.4f}")
        
        print("\n✓ Pipeline test completed successfully!")
        print(f"   Output directory: {config.data.output_dir}")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)