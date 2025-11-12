"""
데이터 로드 및 전처리 파이프라인
Data Loading and Preprocessing Pipeline
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """운수종사자 적성검사 데이터 로더"""
    
    def __init__(self, config):
        """
        Args:
            config: ProjectConfig 인스턴스
        """
        self.config = config
        self.data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.scaler = StandardScaler()
        
    def load_json_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        JSON 파일 로드
        
        Args:
            file_path: JSON 파일 경로 (None이면 config에서 가져옴)
            
        Returns:
            pd.DataFrame: 로드된 데이터프레임
        """
        if file_path is None:
            file_path = self.config.data.data_path
            
        print(f"Loading data from: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # JSON이 리스트 형태인 경우
            if isinstance(data, list):
                df = pd.DataFrame(data)
            # JSON이 딕셔너리 형태인 경우
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                raise ValueError("Unsupported JSON format")
            
            print(f"✓ Data loaded successfully: {df.shape}")
            self.data = df
            return df
            
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            raise
    
    def preprocess_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        데이터 전처리
        
        Args:
            df: 전처리할 데이터프레임 (None이면 self.data 사용)
            
        Returns:
            pd.DataFrame: 전처리된 데이터프레임
        """
        if df is None:
            df = self.data.copy()
        else:
            df = df.copy()
        
        print("\nPreprocessing data...")
        
        # 1. 제외 컬럼 제거
        cols_to_drop = [col for col in self.config.data.exclude_cols if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            print(f"  - Dropped columns: {cols_to_drop}")
        
        # 2. 결측치 확인
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"  - Missing values found:")
            for col, count in missing_counts[missing_counts > 0].items():
                print(f"    {col}: {count} ({count/len(df)*100:.2f}%)")
            
            # 수치형 컬럼은 평균으로, 범주형은 최빈값으로 대체
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    if df[col].dtype in ['float64', 'int64']:
                        df[col].fillna(df[col].mean(), inplace=True)
                    else:
                        df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            print("  - No missing values found")
        
        # 3. 무한대 값 처리
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if np.isinf(df[col]).any():
                print(f"  - Found inf values in {col}, replacing with max/min")
                df[col].replace([np.inf, -np.inf], [df[col][~np.isinf(df[col])].max(), 
                                                     df[col][~np.isinf(df[col])].min()], 
                               inplace=True)
        
        # 4. 범주형 변수 인코딩 (A7_performance 등)
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != self.config.data.target_col]
        
        if len(categorical_cols) > 0:
            print(f"  - Encoding categorical columns: {list(categorical_cols)}")
            for col in categorical_cols:
                df[col] = pd.Categorical(df[col]).codes
        
        print(f"✓ Preprocessing completed: {df.shape}")
        self.data = df
        return df
    
    def split_data(self, 
                   df: Optional[pd.DataFrame] = None,
                   stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        데이터를 train/val/test로 분할 (Label 기준 균등 분배)
        
        Args:
            df: 분할할 데이터프레임 (None이면 self.data 사용)
            stratify: 층화 추출 여부 (Label 비율 유지)
            
        Returns:
            Tuple[train_df, val_df, test_df]
        """
        if df is None:
            df = self.data.copy()
        else:
            df = df.copy()
        
        print("\nSplitting data...")
        
        target_col = self.config.data.target_col
        train_ratio = self.config.data.train_ratio
        val_ratio = self.config.data.val_ratio
        test_ratio = self.config.data.test_ratio
        seed = self.config.data.random_seed
        
        # 비율 검증
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Train/Val/Test ratios must sum to 1.0"
        
        # Stratify 설정
        stratify_col = df[target_col] if stratify else None
        
        # 1차 분할: train + (val + test)
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_ratio + test_ratio),
            random_state=seed,
            stratify=stratify_col
        )
        
        # 2차 분할: val + test
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        stratify_temp = temp_df[target_col] if stratify else None
        
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_ratio_adjusted),
            random_state=seed,
            stratify=stratify_temp
        )
        
        # 결과 저장
        self.train_data = train_df
        self.val_data = val_df
        self.test_data = test_df
        
        # 분할 결과 출력
        print(f"  - Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
        print(f"    Label 0: {(train_df[target_col]==0).sum()}, Label 1: {(train_df[target_col]==1).sum()}")
        print(f"  - Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
        print(f"    Label 0: {(val_df[target_col]==0).sum()}, Label 1: {(val_df[target_col]==1).sum()}")
        print(f"  - Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
        print(f"    Label 0: {(test_df[target_col]==0).sum()}, Label 1: {(test_df[target_col]==1).sum()}")
        
        return train_df, val_df, test_df
    
    def get_X_y(self, 
                df: pd.DataFrame,
                scale_features: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        피처(X)와 타겟(y) 분리
        
        Args:
            df: 데이터프레임
            scale_features: 피처 스케일링 여부
            
        Returns:
            Tuple[X, y]
        """
        target_col = self.config.data.target_col
        
        # 타겟 분리
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        y = df[target_col].values
        X = df.drop(columns=[target_col]).values
        
        # 스케일링
        if scale_features:
            X = self.scaler.fit_transform(X)
        
        return X, y
    
    def get_feature_names(self, df: Optional[pd.DataFrame] = None) -> List[str]:
        """
        피처 이름 리스트 반환
        
        Args:
            df: 데이터프레임 (None이면 self.train_data 사용)
            
        Returns:
            List[str]: 피처 이름 리스트
        """
        if df is None:
            df = self.train_data if self.train_data is not None else self.data
        
        target_col = self.config.data.target_col
        return [col for col in df.columns if col != target_col]
    
    def get_data_summary(self) -> Dict:
        """
        데이터 요약 정보 반환
        
        Returns:
            Dict: 데이터 요약 정보
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        target_col = self.config.data.target_col
        
        summary = {
            "total_samples": len(self.data),
            "num_features": len(self.get_feature_names()),
            "label_distribution": self.data[target_col].value_counts().to_dict(),
            "label_ratio": f"{(self.data[target_col]==0).sum()}/{(self.data[target_col]==1).sum()}",
            "missing_values": self.data.isnull().sum().sum(),
            "data_types": self.data.dtypes.value_counts().to_dict()
        }
        
        if self.train_data is not None:
            summary["train_samples"] = len(self.train_data)
            summary["val_samples"] = len(self.val_data)
            summary["test_samples"] = len(self.test_data)
        
        return summary
    
    def save_processed_data(self, output_dir: Optional[str] = None):
        """
        전처리된 데이터 저장
        
        Args:
            output_dir: 저장 디렉토리 (None이면 config에서 가져옴)
        """
        if output_dir is None:
            output_dir = self.config.data.output_dir
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.train_data is not None:
            self.train_data.to_csv(output_path / "train_processed.csv", index=False)
            self.val_data.to_csv(output_path / "val_processed.csv", index=False)
            self.test_data.to_csv(output_path / "test_processed.csv", index=False)
            print(f"\n✓ Processed data saved to {output_dir}")
        else:
            print("\n✗ No split data to save. Please run split_data() first.")


if __name__ == "__main__":
    # 테스트 코드
    from config import config
    
    # 데이터 로더 생성
    loader = DataLoader(config)
    
    # 데이터 로드 테스트 (실제 파일이 있다고 가정)
    try:
        df = loader.load_json_data()
        df = loader.preprocess_data()
        train_df, val_df, test_df = loader.split_data()
        
        # 요약 정보 출력
        summary = loader.get_data_summary()
        print("\n" + "="*60)
        print("Data Summary:")
        print("="*60)
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        # 데이터 저장
        loader.save_processed_data()
        
    except Exception as e:
        print(f"Test failed: {e}")
        print("This is expected if train_A.json doesn't exist yet.")