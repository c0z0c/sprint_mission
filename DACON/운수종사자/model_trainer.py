"""
모델 학습 및 평가 파이프라인
Model Training and Evaluation Pipeline
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from typing import Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """모델 학습 및 평가 클래스"""
    
    def __init__(self, config, data_loader):
        """
        Args:
            config: ProjectConfig 인스턴스
            data_loader: DataLoader 인스턴스
        """
        self.config = config
        self.loader = data_loader
        self.model = None
        self.model_type = config.model.model_type
        self.feature_names = None
        self.results = {}
        
        self.model_dir = Path(config.data.output_dir) / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.plot_dir = Path(config.data.output_dir) / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)
    
    def _calculate_scale_pos_weight(self, y_train: np.ndarray) -> float:
        """클래스 불균형 가중치 계산"""
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        return n_neg / n_pos if n_pos > 0 else 1.0
    
    def train(self, 
              X_train: Optional[np.ndarray] = None,
              y_train: Optional[np.ndarray] = None,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None):
        """
        모델 학습
        
        Args:
            X_train, y_train: 학습 데이터 (None이면 loader에서 가져옴)
            X_val, y_val: 검증 데이터 (None이면 loader에서 가져옴)
        """
        print("\n" + "="*60)
        print(f"Training {self.model_type.upper()} Model")
        print("="*60)
        
        # 데이터 준비
        if X_train is None or y_train is None:
            X_train, y_train = self.loader.get_X_y(self.loader.train_data)
            X_val, y_val = self.loader.get_X_y(self.loader.val_data)
        
        self.feature_names = self.loader.get_feature_names()
        
        print(f"  Train samples: {len(X_train)}")
        print(f"  Val samples: {len(X_val)}")
        print(f"  Features: {len(self.feature_names)}")
        
        # 클래스 불균형 처리
        if self.config.training.handle_imbalance:
            scale_pos_weight = self._calculate_scale_pos_weight(y_train)
            print(f"  Scale pos weight: {scale_pos_weight:.2f}")
        else:
            scale_pos_weight = 1.0
        
        # 모델 학습
        if self.model_type == "lightgbm":
            self.model = self._train_lightgbm(X_train, y_train, X_val, y_val, scale_pos_weight)
        elif self.model_type == "xgboost":
            self.model = self._train_xgboost(X_train, y_train, X_val, y_val, scale_pos_weight)
        elif self.model_type == "random_forest":
            self.model = self._train_random_forest(X_train, y_train, scale_pos_weight)
        elif self.model_type == "logistic":
            self.model = self._train_logistic(X_train, y_train)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print("\n✓ Model training completed")
        
        return self.model
    
    def _train_lightgbm(self, X_train, y_train, X_val, y_val, scale_pos_weight):
        """LightGBM 모델 학습"""
        print("\n  Training LightGBM...")
        
        # 파라미터 설정
        params = self.config.model.lgb_params.copy()
        if self.config.training.handle_imbalance:
            params['scale_pos_weight'] = scale_pos_weight
        
        # 데이터셋 생성
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=self.feature_names)
        
        # 학습
        evals_result = {}
        model = lgb.train(
            params,
            train_data,
            num_boost_round=self.config.training.num_boost_round,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.config.training.early_stopping_rounds),
                lgb.log_evaluation(period=self.config.training.verbose_eval),
                lgb.record_evaluation(evals_result)
            ]
        )
        
        # 학습 이력 저장
        self.results['train_history'] = evals_result
        self.results['best_iteration'] = model.best_iteration
        
        print(f"  Best iteration: {model.best_iteration}")
        print(f"  Best score: {model.best_score['val']['auc']:.4f}")
        
        return model
    
    def _train_xgboost(self, X_train, y_train, X_val, y_val, scale_pos_weight):
        """XGBoost 모델 학습"""
        print("\n  Training XGBoost...")
        
        # 파라미터 설정
        params = self.config.model.xgb_params.copy()
        if self.config.training.handle_imbalance:
            params['scale_pos_weight'] = scale_pos_weight
        
        # 데이터셋 생성
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
        
        # 학습
        evals_result = {}
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.config.training.num_boost_round,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=self.config.training.early_stopping_rounds,
            verbose_eval=self.config.training.verbose_eval,
            evals_result=evals_result
        )
        
        # 학습 이력 저장
        self.results['train_history'] = evals_result
        self.results['best_iteration'] = model.best_iteration
        
        print(f"  Best iteration: {model.best_iteration}")
        
        return model
    
    def _train_random_forest(self, X_train, y_train, scale_pos_weight):
        """Random Forest 모델 학습"""
        print("\n  Training Random Forest...")
        
        params = self.config.model.rf_params.copy()
        
        # 클래스 가중치 설정
        if self.config.training.handle_imbalance:
            params['class_weight'] = {0: 1.0, 1: scale_pos_weight}
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        return model
    
    def _train_logistic(self, X_train, y_train):
        """Logistic Regression 모델 학습"""
        print("\n  Training Logistic Regression...")
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        
        return model
    
    def evaluate(self, 
                 X_test: Optional[np.ndarray] = None,
                 y_test: Optional[np.ndarray] = None,
                 dataset_name: str = "Test") -> Dict:
        """
        모델 평가
        
        Args:
            X_test, y_test: 테스트 데이터 (None이면 loader에서 가져옴)
            dataset_name: 데이터셋 이름
            
        Returns:
            Dict: 평가 결과
        """
        if self.model is None:
            raise ValueError("No trained model found. Please train a model first.")
        
        # 데이터 준비
        if X_test is None or y_test is None:
            X_test, y_test = self.loader.get_X_y(self.loader.test_data)
        
        print("\n" + "="*60)
        print(f"Evaluating on {dataset_name} Set")
        print("="*60)
        
        # 예측
        if self.model_type == "lightgbm":
            y_pred_proba = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        elif self.model_type == "xgboost":
            dtest = xgb.DMatrix(X_test, feature_names=self.feature_names)
            y_pred_proba = self.model.predict(dtest, iteration_range=(0, self.model.best_iteration))
        else:
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # 메트릭 계산
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # 결과 출력
        print(f"\nMetrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  AUC:       {metrics['auc']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"  TN: {cm[0, 0]:5d}  FP: {cm[0, 1]:5d}")
        print(f"  FN: {cm[1, 0]:5d}  TP: {cm[1, 1]:5d}")
        
        # 분류 리포트
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Safe', 'Risk']))
        
        # 결과 저장
        eval_results = {
            'metrics': metrics,
            'confusion_matrix': cm,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        self.results[f'{dataset_name.lower()}_results'] = eval_results
        
        return eval_results
    
    def plot_confusion_matrix(self, dataset_name: str = "test"):
        """Confusion Matrix 시각화"""
        results_key = f'{dataset_name}_results'
        if results_key not in self.results:
            print(f"No results found for {dataset_name} set")
            return
        
        cm = self.results[results_key]['confusion_matrix']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Safe (0)', 'Risk (1)'],
                   yticklabels=['Safe (0)', 'Risk (1)'],
                   cbar_kws={'label': 'Count'},
                   ax=ax)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(f'Confusion Matrix - {dataset_name.capitalize()} Set', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / f'confusion_matrix_{dataset_name}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, dataset_name: str = "test"):
        """ROC Curve 시각화"""
        results_key = f'{dataset_name}_results'
        if results_key not in self.results:
            print(f"No results found for {dataset_name} set")
            return
        
        y_true = self.results[results_key]['y_true']
        y_pred_proba = self.results[results_key]['y_pred_proba']
        
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc_score = self.results[results_key]['metrics']['auc']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC curve (AUC = {auc_score:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curve - {dataset_name.capitalize()} Set', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / f'roc_curve_{dataset_name}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, top_n: int = 20):
        """피처 중요도 시각화"""
        if self.model is None:
            print("No trained model found")
            return
        
        # 피처 중요도 추출
        if self.model_type == "lightgbm":
            importance = self.model.feature_importance(importance_type='gain')
        elif self.model_type == "xgboost":
            importance = self.model.get_score(importance_type='gain')
            importance = np.array([importance.get(f, 0) for f in self.feature_names])
        elif self.model_type == "random_forest":
            importance = self.model.feature_importances_
        else:
            print(f"Feature importance not supported for {self.model_type}")
            return
        
        # 데이터프레임 생성
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        # 시각화
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.viridis(feature_importance_df['importance'].values / 
                               feature_importance_df['importance'].values.max())
        ax.barh(range(len(feature_importance_df)), 
               feature_importance_df['importance'].values,
               color=colors)
        ax.set_yticks(range(len(feature_importance_df)))
        ax.set_yticklabels(feature_importance_df['feature'].values, fontsize=10)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importance - {self.model_type.upper()}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'feature_importance.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # 중요도 출력
        print("\n" + "="*60)
        print(f"Top {top_n} Feature Importance")
        print("="*60)
        for i, row in feature_importance_df.iterrows():
            print(f"{row['feature']:30s}: {row['importance']:10.2f}")
        print("="*60)
        
        return feature_importance_df
    
    def save_model(self, filename: Optional[str] = None):
        """모델 저장"""
        if self.model is None:
            print("No model to save")
            return
        
        if filename is None:
            filename = f"{self.model_type}_model.pkl"
        
        filepath = self.model_dir / filename
        joblib.dump(self.model, filepath)
        print(f"\n✓ Model saved to: {filepath}")
    
    def load_model(self, filename: str):
        """모델 로드"""
        filepath = self.model_dir / filename
        self.model = joblib.load(filepath)
        print(f"✓ Model loaded from: {filepath}")
        return self.model


if __name__ == "__main__":
    # 테스트 코드
    from config import config
    from data_loader import DataLoader
    
    try:
        # 데이터 로드
        loader = DataLoader(config)
        loader.load_json_data()
        loader.preprocess_data()
        loader.split_data()
        
        # 트레이너 생성
        trainer = ModelTrainer(config, loader)
        
        # 학습
        trainer.train()
        
        # 평가
        trainer.evaluate(dataset_name="Test")
        
        # 시각화
        trainer.plot_confusion_matrix()
        trainer.plot_roc_curve()
        trainer.plot_feature_importance()
        
        # 모델 저장
        trainer.save_model()
        
    except Exception as e:
        print(f"Test failed: {e}")
        print("This is expected if train_A.json doesn't exist yet.")