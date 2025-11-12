"""
데이터 분석 및 시각화 파이프라인
Data Analysis and Visualization Pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class DataAnalyzer:
    """운수종사자 적성검사 데이터 분석 및 시각화"""
    
    def __init__(self, config, data_loader):
        """
        Args:
            config: ProjectConfig 인스턴스
            data_loader: DataLoader 인스턴스
        """
        self.config = config
        self.loader = data_loader
        self.plot_dir = Path(config.data.output_dir) / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        
        # 시각화 스타일 설정
        try:
            plt.style.use(config.visualization.style)
        except:
            plt.style.use('default')
        
        # 한글 폰트 설정 (matplotlib 기본 설정)
        plt.rcParams['axes.unicode_minus'] = False
        
    def _save_plot(self, filename: str):
        """그래프 저장"""
        if self.config.visualization.save_plots:
            filepath = self.plot_dir / f"{filename}.{self.config.visualization.plot_format}"
            plt.savefig(filepath, dpi=self.config.visualization.dpi, bbox_inches='tight')
            print(f"  ✓ Plot saved: {filepath}")
    
    def plot_label_distribution(self, df: Optional[pd.DataFrame] = None):
        """
        레이블 분포 시각화
        
        Args:
            df: 분석할 데이터프레임 (None이면 전체 데이터 사용)
        """
        if df is None:
            df = self.loader.data
        
        target_col = self.config.data.target_col
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 카운트 플롯
        label_counts = df[target_col].value_counts()
        axes[0].bar(label_counts.index, label_counts.values, color=['skyblue', 'salmon'])
        axes[0].set_xlabel('Label', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('Label Distribution (Count)', fontsize=14, fontweight='bold')
        axes[0].set_xticks([0, 1])
        axes[0].set_xticklabels(['Safe (0)', 'Risk (1)'])
        
        # 텍스트 추가
        for i, v in enumerate(label_counts.values):
            axes[0].text(i, v + 100, str(v), ha='center', fontsize=11, fontweight='bold')
        
        # 비율 파이 차트
        colors = ['skyblue', 'salmon']
        explode = (0.05, 0.05)
        axes[1].pie(label_counts.values, labels=['Safe (0)', 'Risk (1)'], 
                   autopct='%1.1f%%', colors=colors, explode=explode,
                   startangle=90, textprops={'fontsize': 11})
        axes[1].set_title('Label Distribution (Ratio)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        self._save_plot('01_label_distribution')
        plt.show()
        
        # 통계 출력
        print("\n" + "="*60)
        print("Label Distribution Statistics")
        print("="*60)
        print(f"Total samples: {len(df)}")
        print(f"Safe (0): {label_counts[0]} ({label_counts[0]/len(df)*100:.2f}%)")
        print(f"Risk (1): {label_counts[1]} ({label_counts[1]/len(df)*100:.2f}%)")
        print(f"Imbalance ratio: {label_counts[0]/label_counts[1]:.2f}:1")
        print("="*60)
    
    def plot_feature_distributions(self, 
                                   features: Optional[List[str]] = None,
                                   max_features: int = 20):
        """
        피처 분포 시각화
        
        Args:
            features: 시각화할 피처 리스트 (None이면 상위 max_features개 사용)
            max_features: 최대 피처 수
        """
        df = self.loader.data
        target_col = self.config.data.target_col
        
        if features is None:
            features = self.loader.get_feature_names()[:max_features]
        
        n_features = len(features)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*4))
        axes = axes.flatten()
        
        for idx, feature in enumerate(features):
            ax = axes[idx]
            
            # 각 Label별 분포
            for label in [0, 1]:
                data = df[df[target_col] == label][feature]
                ax.hist(data, bins=30, alpha=0.6, 
                       label=f'Label {label}', 
                       color='skyblue' if label == 0 else 'salmon')
            
            ax.set_xlabel(feature, fontsize=9)
            ax.set_ylabel('Frequency', fontsize=9)
            ax.set_title(f'{feature}', fontsize=10, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # 빈 서브플롯 제거
        for idx in range(n_features, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        self._save_plot('02_feature_distributions')
        plt.show()
    
    def plot_correlation_matrix(self, 
                                feature_group: Optional[str] = None,
                                top_n: int = 30):
        """
        상관관계 행렬 히트맵
        
        Args:
            feature_group: 분석할 검사 그룹 (None이면 전체, "A1", "A2" 등)
            top_n: 전체 분석 시 상위 N개 피처
        """
        df = self.loader.data
        target_col = self.config.data.target_col
        
        # 피처 선택
        if feature_group is not None:
            if feature_group in self.config.data.feature_groups:
                features = self.config.data.feature_groups[feature_group]
                title = f'Correlation Matrix - {feature_group}'
            else:
                print(f"Warning: Unknown feature group '{feature_group}'")
                return
        else:
            # 타겟과의 상관계수 기준 상위 N개
            numeric_features = df.select_dtypes(include=[np.number]).columns
            numeric_features = [f for f in numeric_features if f != target_col]
            
            correlations = df[numeric_features].corrwith(df[target_col]).abs()
            top_features = correlations.nlargest(top_n).index.tolist()
            features = top_features
            title = f'Correlation Matrix - Top {top_n} Features'
        
        # 상관관계 계산
        corr_matrix = df[features + [target_col]].corr()
        
        # 히트맵 그리기
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, 
                   cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        suffix = feature_group if feature_group else 'top'
        self._save_plot(f'03_correlation_matrix_{suffix}')
        plt.show()
    
    def plot_feature_importance_preliminary(self, top_n: int = 20):
        """
        타겟과의 상관계수 기반 사전 피처 중요도
        
        Args:
            top_n: 상위 N개 피처
        """
        df = self.loader.data
        target_col = self.config.data.target_col
        
        # 수치형 피처만 선택
        numeric_features = df.select_dtypes(include=[np.number]).columns
        numeric_features = [f for f in numeric_features if f != target_col]
        
        # 타겟과의 상관계수 계산
        correlations = df[numeric_features].corrwith(df[target_col]).abs()
        top_features = correlations.nlargest(top_n).sort_values(ascending=True)
        
        # 시각화
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.RdYlGn(top_features.values / top_features.values.max())
        ax.barh(range(len(top_features)), top_features.values, color=colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features.index, fontsize=10)
        ax.set_xlabel('Absolute Correlation with Label', fontsize=12)
        ax.set_title(f'Top {top_n} Features by Correlation with Target', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        self._save_plot('04_feature_importance_preliminary')
        plt.show()
        
        # 상위 피처 출력
        print("\n" + "="*60)
        print(f"Top {top_n} Features by Correlation")
        print("="*60)
        for i, (feature, corr) in enumerate(top_features.iloc[::-1].items(), 1):
            print(f"{i:2d}. {feature:30s}: {corr:.4f}")
        print("="*60)
    
    def plot_test_group_analysis(self):
        """
        A검사 그룹별 분석
        """
        df = self.loader.data
        target_col = self.config.data.target_col
        
        n_groups = len(self.config.data.feature_groups)
        n_cols = 3
        n_rows = (n_groups + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows*5))
        axes = axes.flatten()
        
        for idx, (group_name, features) in enumerate(self.config.data.feature_groups.items()):
            ax = axes[idx]
            
            # 각 검사 그룹의 평균값 비교
            safe_means = df[df[target_col] == 0][features].mean()
            risk_means = df[df[target_col] == 1][features].mean()
            
            x = np.arange(min(len(features), 10))  # 최대 10개 피처만 표시
            width = 0.35
            
            display_features = features[:10]
            safe_values = safe_means[display_features].values
            risk_values = risk_means[display_features].values
            
            ax.bar(x - width/2, safe_values, width, label='Safe (0)', 
                  color='skyblue', alpha=0.8)
            ax.bar(x + width/2, risk_values, width, label='Risk (1)', 
                  color='salmon', alpha=0.8)
            
            ax.set_xlabel('Features', fontsize=10)
            ax.set_ylabel('Mean Value', fontsize=10)
            ax.set_title(f'{group_name} Test Features', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([f.split('_')[-1] if '_' in f else f 
                               for f in display_features], 
                              rotation=45, ha='right', fontsize=8)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
        
        # 빈 서브플롯 제거
        for idx in range(n_groups, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        self._save_plot('05_test_group_analysis')
        plt.show()
    
    def plot_boxplots_by_label(self, features: Optional[List[str]] = None, max_features: int = 12):
        """
        레이블별 박스플롯
        
        Args:
            features: 시각화할 피처 (None이면 상위 중요 피처)
            max_features: 최대 피처 수
        """
        df = self.loader.data
        target_col = self.config.data.target_col
        
        if features is None:
            # 타겟과 상관관계 높은 피처 선택
            numeric_features = df.select_dtypes(include=[np.number]).columns
            numeric_features = [f for f in numeric_features if f != target_col]
            correlations = df[numeric_features].corrwith(df[target_col]).abs()
            features = correlations.nlargest(max_features).index.tolist()
        
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows*4))
        axes = axes.flatten()
        
        for idx, feature in enumerate(features):
            ax = axes[idx]
            
            data_to_plot = [df[df[target_col] == 0][feature].dropna(),
                           df[df[target_col] == 1][feature].dropna()]
            
            bp = ax.boxplot(data_to_plot, labels=['Safe (0)', 'Risk (1)'],
                           patch_artist=True, widths=0.6)
            
            # 색상 설정
            colors = ['skyblue', 'salmon']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel(feature, fontsize=9)
            ax.set_title(f'{feature}', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        # 빈 서브플롯 제거
        for idx in range(n_features, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        self._save_plot('06_boxplots_by_label')
        plt.show()
    
    def generate_full_report(self):
        """
        전체 분석 리포트 생성
        """
        print("\n" + "="*60)
        print("Generating Full Analysis Report")
        print("="*60 + "\n")
        
        print("1. Label Distribution Analysis...")
        self.plot_label_distribution()
        
        print("\n2. Feature Distributions...")
        self.plot_feature_distributions(max_features=20)
        
        print("\n3. Correlation Analysis...")
        self.plot_correlation_matrix(top_n=30)
        
        print("\n4. Preliminary Feature Importance...")
        self.plot_feature_importance_preliminary(top_n=20)
        
        print("\n5. Test Group Analysis...")
        self.plot_test_group_analysis()
        
        print("\n6. Boxplot Analysis...")
        self.plot_boxplots_by_label(max_features=12)
        
        # 그룹별 상관관계 (주요 검사만)
        for group in ['A3', 'A5', 'A9']:
            print(f"\n7. Correlation Matrix - {group}...")
            self.plot_correlation_matrix(feature_group=group)
        
        print("\n" + "="*60)
        print("✓ Full Analysis Report Generated")
        print(f"  Plots saved to: {self.plot_dir}")
        print("="*60)


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
        
        # 분석기 생성
        analyzer = DataAnalyzer(config, loader)
        
        # 전체 리포트 생성
        analyzer.generate_full_report()
        
    except Exception as e:
        print(f"Test failed: {e}")
        print("This is expected if train_A.json doesn't exist yet.")