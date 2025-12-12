# -*- coding: utf-8 -*-
"""MNIST 예측 결과 시각화 API - 통합 시각화 매니저

이 모듈은 모든 시각화 기능을 통합 관리하는 VisualizationManager 클래스를 제공합니다.

주요 기능:
    - ImageVisualizer와 PredictionVisualizer 통합 관리
    - 예측 결과 대시보드 생성
    - 단일 인터페이스로 모든 시각화 접근

구조:
    VisualizationManager
    ├── prediction_viz: PredictionVisualizer
    └── image_viz: ImageVisualizer

사용 예:
    manager = VisualizationManager()
    fig = manager.prediction_viz.plot_bar_chart(probs, label)
    dashboard = manager.create_prediction_dashboard(probs, label, confidence)
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from helper_dev_utils import get_auto_logger
from helper_plot_hangul import matplotlib_font_reset
from matplotlib.figure import Figure

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.visualization.ImageVisualizer import ImageVisualizer
from src.visualization.PredictionVisualizer import PredictionVisualizer

logger = get_auto_logger(log_level=logging.DEBUG)

# ============================================================================
# 통합 시각화 매니저
# ============================================================================


class VisualizationManager:
    """모든 시각화 기능을 통합 관리하는 클래스"""

    def __init__(self):
        """시각화 매니저를 초기화합니다."""
        self.prediction_viz = PredictionVisualizer()
        self.image_viz = ImageVisualizer()

    def create_prediction_dashboard(
        self,
        probabilities: np.ndarray,
        predicted_label: int,
        confidence: float,
        preprocessed_image: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (12, 8),
    ) -> Figure:
        """예측 결과를 종합적으로 보여주는 대시보드를 생성합니다.

        전처리 이미지, 예측 레이블, 신뢰도, 확률 분포를 한 화면에 표시합니다.

        Args:
            probabilities: 각 클래스별 확률 배열 (10개)
            predicted_label: 예측된 레이블 (0-9)
            confidence: 신뢰도 (0.0 ~ 1.0)
            preprocessed_image: 전처리된 28x28 이미지 (선택적)
            figsize: Figure 크기 (width, height)

        Returns:
            matplotlib Figure 객체
        """
        if preprocessed_image is not None:
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

            # 전처리 이미지
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(preprocessed_image, cmap="gray")
            ax1.set_title("전처리된 이미지 (28x28)", fontsize=12, fontweight="bold")
            ax1.axis("off")

            # 예측 레이블 및 신뢰도
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.text(
                0.5,
                0.6,
                f"{predicted_label}",
                ha="center",
                va="center",
                fontsize=80,
                fontweight="bold",
                color=self.prediction_viz.highlight_color,
            )
            ax2.text(
                0.5,
                0.2,
                f"신뢰도: {confidence:.1%}",
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
            )
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.axis("off")

            # 막대 차트
            ax3 = fig.add_subplot(gs[1, :])
            labels = [str(i) for i in range(10)]
            colors = [
                (
                    self.prediction_viz.highlight_color
                    if i == predicted_label
                    else self.prediction_viz.normal_color
                )
                for i in range(10)
            ]
            bars = ax3.bar(labels, probabilities, color=colors, alpha=0.8)
            bars[predicted_label].set_edgecolor("black")
            bars[predicted_label].set_linewidth(2.5)

            for i, (bar, prob) in enumerate(zip(bars, probabilities)):
                height = bar.get_height()
                ax3.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{prob:.1%}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            ax3.set_xlabel("숫자 (Digit)", fontsize=12, fontweight="bold")
            ax3.set_ylabel("확률 (Probability)", fontsize=12, fontweight="bold")
            ax3.set_title("예측 확률 분포", fontsize=12, fontweight="bold")
            ax3.set_ylim(0, 1.0)
            ax3.grid(axis="y", alpha=0.3, linestyle="--")

        else:
            # 이미지가 없는 경우 간단한 버전
            fig = self.prediction_viz.plot_bar_chart(
                probabilities,
                predicted_label,
                title=f"예측: {predicted_label} (신뢰도: {confidence:.1%})",
            )

        return fig
