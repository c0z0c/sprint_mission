# -*- coding: utf-8 -*-
"""MNIST 예측 결과 시각화 API - 클래스 기반 설계

이 모듈은 MNIST 예측 결과를 시각화하는 다양한 방법을 제공합니다.
"""

from typing import Optional, Tuple
import logging
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from helper_plot_hangul import matplotlib_font_reset
from helper_dev_utils import get_auto_logger

logger = get_auto_logger(log_level=logging.DEBUG)


# ============================================================================
# 예측 결과 시각화 클래스
# ============================================================================


class PredictionVisualizer:
    """MNIST 예측 결과를 시각화하는 클래스

    막대 차트, 파이 차트, 히트맵 등 다양한 시각화 방법을 제공합니다.
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (8, 4),
        highlight_color: str = "#ff6b6b",
        normal_color: str = "#4ecdc4",
    ):
        """
        Args:
            figsize: Figure 크기 (width, height)
            highlight_color: 예측된 클래스 강조 색상
            normal_color: 일반 클래스 색상
        """
        self.figsize = figsize
        self.highlight_color = highlight_color
        self.normal_color = normal_color

    def plot_bar_chart(
        self,
        probabilities: np.ndarray,
        predicted_label: int,
        title: Optional[str] = None,
    ) -> Figure:
        """예측 확률을 막대 차트로 시각화합니다.

        Args:
            probabilities: 각 클래스별 확률 (10개)
            predicted_label: 예측된 레이블
            title: 차트 제목 (None일 경우 기본 제목)

        Returns:
            matplotlib Figure 객체
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # 레이블 및 색상 설정 (정수 사용으로 경고 해소)
        labels = list(range(10))
        colors = [
            self.highlight_color if i == predicted_label else self.normal_color
            for i in range(10)
        ]

        # 막대 차트 그리기
        bars = ax.bar(labels, probabilities, color=colors, alpha=0.8)

        # 예측된 레이블 막대 강조
        bars[predicted_label].set_edgecolor("black")
        bars[predicted_label].set_linewidth(2.5)

        # 각 막대 위에 확률 값 표시
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{prob:.1%}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold" if i == predicted_label else "normal",
            )

        # 축 및 제목 설정
        ax.set_xlabel("숫자 (Digit)", fontsize=12, fontweight="bold")
        ax.set_ylabel("확률 (Probability)", fontsize=12, fontweight="bold")
        ax.set_xticks(labels)  # X축 틱 명시

        if title is None:
            title = "예측 확률 분포 (Prediction Probability Distribution)"
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        plt.tight_layout(pad=1.5)
        plt.close(fig)
        return fig

    def plot_compact_bar_chart(
        self, probabilities: np.ndarray, predicted_label: int
    ) -> Figure:
        """예측 확률을 작은 막대 차트로 시각화합니다 (히스토리 표시용).

        Args:
            probabilities: 각 클래스별 확률 (10개)
            predicted_label: 예측된 레이블

        Returns:
            matplotlib Figure 객체
        """
        fig, ax = plt.subplots(figsize=(1.5, 0.22))

        # 정수 레이블 사용
        labels = list(range(10))
        colors = [
            self.highlight_color if i == predicted_label else self.normal_color
            for i in range(10)
        ]

        bars = ax.bar(labels, probabilities, color=colors, alpha=0.8)

        # 예측된 막대에만 퍼센트 표시
        height = bars[predicted_label].get_height()
        ax.text(
            bars[predicted_label].get_x() + bars[predicted_label].get_width() / 2.0,
            height,
            f"{probabilities[predicted_label]:.1%}",
            ha="center",
            va="bottom",
            fontsize=6,
        )

        # 단순화된 축
        ax.set_ylim(0, 1.0)
        ax.set_xticks(labels)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.tick_params(labelsize=7)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # tight_layout() 제거 (작은 figsize에서 경고 발생)
        plt.close(fig)
        return fig

    def plot_horizontal_bar_chart(
        self,
        probabilities: np.ndarray,
        predicted_label: int,
        top_k: int = 5,
        title: Optional[str] = None,
    ) -> Figure:
        """상위 K개의 예측 확률을 가로 막대 차트로 시각화합니다.

        Args:
            probabilities: 각 클래스별 확률 (10개)
            predicted_label: 예측된 레이블
            top_k: 표시할 상위 클래스 개수
            title: 차트 제목

        Returns:
            matplotlib Figure 객체
        """
        # 상위 K개 인덱스 추출
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        top_probs = probabilities[top_indices]
        top_labels = list(top_indices)  # 정수 리스트

        # 색상 설정
        colors = [
            self.highlight_color if i == predicted_label else self.normal_color
            for i in top_indices
        ]

        fig, ax = plt.subplots(figsize=(8, top_k * 0.6))

        # 가로 막대 차트
        bars = ax.barh(top_labels, top_probs, color=colors, alpha=0.8)

        # 예측된 레이블 강조
        for i, idx in enumerate(top_indices):
            if idx == predicted_label:
                bars[i].set_edgecolor("black")
                bars[i].set_linewidth(2.5)

        # 확률 값 표시
        for i, (bar, prob) in enumerate(zip(bars, top_probs)):
            width = bar.get_width()
            ax.text(
                width,
                bar.get_y() + bar.get_height() / 2.0,
                f" {prob:.2%}",
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold" if top_indices[i] == predicted_label else "normal",
            )

        ax.set_xlabel("확률 (Probability)", fontsize=12, fontweight="bold")
        ax.set_ylabel("숫자 (Digit)", fontsize=12, fontweight="bold")
        ax.set_yticks(top_labels)

        if title is None:
            title = f"상위 {top_k}개 예측 확률"
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

        ax.set_xlim(0, 1.0)
        ax.grid(axis="x", alpha=0.3, linestyle="--")

        plt.tight_layout(pad=1.5)
        plt.close(fig)
        return fig

    def plot_pie_chart(
        self,
        probabilities: np.ndarray,
        predicted_label: int,
        threshold: float = 0.05,
        title: Optional[str] = None,
    ) -> Figure:
        """예측 확률을 파이 차트로 시각화합니다.

        Args:
            probabilities: 각 클래스별 확률 (10개)
            predicted_label: 예측된 레이블
            threshold: 표시할 최소 확률 (이하는 '기타'로 묶음)
            title: 차트 제목

        Returns:
            matplotlib Figure 객체
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # threshold 이상인 클래스만 표시
        labels = []
        values = []
        colors = []

        other_sum = 0.0
        for i, prob in enumerate(probabilities):
            if prob >= threshold:
                labels.append(f"숫자 {i}")
                values.append(prob)
                colors.append(
                    self.highlight_color if i == predicted_label else self.normal_color
                )
            else:
                other_sum += prob

        # 기타 항목 추가
        if other_sum > 0:
            labels.append("기타")
            values.append(other_sum)
            colors.append("#cccccc")

        # 파이 차트 그리기
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            explode=[
                0.1 if label == f"숫자 {predicted_label}" else 0 for label in labels
            ],
        )

        # 텍스트 스타일 설정
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
            autotext.set_fontsize(10)

        if title is None:
            title = "예측 확률 분포 (파이 차트)"
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

        plt.tight_layout(pad=1.5)
        plt.close(fig)
        return fig

    def plot_confidence_gauge(
        self, confidence: float, predicted_label: int, title: Optional[str] = None
    ) -> Figure:
        """신뢰도를 게이지 형태로 시각화합니다.

        Args:
            confidence: 신뢰도 (0.0 ~ 1.0)
            predicted_label: 예측된 레이블
            title: 차트 제목

        Returns:
            matplotlib Figure 객체
        """
        fig, ax = plt.subplots(figsize=(6, 3))

        # 게이지 바 그리기
        ax.barh(0, confidence, height=0.5, color=self._get_confidence_color(confidence))
        ax.barh(0, 1 - confidence, height=0.5, left=confidence, color="#e0e0e0")

        # 신뢰도 텍스트
        ax.text(
            0.5,
            0,
            f"{confidence:.1%}",
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
            color="white",
        )

        # 예측 레이블 표시
        ax.text(
            0.5,
            -1,
            f"예측: {predicted_label}",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(-1.5, 0.5)
        ax.axis("off")

        if title is None:
            title = "예측 신뢰도"
        fig.suptitle(title, fontsize=14, fontweight="bold")

        plt.tight_layout(pad=1.5)
        plt.close(fig)
        return fig

    def _get_confidence_color(self, confidence: float) -> str:
        """신뢰도에 따른 색상을 반환합니다.

        Args:
            confidence: 신뢰도 (0.0 ~ 1.0)

        Returns:
            색상 코드
        """
        if confidence >= 0.8:
            return "#4caf50"  # 녹색 (높음)
        elif confidence >= 0.5:
            return "#ff9800"  # 주황색 (중간)
        else:
            return "#f44336"  # 빨간색 (낮음)
