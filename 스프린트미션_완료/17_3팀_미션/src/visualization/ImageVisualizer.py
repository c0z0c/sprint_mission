# -*- coding: utf-8 -*-
"""MNIST 예측 결과 시각화 API - 이미지 시각화

이 모듈은 이미지 관련 시각화를 담당하는 ImageVisualizer 클래스를 제공합니다.

주요 기능:
    - 전처리 단계별 이미지 비교 표시
    - 두 이미지 나란히 비교 시각화
    - matplotlib을 사용한 이미지 렌더링

사용 예:
    viz = ImageVisualizer()
    fig = viz.plot_preprocessing_steps(original, grayscale, resized, final)
    fig = viz.plot_side_by_side(image1, image2, "Before", "After")
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
logger = get_auto_logger(log_level=logging.DEBUG)

# ============================================================================
# 이미지 시각화 클래스
# ============================================================================


class ImageVisualizer:
    """이미지 관련 시각화를 담당하는 클래스"""

    @staticmethod
    def plot_preprocessing_steps(
        original_image: np.ndarray,
        grayscale_image: np.ndarray,
        resized_image: np.ndarray,
        final_image: np.ndarray,
        figsize: Tuple[int, int] = (12, 3),
    ) -> Figure:
        """전처리 단계별 이미지를 시각화합니다.

        Args:
            original_image: 원본 이미지
            grayscale_image: 그레이스케일 이미지
            resized_image: 리사이즈된 이미지
            final_image: 최종 전처리 이미지
            figsize: Figure 크기

        Returns:
            matplotlib Figure 객체
        """
        fig, axes = plt.subplots(1, 4, figsize=figsize)

        steps = [
            ("원본 이미지", original_image),
            ("그레이스케일", grayscale_image),
            ("리사이즈 (28x28)", resized_image),
            ("반전 & 정규화", final_image),
        ]

        for ax, (title, img) in zip(axes, steps):
            if len(img.shape) == 3 and img.shape[2] == 4:
                # RGBA
                ax.imshow(img)
            elif len(img.shape) == 3:
                # RGB
                ax.imshow(img)
            else:
                # 그레이스케일
                ax.imshow(img, cmap="gray")

            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.axis("off")

        plt.tight_layout(pad=1.5)
        return fig

    @staticmethod
    def plot_side_by_side(
        image1: np.ndarray,
        image2: np.ndarray,
        title1: str = "이미지 1",
        title2: str = "이미지 2",
        figsize: Tuple[int, int] = (8, 4),
    ) -> Figure:
        """두 이미지를 나란히 비교하여 표시합니다.

        Args:
            image1: 첫 번째 이미지
            image2: 두 번째 이미지
            title1: 첫 번째 이미지 제목
            title2: 두 번째 이미지 제목
            figsize: Figure 크기

        Returns:
            matplotlib Figure 객체
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # 첫 번째 이미지
        if len(image1.shape) == 2:
            axes[0].imshow(image1, cmap="gray")
        else:
            axes[0].imshow(image1)
        axes[0].set_title(title1, fontsize=12, fontweight="bold")
        axes[0].axis("off")

        # 두 번째 이미지
        if len(image2.shape) == 2:
            axes[1].imshow(image2, cmap="gray")
        else:
            axes[1].imshow(image2)
        axes[1].set_title(title2, fontsize=12, fontweight="bold")
        axes[1].axis("off")

        plt.tight_layout(pad=1.5)
        return fig
