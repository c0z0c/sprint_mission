# -*- coding: utf-8 -*-
"""MNIST 예측 결과 시각화 API

이 모듈은 MNIST 예측 결과를 시각화하는 다양한 방법을 제공합니다.
"""

import logging
import os
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
from src.visualization.VisualizationManager import VisualizationManager

logger = get_auto_logger(log_level=logging.DEBUG)

# ============================================================================
# 더미 데이터 생성
# ============================================================================


def generate_dummy_visualization_data():
    """시각화 테스트용 더미 데이터를 생성합니다.

    Returns:
        probabilities, predicted_label, confidence, dummy_image
    """
    # 랜덤 확률 생성
    logits = np.random.randn(10)
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / np.sum(exp_logits)

    predicted_label = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_label])

    # 더미 이미지 (28x28)
    dummy_image = np.random.randint(0, 255, (28, 28), dtype=np.uint8)

    return probabilities, predicted_label, confidence, dummy_image


# ============================================================================
# 사용 예시 및 테스트
# ============================================================================


def main():
    """사용 예시"""
    logger.debug("=" * 60)
    logger.debug("시각화 API 테스트")
    logger.debug("=" * 60)

    # 더미 데이터 생성
    probs, label, conf, img = generate_dummy_visualization_data()

    logger.debug(f"\n예측 레이블: {label}")
    logger.debug(f"신뢰도: {conf:.2%}")
    logger.debug(f"확률 분포: {probs}")

    # 1. 막대 차트
    logger.debug("\n[1] 막대 차트 생성 중...")
    viz = PredictionVisualizer()
    fig1 = viz.plot_bar_chart(probs, label)
    plt.savefig("test_bar_chart.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    logger.debug("저장 완료: test_bar_chart.png")

    # 2. 가로 막대 차트
    logger.debug("\n[2] 가로 막대 차트 생성 중...")
    fig2 = viz.plot_horizontal_bar_chart(probs, label, top_k=5)
    plt.savefig("test_horizontal_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    logger.debug("저장 완료: test_horizontal_bar.png")

    # 3. 파이 차트
    logger.debug("\n[3] 파이 차트 생성 중...")
    fig3 = viz.plot_pie_chart(probs, label)
    plt.savefig("test_pie_chart.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    logger.debug("저장 완료: test_pie_chart.png")

    # 4. 통합 대시보드
    logger.debug("\n[4] 통합 대시보드 생성 중...")
    manager = VisualizationManager()
    fig4 = manager.create_prediction_dashboard(probs, label, conf, img)
    plt.savefig("test_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)
    logger.debug("저장 완료: test_dashboard.png")

    logger.debug("\n" + "=" * 60)
    logger.debug("테스트 완료!")
    logger.debug("=" * 60)


if __name__ == "__main__":
    main()
