"""
감성 분석 더미 클래스
"""

import random


class SentimentPredictor:
    """
    리뷰 텍스트의 감성을 분석하는 더미 클래스
    실제 AI 모델로 대체될 예정
    """

    def __init__(self):
        """
        SentimentPredictor 초기화
        """
        pass

    def predict(self, text: str) -> int:
        """
        텍스트의 감성을 예측 (더미 버전 - 랜덤 반환)

        Args:
            text: 분석할 리뷰 텍스트

        Returns:
            int: 0 (부정) 또는 1 (긍정)
        """
        # 더미 구현: 랜덤하게 0 또는 1 반환
        return random.randint(0, 1)
