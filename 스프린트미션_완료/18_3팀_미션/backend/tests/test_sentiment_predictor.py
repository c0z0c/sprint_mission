"""
SentimentPredictor 단위 테스트
AI 감성 분석 모델이 랜덤이 아닌 실제 감성 분류를 수행하는지 검증
"""

import pytest
from app.ai.SentimentPredictor import SentimentPredictor


class TestSentimentPredictor:
    """SentimentPredictor 단위 테스트 클래스"""

    @pytest.fixture(scope="class")
    def predictor(self):
        """모든 테스트에서 공유할 predictor 인스턴스 (모델 한 번만 로딩)"""
        return SentimentPredictor()

    def test_model_initialization(self, predictor):
        """모델 초기화 검증"""
        assert predictor is not None
        assert predictor.model is not None
        assert predictor.max_length == 512

    def test_predict_positive_samples(self, predictor):
        """긍정 샘플 예측 테스트"""
        positive_samples = [
            "이 영화는 정말 최고였어요! 감동적이고 재미있었습니다.",
            "배우들의 연기가 환상적이었고 스토리도 훌륭했습니다.",
            "꼭 봐야 할 명작입니다. 강력 추천합니다!",
            "완벽한 영화! 다시 보고 싶어요.",
        ]

        for text in positive_samples:
            result = predictor.predict(text)
            assert result == 1, f"긍정 텍스트가 부정으로 분류됨: {text}"

    def test_predict_negative_samples(self, predictor):
        """부정 샘플 예측 테스트"""
        negative_samples = [
            "최악의 영화였습니다. 돈과 시간이 아까웠어요.",
            "지루하고 재미없었습니다. 중간에 나왔어요.",
            "스토리도 엉망이고 연기도 형편없었습니다.",
            "다시는 보고 싶지 않은 영화입니다.",
        ]

        for text in negative_samples:
            result = predictor.predict(text)
            assert result == 0, f"부정 텍스트가 긍정으로 분류됨: {text}"

    def test_predict_batch(self, predictor):
        """배치 예측 테스트"""
        texts = [
            "정말 훌륭한 영화였습니다!",
            "최악이었어요.",
            "감동적이고 재미있었습니다.",
            "지루하고 형편없었습니다.",
        ]
        expected = [1, 0, 1, 0]

        results = predictor.predict_batch(texts)

        assert len(results) == len(texts)
        assert results == expected, f"예상: {expected}, 실제: {results}"

    def test_long_text_chunking(self, predictor):
        """긴 텍스트 청크 분할 처리 테스트 (512토큰 초과)"""
        # 긍정 텍스트를 반복하여 긴 텍스트 생성
        long_positive_text = "이 영화는 정말 최고였어요! " * 200

        result = predictor.predict(long_positive_text)
        assert result == 1, "긴 긍정 텍스트가 부정으로 분류됨"

        # 부정 텍스트를 반복하여 긴 텍스트 생성
        long_negative_text = "최악의 영화였습니다. " * 200

        result = predictor.predict(long_negative_text)
        assert result == 0, "긴 부정 텍스트가 긍정으로 분류됨"

    def test_predict_texts_batch_with_long_texts(self, predictor):
        """여러 긴 텍스트 일괄 처리 테스트"""
        texts = [
            "정말 훌륭한 영화였습니다! " * 100,
            "최악이었어요. " * 100,
            "감동적이고 재미있었습니다. " * 100,
        ]
        expected = [1, 0, 1]

        results = predictor.predict_texts(texts)

        assert len(results) == len(texts)
        assert results == expected, f"예상: {expected}, 실제: {results}"

    def test_edge_case_empty_string(self, predictor):
        """엣지 케이스: 빈 문자열"""
        result = predictor._predict("")
        assert result == 0, "빈 문자열은 부정(0)을 반환해야 함"

    def test_edge_case_none(self, predictor):
        """엣지 케이스: None"""
        result = predictor._predict(None)
        assert result == 0, "None은 부정(0)을 반환해야 함"

    def test_edge_case_special_characters(self, predictor):
        """엣지 케이스: 특수문자만 포함된 텍스트"""
        result = predictor.predict("!@#$%^&*()")
        assert isinstance(result, int)
        assert result in [0, 1]

    def test_mixed_sentiment_majority_vote(self, predictor):
        """혼합 감성 텍스트의 다수결 투표 로직 테스트"""
        # 긍정 내용이 더 많은 긴 텍스트
        mixed_text = (
            "정말 훌륭한 영화였습니다! " * 150 + "조금 아쉬운 부분도 있었지만 " * 50
        )

        result = predictor.predict(mixed_text)
        assert result == 1, "긍정이 더 많은 혼합 텍스트는 긍정으로 분류되어야 함"

    def test_prediction_consistency(self, predictor):
        """동일 텍스트에 대한 예측 재현성 테스트"""
        text = "이 영화는 정말 감동적이었습니다."

        result1 = predictor.predict(text)
        result2 = predictor.predict(text)
        result3 = predictor.predict(text)

        assert (
            result1 == result2 == result3
        ), "동일 텍스트에 대한 예측 결과가 일관되어야 함"

    def test_batch_vs_single_prediction(self, predictor):
        """배치 예측과 단일 예측 결과 일치 검증"""
        texts = [
            "정말 훌륭한 영화였습니다!",
            "최악이었어요.",
            "감동적이고 재미있었습니다.",
        ]

        # 단일 예측
        single_results = [predictor.predict(text) for text in texts]

        # 배치 예측
        batch_results = predictor.predict_batch(texts)

        assert (
            single_results == batch_results
        ), "배치 예측과 단일 예측 결과가 일치해야 함"
