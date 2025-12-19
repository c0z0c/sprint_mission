"""
감성 분석 더미 클래스
"""

import logging
from transformers import pipeline
from helper_dev_utils import get_auto_logger

logger = get_auto_logger(log_level=logging.DEBUG)


class SentimentPredictor:
    """
    한국어 영화 리뷰 감성 분석 클래스 (전처리 통합)
    daekeun-ml/koelectra-small-v3-nsmc 모델 사용
    """

    def __init__(self):
        """감성 분석 파이프라인 초기화"""
        from transformers import pipeline

        self.model = pipeline(
            "text-classification", model="daekeun-ml/koelectra-small-v3-nsmc"
        )
        self.max_length = 512  # koelectra 모델 최대 토큰 수
        logger.info("SentimentPredictor 초기화 완료")

    def _predict(self, text: str) -> int:
        """
        텍스트의 감성을 예측합니다.

        Args:
            text: 분석할 텍스트 (원본 그대로 가능)

        Returns:
            int: 0 (부정) 또는 1 (긍정)
        """
        if not text or not isinstance(text, str):
            return 0

        # pipeline이 자동 처리
        result = self.model(text)[0]

        # 레이블 매핑
        is_positive = (
            1 if result["label"].lower() in ["positive", "label_1", "1"] else 0
        )
        return is_positive

    def _predict_batch(self, texts: list[str]) -> list[int]:
        """
        여러 텍스트의 감성을 일괄 예측합니다.

        Args:
            texts: 분석할 텍스트 리스트

        Returns:
            list[int]: 각 텍스트의 예측 결과 (0: 부정, 1: 긍정)
        """

        if not texts:
            logger.warning("유효한 텍스트가 없습니다.")
            return [0] * len(texts)

        # 배치 감성 분석
        results = self.model(texts)

        # 레이블 매핑
        predictions = []
        for result in results:
            raw_label = result["label"]
            is_positive = 1 if raw_label.lower() in ["positive", "label_1", "1"] else 0
            predictions.append(is_positive)

        # logger.debug(f"배치 예측 완료: {len(predictions)}개 텍스트")
        return predictions

    def predict(self, text: str) -> int:
        """
        512 보다 긴 텍스트의 경우 문장 단위 혹은 길이 단위로 나워 감정 분류를 한다음에
        가장 많이 나온 결과를 반환합니다.
        동일 할경우 부정을 반환 합니다.
        20%를 겹치게 한다.

        Args:
            text: 분석할 텍스트

        Returns:
            텍스트의 예측 결과 (0: 부정, 1: 긍정)
        """

        if len(text) > (self.max_length * 1.2):
            texts = []
            for i in range(0, len(text), int(self.max_length * 0.8)):
                chunk = text[i : i + self.max_length]
                texts.append(chunk)
            predictions = self.predict_batch(texts)
            pos_count = predictions.count(1)
            neg_count = predictions.count(0)

            # 가장 많이 나온 결과 반환
            # 동일 할경우 부정을 반환
            if pos_count > neg_count:
                return 1
            else:
                return 0
        else:
            return self._predict(text)

    def predict_batch(self, texts: list[str]) -> list[int]:
        return self._predict_batch(texts)

    def predict_texts(self, texts: list[str]) -> list[int]:
        """
        여러 텍스트의 감성을 일괄 예측합니다.
        긴 텍스트는 청크로 분할 후 배치 처리합니다.

        Args:
            texts: 분석할 텍스트 리스트

        Returns:
            list[int]: 각 텍스트의 예측 결과 (0: 부정, 1: 긍정)
        """
        if not texts:
            logger.warning("유효한 텍스트가 없습니다.")
            return [0] * len(texts)

        all_chunks = []
        chunk_counts = []  # 각 텍스트의 청크 개수

        # 모든 텍스트를 청크로 분할
        for text in texts:
            if len(text) > (self.max_length * 1.2):
                chunks = []
                for i in range(0, len(text), int(self.max_length * 0.8)):
                    chunk = text[i : i + self.max_length]
                    chunks.append(chunk)
                all_chunks.extend(chunks)
                chunk_counts.append(len(chunks))
            else:
                all_chunks.append(text)
                chunk_counts.append(1)

        # 모든 청크를 한 번에 배치 처리
        all_predictions = self._predict_batch(all_chunks)

        # 청크별 예측을 원본 텍스트 단위로 재조합
        results = []
        idx = 0
        for count in chunk_counts:
            chunk_preds = all_predictions[idx : idx + count]
            if count == 1:
                results.append(chunk_preds[0])
            else:
                # 다수결 투표 (동점 시 부정)
                pos_count = chunk_preds.count(1)
                neg_count = chunk_preds.count(0)
                results.append(1 if pos_count > neg_count else 0)
            idx += count

        return results
