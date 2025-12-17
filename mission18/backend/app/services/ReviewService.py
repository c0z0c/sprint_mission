"""
리뷰(Review) 서비스 클래스
"""

from sqlmodel import Session, select
from typing import List, Optional

from app.models.ReviewModel import ReviewModel
from app.schemas import ReviewCreate
from app.ai.SentimentPredictor import SentimentPredictor

import logging
from helper_dev_utils import get_auto_logger

logger = get_auto_logger(log_level=logging.DEBUG)


class ReviewService:
    """
    리뷰 관련 비즈니스 로직을 처리하는 서비스 클래스
    """

    def __init__(self, session: Session):
        """
        ReviewService 초기화

        Args:
            session: 데이터베이스 세션
        """
        self.session = session
        self.sentiment_predictor = SentimentPredictor()

    def create_review(self, review_data: ReviewCreate) -> ReviewModel:
        """
        리뷰 등록 (감성 분석 포함)

        Args:
            review_data: 리뷰 등록 데이터

        Returns:
            ReviewModel: 등록된 리뷰 모델
        """
        # 감성 분석 수행
        is_positive = self.sentiment_predictor.predict(review_data.content)

        # 리뷰 모델 생성
        review = ReviewModel(
            tmdb_id=review_data.tmdb_id,
            author=review_data.author,
            content=review_data.content,
            is_positive=is_positive,
        )

        self.session.add(review)
        self.session.commit()
        self.session.refresh(review)

        return review

    def get_all_reviews(self, limit: int = 10) -> List[ReviewModel]:
        """
        전체 리뷰 목록 조회 (최신순)

        Args:
            limit: 조회할 리뷰 개수 (기본값: 10)

        Returns:
            List[ReviewModel]: 리뷰 모델 리스트
        """
        statement = (
            select(ReviewModel).order_by(ReviewModel.created_at.desc()).limit(limit)
        )
        results = self.session.exec(statement)
        return results.all()

    def get_reviews_by_tmdb_id(self, tmdb_id: int) -> List[ReviewModel]:
        """
        특정 영화의 리뷰 목록 조회

        Args:
            tmdb_id: TMDB 영화 ID

        Returns:
            List[ReviewModel]: 리뷰 모델 리스트
        """
        statement = select(ReviewModel).where(ReviewModel.tmdb_id == tmdb_id)
        results = self.session.exec(statement)
        return results.all()

    def get_review_by_id(self, review_id: int) -> Optional[ReviewModel]:
        """
        특정 리뷰 조회

        Args:
            review_id: 리뷰 ID

        Returns:
            Optional[ReviewModel]: 리뷰 모델 또는 None
        """
        return self.session.get(ReviewModel, review_id)

    def delete_review(self, review_id: int) -> bool:
        """
        리뷰 삭제

        Args:
            review_id: 리뷰 ID

        Returns:
            bool: 삭제 성공 여부
        """
        review = self.session.get(ReviewModel, review_id)
        if not review:
            return False

        self.session.delete(review)
        self.session.commit()
        return True

    def get_movie_rating(self, tmdb_id: int) -> dict:
        """
        영화 평점 조회 (리뷰 감성 분석 기반)

        Args:
            tmdb_id: TMDB 영화 ID

        Returns:
            dict: 평점 정보
        """
        reviews = self.get_reviews_by_tmdb_id(tmdb_id)

        if not reviews:
            return {
                "total_reviews": 0,
                "positive_reviews": 0,
                "negative_reviews": 0,
                "positive_ratio": 0.0,
                "ai_rating": 0.0,
            }

        positive_count = sum(1 for r in reviews if r.is_positive == 1)
        negative_count = sum(1 for r in reviews if r.is_positive == 0)
        total_count = len(reviews)

        positive_ratio = positive_count / total_count if total_count > 0 else 0.0
        ai_rating = positive_ratio * 5.0  # 5점 만점

        return {
            "total_reviews": total_count,
            "positive_reviews": positive_count,
            "negative_reviews": negative_count,
            "positive_ratio": positive_ratio,
            "ai_rating": round(ai_rating, 2),
        }
