"""
리뷰(Review) 서비스 클래스
"""

from sqlmodel import Session, select, func
from sqlalchemy.orm import selectinload
from typing import List, Optional

from app.models.ReviewModel import ReviewModel
from app.models.MovieModel import MovieModel
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
        review_dict = {
            "tmdb_id": review_data.tmdb_id,
            "author": review_data.author,
            "content": review_data.content,
            "is_positive": is_positive,
        }

        # 사용자가 제공한 datetime 사용 (없으면 default_factory 사용)
        if review_data.created_at:
            review_dict["created_at"] = review_data.created_at
        if review_data.updated_at:
            review_dict["updated_at"] = review_data.updated_at

        review = ReviewModel(**review_dict)

        self.session.add(review)
        self.session.commit()
        self.session.refresh(review)

        # 영화 AI 평점 업데이트
        from app.services.MovieService import MovieService

        movie_service = MovieService(self.session)
        movie_service.update_movie_ai_rating(review_data.tmdb_id)

        logger.debug(
            f"[Service] Review created and AI rating updated for TMDB ID: {review_data.tmdb_id}"
        )

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

    def get_reviews_paginated(
        self, page: int = 1, page_size: int = 10, tmdb_ids: List[int] = None
    ) -> tuple[List[ReviewModel], int]:
        """
        페이지네이션된 리뷰 목록 조회 (최신순, TMDB ID 필터링 가능)

        Args:
            page: 페이지 번호 (1부터 시작)
            page_size: 페이지당 항목 수
            tmdb_ids: 필터링할 영화 TMDB ID 목록 (선택 사항)

        Returns:
            tuple[List[ReviewModel], int]: (리뷰 목록, 전체 리뷰 수)
        """
        # 전체 리뷰 수 조회 (tmdb_ids 필터링 적용)
        count_statement = select(func.count(ReviewModel.id))
        if tmdb_ids:
            count_statement = count_statement.where(ReviewModel.tmdb_id.in_(tmdb_ids))
        total = self.session.exec(count_statement).one()

        # 페이지네이션된 리뷰 목록 조회 (최신순, 영화 정보 eager loading, tmdb_ids 필터링 적용)
        offset = (page - 1) * page_size
        statement = (
            select(ReviewModel)
            .options(selectinload(ReviewModel.movie))
            .order_by(ReviewModel.created_at.desc())
        )

        if tmdb_ids:
            statement = statement.where(ReviewModel.tmdb_id.in_(tmdb_ids))

        statement = statement.offset(offset).limit(page_size)
        results = self.session.exec(statement)

        return results.all(), total

    def search_reviews(
        self, filters: "ReviewSearchFilters"
    ) -> tuple[List[ReviewModel], int]:
        """
        리뷰 검색 (복합 필터링, 정렬, 페이지네이션)

        Args:
            filters: 리뷰 검색 필터

        Returns:
            tuple[List[ReviewModel], int]: (리뷰 목록, 전체 검색 결과 수)
        """
        from app.schemas.review import ReviewSearchFilters

        logger.debug(f"[Service] Searching reviews with filters: {filters}")

        # 기본 쿼리 (eager loading)
        statement = select(ReviewModel).options(selectinload(ReviewModel.movie))

        # 필터 적용 (AND 조합)
        if filters.author:
            # 대소문자 무시 검색
            statement = statement.where(ReviewModel.author.ilike(f"%{filters.author}%"))

        if filters.content:
            # 대소문자 무시 검색
            statement = statement.where(
                ReviewModel.content.ilike(f"%{filters.content}%")
            )

        if filters.sentiment != "all":
            # 감성 필터 (positive: 1, negative: 0)
            sentiment_value = 1 if filters.sentiment == "positive" else 0
            statement = statement.where(ReviewModel.is_positive == sentiment_value)

        if filters.tmdb_id is not None:
            statement = statement.where(ReviewModel.tmdb_id == filters.tmdb_id)

        if filters.movie_title:
            # 영화 제목으로 필터 (JOIN 필요)
            statement = statement.join(MovieModel).where(
                MovieModel.title.ilike(f"%{filters.movie_title}%")
            )

        if filters.created_from:
            statement = statement.where(ReviewModel.created_at >= filters.created_from)

        if filters.created_to:
            statement = statement.where(ReviewModel.created_at <= filters.created_to)

        # 전체 검색 결과 수 조회 (필터 적용 후)
        count_statement = select(func.count()).select_from(statement.subquery())
        total = self.session.exec(count_statement).one()

        # 정렬 적용
        sort_field = getattr(ReviewModel, filters.sort_by, ReviewModel.created_at)
        if filters.sort_order == "desc":
            statement = statement.order_by(sort_field.desc())
        else:
            statement = statement.order_by(sort_field.asc())

        # 페이지네이션 적용
        offset = (filters.page - 1) * filters.page_size
        statement = statement.offset(offset).limit(filters.page_size)

        # 실행
        results = self.session.exec(statement)
        reviews = results.all()

        logger.debug(
            f"[Service] Found {total} reviews, returning page {filters.page} with {len(reviews)} items"
        )

        return reviews, total

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

        tmdb_id = review.tmdb_id  # AI 평점 업데이트를 위해 저장

        self.session.delete(review)
        self.session.commit()

        # 영화 AI 평점 업데이트
        from app.services.MovieService import MovieService

        movie_service = MovieService(self.session)
        movie_service.update_movie_ai_rating(tmdb_id)

        logger.debug(
            f"[Service] Review deleted and AI rating updated for TMDB ID: {tmdb_id}"
        )

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

    def update_review(
        self, review_id: int, review_data: "ReviewUpdate"
    ) -> Optional[ReviewModel]:
        """
        리뷰 정보 업데이트 (전체 또는 부분 업데이트 - PUT/PATCH)

        Args:
            review_id: 리뷰 ID
            review_data: 리뷰 업데이트 데이터 (ReviewUpdate 또는 ReviewPatch)

        Returns:
            Optional[ReviewModel]: 업데이트된 리뷰 모델 또는 None

        Raises:
            ValueError: UniqueConstraint 위반 시 (동일한 tmdb_id, author, content 조합)
        """
        from app.schemas.review import ReviewUpdate, ReviewPatch
        from sqlalchemy.exc import IntegrityError
        from datetime import datetime

        logger.debug(f"[Service] update_review started for review ID: {review_id}")

        # 리뷰 조회
        review = self.session.get(ReviewModel, review_id)
        if not review:
            logger.warning(f"[Service] Review not found for ID: {review_id}")
            return None

        # 필드 업데이트 (exclude_unset=True로 PATCH 지원)
        update_dict = review_data.model_dump(exclude_unset=True)

        # content 변경 감지
        content_changed = (
            "content" in update_dict and update_dict["content"] != review.content
        )

        # 필드 업데이트
        for key, value in update_dict.items():
            setattr(review, key, value)

        # content 변경 시 AI 감성 분석 재수행
        if content_changed:
            is_positive = self.sentiment_predictor.predict(review.content)
            review.is_positive = is_positive
            logger.debug(
                f"[Service] Content changed, re-analyzed sentiment: is_positive={is_positive}"
            )

        # updated_at 갱신 (입력값이 있으면 사용, 없으면 자동)
        if "updated_at" in update_dict and update_dict["updated_at"]:
            review.updated_at = update_dict["updated_at"]
            logger.debug(
                f"[Service] Using provided updated_at: {update_dict['updated_at']}"
            )
        else:
            review.updated_at = datetime.now()
            logger.debug("[Service] Using auto-generated updated_at")

        try:
            self.session.add(review)
            self.session.commit()
            self.session.refresh(review)
        except IntegrityError as e:
            self.session.rollback()
            logger.error(
                f"[Service] UniqueConstraint violation for review ID: {review_id} - {str(e)}"
            )
            raise ValueError(
                "동일한 영화에 동일한 작성자가 동일한 내용의 리뷰를 이미 작성했습니다. "
                "리뷰 내용을 변경해주세요."
            )

        # 영화 AI 평점 업데이트 (content 변경 시)
        if content_changed:
            from app.services.MovieService import MovieService

            movie_service = MovieService(self.session)
            movie_service.update_movie_ai_rating(review.tmdb_id)
            logger.debug(
                f"[Service] Movie AI rating updated for TMDB ID: {review.tmdb_id}"
            )

        logger.debug(f"[Service] Review updated successfully: ID={review_id}")
        return review
