"""
리뷰(Review) API 라우터
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session
from typing import List

from app.database import get_db
from app.services.ReviewService import ReviewService
from app.services.MovieService import MovieService
from app.schemas import ReviewCreate, ReviewResponse, ReviewWithMovie, MovieRating

import logging
from helper_dev_utils import get_auto_logger

logger = get_auto_logger(log_level=logging.DEBUG)


class ReviewRouter:
    """
    리뷰 관련 API 엔드포인트를 관리하는 라우터 클래스
    """

    def __init__(self):
        """
        ReviewRouter 초기화 및 라우트 설정
        """
        self.router = APIRouter(prefix="/reviews", tags=["reviews"])
        self._setup_routes()

    def _setup_routes(self):
        """
        라우트 설정
        """
        self.router.add_api_route(
            "/",
            self.create_review,
            methods=["POST"],
            response_model=ReviewResponse,
            status_code=status.HTTP_201_CREATED,
            summary="리뷰 등록",
            description="새로운 리뷰를 등록합니다. 감성 분석이 자동으로 수행됩니다.",
        )
        self.router.add_api_route(
            "/",
            self.get_recent_reviews,
            methods=["GET"],
            response_model=List[ReviewWithMovie],
            summary="최근 리뷰 목록 조회",
            description="최근 등록된 리뷰 목록을 조회합니다 (기본 10개).",
        )
        self.router.add_api_route(
            "/movie/{tmdb_id}",
            self.get_reviews_by_movie,
            methods=["GET"],
            response_model=List[ReviewResponse],
            summary="특정 영화의 리뷰 목록 조회",
            description="TMDB 영화 ID로 해당 영화의 모든 리뷰를 조회합니다.",
        )
        self.router.add_api_route(
            "/movie/{tmdb_id}/rating",
            self.get_movie_rating,
            methods=["GET"],
            response_model=MovieRating,
            summary="영화 평점 조회",
            description="영화의 AI 평점 및 감성 분석 통계를 조회합니다.",
        )
        self.router.add_api_route(
            "/{review_id}",
            self.get_review,
            methods=["GET"],
            response_model=ReviewWithMovie,
            summary="특정 리뷰 조회",
            description="리뷰 ID로 특정 리뷰를 조회합니다.",
        )
        self.router.add_api_route(
            "/{review_id}",
            self.delete_review,
            methods=["DELETE"],
            status_code=status.HTTP_204_NO_CONTENT,
            summary="리뷰 삭제",
            description="리뷰 ID로 특정 리뷰를 삭제합니다.",
        )

    def create_review(
        self, review_data: ReviewCreate, db: Session = Depends(get_db)
    ) -> ReviewResponse:
        """
        리뷰 등록 (감성 분석 자동 수행)

        Args:
            review_data: 리뷰 등록 데이터
            db: 데이터베이스 세션

        Returns:
            ReviewResponse: 등록된 리뷰 정보

        Raises:
            HTTPException: 영화를 찾을 수 없거나 중복 리뷰인 경우
        """
        # 영화 존재 여부 확인
        movie_service = MovieService(db)
        movie = movie_service.get_movie_by_tmdb_id(review_data.tmdb_id)
        if not movie:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"TMDB 영화 ID {review_data.tmdb_id}를 찾을 수 없습니다.",
            )

        # 리뷰 등록
        review_service = ReviewService(db)
        review = review_service.create_review(review_data)
        return review

    def get_recent_reviews(
        self, limit: int = 10, db: Session = Depends(get_db)
    ) -> List[ReviewWithMovie]:
        """
        최근 리뷰 목록 조회

        Args:
            limit: 조회할 리뷰 개수 (기본값: 10)
            db: 데이터베이스 세션

        Returns:
            List[ReviewWithMovie]: 리뷰 목록 (영화 정보 포함)
        """
        service = ReviewService(db)
        reviews = service.get_all_reviews(limit=limit)
        return reviews

    def get_reviews_by_movie(
        self, tmdb_id: int, db: Session = Depends(get_db)
    ) -> List[ReviewResponse]:
        """
        특정 영화의 리뷰 목록 조회

        Args:
            tmdb_id: TMDB 영화 ID
            db: 데이터베이스 세션

        Returns:
            List[ReviewResponse]: 리뷰 목록

        Raises:
            HTTPException: 영화를 찾을 수 없는 경우
        """
        # 영화 존재 여부 확인
        movie_service = MovieService(db)
        movie = movie_service.get_movie_by_tmdb_id(tmdb_id)
        if not movie:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"TMDB 영화 ID {tmdb_id}를 찾을 수 없습니다.",
            )

        service = ReviewService(db)
        reviews = service.get_reviews_by_tmdb_id(tmdb_id)
        return reviews

    def get_movie_rating(
        self, tmdb_id: int, db: Session = Depends(get_db)
    ) -> MovieRating:
        """
        영화 평점 조회 (AI 평점 및 감성 분석 통계)

        Args:
            tmdb_id: TMDB 영화 ID
            db: 데이터베이스 세션

        Returns:
            MovieRating: 평점 정보

        Raises:
            HTTPException: 영화를 찾을 수 없는 경우
        """
        # 영화 존재 여부 확인
        movie_service = MovieService(db)
        movie = movie_service.get_movie_by_tmdb_id(tmdb_id)
        if not movie:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"TMDB 영화 ID {tmdb_id}를 찾을 수 없습니다.",
            )

        # 평점 계산
        review_service = ReviewService(db)
        rating_data = review_service.get_movie_rating(tmdb_id)

        return MovieRating(
            movie_id=movie.id,
            title=movie.title,
            total_reviews=rating_data["total_reviews"],
            positive_reviews=rating_data["positive_reviews"],
            negative_reviews=rating_data["negative_reviews"],
            positive_ratio=rating_data["positive_ratio"],
            ai_rating=rating_data["ai_rating"],
        )

    def get_review(
        self, review_id: int, db: Session = Depends(get_db)
    ) -> ReviewWithMovie:
        """
        특정 리뷰 조회

        Args:
            review_id: 리뷰 ID
            db: 데이터베이스 세션

        Returns:
            ReviewWithMovie: 리뷰 정보 (영화 정보 포함)

        Raises:
            HTTPException: 리뷰를 찾을 수 없는 경우
        """
        service = ReviewService(db)
        review = service.get_review_by_id(review_id)

        if not review:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"리뷰 ID {review_id}를 찾을 수 없습니다.",
            )

        return review

    def delete_review(self, review_id: int, db: Session = Depends(get_db)) -> None:
        """
        리뷰 삭제

        Args:
            review_id: 리뷰 ID
            db: 데이터베이스 세션

        Raises:
            HTTPException: 리뷰를 찾을 수 없는 경우
        """
        service = ReviewService(db)
        success = service.delete_review(review_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"리뷰 ID {review_id}를 찾을 수 없습니다.",
            )


# 라우터 인스턴스 생성
review_router = ReviewRouter()
router = review_router.router
