"""
리뷰(Review) API 라우터
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlmodel import Session
from typing import List, Optional
from datetime import datetime
import math

from app.database import get_db
from app.services.ReviewService import ReviewService
from app.services.MovieService import MovieService
from app.schemas import (
    ReviewCreate,
    ReviewResponse,
    ReviewWithMovie,
    MovieRating,
    ReviewPaginationResponse,
    ReviewUpdate,
    ReviewPatch,
)

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
            "/paginated",
            self.get_reviews_paginated,
            methods=["GET"],
            response_model=ReviewPaginationResponse,
            summary="페이지네이션된 리뷰 목록 조회",
            description="페이지 단위로 리뷰 목록을 조회합니다 (영화 정보 포함).",
        )
        self.router.add_api_route(
            "/search",
            self.search_reviews,
            methods=["GET"],
            response_model=ReviewPaginationResponse,
            summary="리뷰 검색 (복합 필터)",
            description="다양한 조건으로 리뷰를 검색합니다. 모든 필터는 AND 조합으로 작동합니다.",
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
            self.update_review_put,
            methods=["PUT"],
            response_model=ReviewResponse,
            summary="리뷰 전체 업데이트",
            description="리뷰 ID로 특정 리뷰의 모든 필드를 업데이트합니다. content 변경 시 AI 감성 분석이 자동으로 재수행되며, 영화의 AI 평점이 업데이트됩니다. tmdb_id는 수정 불가합니다.",
        )
        self.router.add_api_route(
            "/{review_id}",
            self.update_review_patch,
            methods=["PATCH"],
            response_model=ReviewResponse,
            summary="리뷰 부분 업데이트",
            description="리뷰 ID로 특정 리뷰의 일부 필드만 선택적으로 업데이트합니다. content 변경 시 AI 감성 분석이 자동으로 재수행되며, 영화의 AI 평점이 업데이트됩니다. tmdb_id는 수정 불가합니다.",
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

    def get_reviews_paginated(
        self,
        page: int = Query(1, ge=1, description="페이지 번호 (1부터 시작)"),
        page_size: int = Query(
            10, ge=1, le=100, description="페이지당 항목 수 (최대 100)"
        ),
        tmdb_ids: Optional[str] = Query(
            None, description="필터링할 영화 TMDB ID 목록 (쉼표로 구분)"
        ),
        db: Session = Depends(get_db),
    ) -> ReviewPaginationResponse:
        """
        페이지네이션된 리뷰 목록 조회 (영화 정보 포함)

        Args:
            page: 페이지 번호 (1부터 시작)
            page_size: 페이지당 항목 수
            tmdb_ids: 필터링할 영화 TMDB ID 목록 (쉼표로 구분)
            db: 데이터베이스 세션

        Returns:
            ReviewPaginationResponse: 페이지네이션 정보와 리뷰 목록 (영화 정보 포함)
        """
        service = ReviewService(db)

        # tmdb_ids 문자열을 정수 리스트로 변환
        tmdb_id_list = None
        if tmdb_ids:
            try:
                tmdb_id_list = [int(id.strip()) for id in tmdb_ids.split(",")]
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid tmdb_ids format")

        reviews, total = service.get_reviews_paginated(page, page_size, tmdb_id_list)

        # 전체 페이지 수 계산
        total_pages = math.ceil(total / page_size) if total > 0 else 0

        return ReviewPaginationResponse(
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            reviews=reviews,  # SQLModel의 관계 자동 로드 활용
        )

    def search_reviews(
        self,
        author: Optional[str] = Query(
            None, description="작성자 이름 (부분 검색, 대소문자 무시)"
        ),
        content: Optional[str] = Query(
            None, description="리뷰 내용 (부분 검색, 대소문자 무시)"
        ),
        sentiment: str = Query(
            "all", description="감성 필터 (positive, negative, all)"
        ),
        movie_title: Optional[str] = Query(
            None, description="영화 제목 (부분 검색, 대소문자 무시)"
        ),
        tmdb_id: Optional[int] = Query(None, description="TMDB 영화 ID"),
        created_from: Optional[datetime] = Query(
            None, description="생성일 시작 (ISO 8601 형식)"
        ),
        created_to: Optional[datetime] = Query(
            None, description="생성일 종료 (ISO 8601 형식)"
        ),
        sort_by: str = Query(
            "created_at", description="정렬 필드 (created_at, author)"
        ),
        sort_order: str = Query("desc", description="정렬 방향 (asc, desc)"),
        page: int = Query(1, ge=1, description="페이지 번호 (1부터 시작)"),
        page_size: int = Query(
            10, ge=1, le=100, description="페이지당 항목 수 (최대 100)"
        ),
        db: Session = Depends(get_db),
    ) -> ReviewPaginationResponse:
        """
        리뷰 검색 (복합 필터, 정렬, 페이지네이션)

        모든 필터는 AND 조합으로 작동합니다.

        Query Parameters:
            - author: 작성자 이름 부분 검색 (대소문자 무시)
            - content: 리뷰 내용 부분 검색 (대소문자 무시)
            - sentiment: 감성 필터 (positive, negative, all)
            - movie_title: 영화 제목 부분 검색 (대소문자 무시)
            - tmdb_id: TMDB 영화 ID
            - created_from: 생성일 시작 (ISO 8601 형식, 예: 2024-01-01T00:00:00)
            - created_to: 생성일 종료 (ISO 8601 형식)
            - sort_by: 정렬 필드 (created_at, author)
            - sort_order: 정렬 방향 (asc, desc)
            - page: 페이지 번호
            - page_size: 페이지당 항목 수

        Returns:
            ReviewPaginationResponse: 검색 결과 및 페이지네이션 정보
        """
        from app.schemas.review import ReviewSearchFilters

        # 필터 객체 생성
        filters = ReviewSearchFilters(
            author=author,
            content=content,
            sentiment=sentiment,
            movie_title=movie_title,
            tmdb_id=tmdb_id,
            created_from=created_from,
            created_to=created_to,
            sort_by=sort_by,
            sort_order=sort_order,
            page=page,
            page_size=page_size,
        )

        service = ReviewService(db)
        reviews, total = service.search_reviews(filters)

        total_pages = math.ceil(total / page_size) if total > 0 else 0

        logger.debug(
            f"[Router] Review search completed: {total} results, page {page}/{total_pages}"
        )

        return ReviewPaginationResponse(
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            reviews=reviews,
        )

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

    def update_review_put(
        self,
        review_id: int,
        review_data: ReviewUpdate,
        db: Session = Depends(get_db),
    ) -> ReviewResponse:
        """
        리뷰 전체 업데이트 (PUT)

        Args:
            review_id: 리뷰 ID
            review_data: 리뷰 업데이트 데이터 (전체 필드)
            db: 데이터베이스 세션

        Returns:
            ReviewResponse: 업데이트된 리뷰 정보

        Raises:
            HTTPException: 리뷰를 찾을 수 없거나 UniqueConstraint 위반 시
        """
        service = ReviewService(db)

        try:
            updated_review = service.update_review(review_id, review_data)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )

        if not updated_review:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"리뷰 ID {review_id}를 찾을 수 없습니다.",
            )

        return updated_review

    def update_review_patch(
        self,
        review_id: int,
        review_data: ReviewPatch,
        db: Session = Depends(get_db),
    ) -> ReviewResponse:
        """
        리뷰 부분 업데이트 (PATCH)

        Args:
            review_id: 리뷰 ID
            review_data: 리뷰 업데이트 데이터 (선택적 필드)
            db: 데이터베이스 세션

        Returns:
            ReviewResponse: 업데이트된 리뷰 정보

        Raises:
            HTTPException: 리뷰를 찾을 수 없거나 UniqueConstraint 위반 시
        """
        service = ReviewService(db)

        try:
            updated_review = service.update_review(review_id, review_data)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )

        if not updated_review:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"리뷰 ID {review_id}를 찾을 수 없습니다.",
            )

        return updated_review


# 라우터 인스턴스 생성
review_router = ReviewRouter()
router = review_router.router
