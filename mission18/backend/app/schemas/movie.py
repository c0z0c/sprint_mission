"""
영화(Movie) 관련 스키마 클래스
"""

from pydantic import BaseModel, Field
from typing import Optional, List, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.schemas.review import ReviewResponse

import logging
from helper_dev_utils import get_auto_logger

logger = get_auto_logger(log_level=logging.DEBUG)


class MovieCreate(BaseModel):
    """
    영화 등록 요청 스키마

    Attributes:
        tmdb_id: TMDB 영화 ID
        title: 영화 제목
        release_date: 개봉일
        director: 감독
        genre: 장르
        poster_url: 포스터 이미지 URL
        tmdb_rating: TMDB 평점
    """

    tmdb_id: int = Field(..., description="TMDB 영화 ID")
    title: str = Field(..., max_length=255, description="영화 제목")
    release_date: Optional[str] = Field(None, max_length=50, description="개봉일")
    director: Optional[str] = Field(None, max_length=100, description="감독")
    genre: Optional[str] = Field(None, max_length=100, description="장르")
    poster_url: Optional[str] = Field(None, description="포스터 이미지 URL")
    tmdb_rating: Optional[float] = Field(None, description="TMDB 평점")


class MovieResponse(BaseModel):
    """
    영화 조회 응답 스키마

    Attributes:
        id: 영화 ID
        tmdb_id: TMDB 영화 ID
        title: 영화 제목
        release_date: 개봉일 (YYYY-MM-DD)
        director: 감독 이름
        genre: 장르
        poster_local_path: 포스터 로컬 경로
        tmdb_rating: TMDB 평점 (0-10)
    """

    id: int = Field(..., description="영화 ID")
    tmdb_id: int = Field(..., description="TMDB 영화 ID")
    title: str = Field(..., description="영화 제목")
    release_date: Optional[str] = Field(None, description="개봉일 (YYYY-MM-DD)")
    director: Optional[str] = Field(None, description="감독 이름")
    genre: Optional[str] = Field(None, description="장르")
    poster_local_path: Optional[str] = Field(None, description="포스터 로컬 경로")
    tmdb_rating: Optional[float] = Field(None, description="TMDB 평점 (0-10)")

    class Config:
        from_attributes = True


class MovieWithReviews(MovieResponse):
    """
    리뷰 포함 영화 조회 응답 스키마

    Attributes:
        reviews: 영화에 달린 리뷰 목록
    """

    reviews: List[Any] = []  # 순환 참조 방지를 위해 Any 사용


class MovieWithReviewsAndRating(MovieWithReviews):
    """
    리뷰 및 AI 평점 포함 영화 응답 스키마

    Attributes:
        total_reviews: 전체 리뷰 수
        positive_reviews: 긍정 리뷰 수
        negative_reviews: 부정 리뷰 수
        positive_ratio: 긍정 비율
        ai_rating: AI 평점 (5점 만점)
    """

    total_reviews: int = Field(0, description="전체 리뷰 수")
    positive_reviews: int = Field(0, description="긍정 리뷰 수")
    negative_reviews: int = Field(0, description="부정 리뷰 수")
    positive_ratio: float = Field(0.0, description="긍정 비율 (0.0 ~ 1.0)")
    ai_rating: float = Field(0.0, description="AI 평점 (긍정 비율 기반 5점 만점)")


class MoviePaginationResponse(BaseModel):
    """
    영화 목록 페이지네이션 응답 스키마

    Attributes:
        total: 전체 영화 수
        page: 현재 페이지 번호
        page_size: 페이지당 항목 수
        total_pages: 전체 페이지 수
        movies: 영화 목록 (리뷰 및 AI 평점 포함)
    """

    total: int = Field(..., description="전체 영화 수")
    page: int = Field(..., description="현재 페이지 번호")
    page_size: int = Field(..., description="페이지당 항목 수")
    total_pages: int = Field(..., description="전체 페이지 수")
    movies: List[MovieWithReviewsAndRating] = Field(
        ..., description="영화 목록 (리뷰 및 AI 평점 포함)"
    )


class MovieSearchFilters(BaseModel):
    """
    영화 검색 필터 스키마

    Attributes:
        title: 영화 제목 (부분 검색, 대소문자 무시)
        director: 감독 이름 (부분 검색, 대소문자 무시)
        genre: 장르 (부분 검색, 대소문자 무시)
        release_date_from: 개봉일 시작 (YYYY-MM-DD)
        release_date_to: 개봉일 종료 (YYYY-MM-DD)
        tmdb_rating_min: 최소 TMDB 평점
        tmdb_rating_max: 최대 TMDB 평점
        ai_rating_min: 최소 AI 평점
        ai_rating_max: 최대 AI 평점
        sort_by: 정렬 필드 (release_date, tmdb_rating, title, ai_rating)
        sort_order: 정렬 방향 (asc, desc)
        page: 페이지 번호
        page_size: 페이지당 항목 수
    """

    title: Optional[str] = Field(
        None, description="영화 제목 (부분 검색, 대소문자 무시)"
    )
    director: Optional[str] = Field(
        None, description="감독 이름 (부분 검색, 대소문자 무시)"
    )
    genre: Optional[str] = Field(None, description="장르 (부분 검색, 대소문자 무시)")
    release_date_from: Optional[str] = Field(
        None, description="개봉일 시작 (YYYY-MM-DD)"
    )
    release_date_to: Optional[str] = Field(None, description="개봉일 종료 (YYYY-MM-DD)")
    tmdb_rating_min: Optional[float] = Field(
        None, ge=0, le=10, description="최소 TMDB 평점 (0-10)"
    )
    tmdb_rating_max: Optional[float] = Field(
        None, ge=0, le=10, description="최대 TMDB 평점 (0-10)"
    )
    ai_rating_min: Optional[float] = Field(
        None, ge=0, le=5, description="최소 AI 평점 (0-5)"
    )
    ai_rating_max: Optional[float] = Field(
        None, ge=0, le=5, description="최대 AI 평점 (0-5)"
    )
    sort_by: str = Field(
        "release_date",
        description="정렬 필드 (release_date, tmdb_rating, title, ai_rating)",
    )
    sort_order: str = Field("desc", description="정렬 방향 (asc, desc)")
    page: int = Field(1, ge=1, description="페이지 번호")
    page_size: int = Field(10, ge=1, le=100, description="페이지당 항목 수")
