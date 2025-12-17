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
