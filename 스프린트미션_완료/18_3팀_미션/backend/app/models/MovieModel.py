"""
영화(Movie) 모델 클래스
"""

from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.ReviewModel import ReviewModel

import logging
from helper_dev_utils import get_auto_logger

logger = get_auto_logger(log_level=logging.DEBUG)


class MovieModel(SQLModel, table=True):
    """
    영화 정보 테이블 모델
    """

    __tablename__ = "movies"

    id: Optional[int] = Field(default=None, primary_key=True)
    tmdb_id: int = Field(
        unique=True, nullable=False, index=True, description="TMDB 영화 ID"
    )
    title: str = Field(max_length=255, nullable=False, description="영화 제목")
    release_date: Optional[str] = Field(
        default=None, max_length=50, description="개봉일"
    )
    director: Optional[str] = Field(default=None, max_length=100, description="감독")
    genre: Optional[str] = Field(default=None, max_length=100, description="장르")
    poster_local_path: Optional[str] = Field(
        default=None, max_length=500, description="포스터 이미지 로컬 경로"
    )
    tmdb_rating: Optional[float] = Field(default=None, description="TMDB 평점")
    ai_rating: Optional[float] = Field(
        default=None, index=True, description="AI 평점 (리뷰 기반 감성 분석, 0-10점)"
    )

    # TMDB 추가 필드
    overview: Optional[str] = Field(default=None, description="영화 줄거리/개요")
    popularity: Optional[float] = Field(
        default=None, index=True, description="TMDB 인기도"
    )
    vote_count: Optional[int] = Field(default=None, description="TMDB 투표 수")
    original_title: Optional[str] = Field(
        default=None, max_length=255, description="원제"
    )
    original_language: Optional[str] = Field(
        default=None, max_length=10, description="원어 (ISO 639-1 코드)"
    )
    adult: Optional[bool] = Field(default=False, description="성인 영화 여부")
    backdrop_path: Optional[str] = Field(
        default=None, max_length=200, description="TMDB 배경 이미지 경로 (URL)"
    )

    # 관계 설정
    reviews: List["ReviewModel"] = Relationship(
        back_populates="movie", cascade_delete=True
    )

    def __repr__(self):
        return (
            f"<MovieModel(id={self.id}, tmdb_id={self.tmdb_id}, title='{self.title}')>"
        )
