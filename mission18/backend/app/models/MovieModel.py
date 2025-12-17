"""
영화(Movie) 모델 클래스
"""

from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.app.models.ReviewModel import ReviewModel

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

    # 관계 설정
    reviews: List["ReviewModel"] = Relationship(
        back_populates="movie", cascade_delete=True
    )

    def __repr__(self):
        return (
            f"<MovieModel(id={self.id}, tmdb_id={self.tmdb_id}, title='{self.title}')>"
        )
