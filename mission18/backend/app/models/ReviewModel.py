"""
리뷰(Review) 모델 클래스
"""

from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import UniqueConstraint
from typing import Optional, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from app.models.MovieModel import MovieModel

import logging
from helper_dev_utils import get_auto_logger

logger = get_auto_logger(log_level=logging.DEBUG)


class ReviewModel(SQLModel, table=True):
    """
    리뷰 정보 테이블 모델
    """

    __tablename__ = "reviews"

    # Unique 제약: (tmdb_id, author, content) 조합
    __table_args__ = (
        UniqueConstraint("tmdb_id", "author", "content", name="uq_tmdb_author_content"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    tmdb_id: int = Field(
        foreign_key="movies.tmdb_id", nullable=False, description="TMDB 영화 ID"
    )
    author: str = Field(max_length=100, nullable=False, description="작성자 이름")
    content: str = Field(max_length=2000, nullable=False, description="리뷰 내용")
    is_positive: Optional[int] = Field(
        default=None, description="감성 분석 결과 (0:부정, 1:긍정)"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, nullable=False, description="생성 시간"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        nullable=False,
        description="수정 시간",
    )

    # 관계 설정
    movie: Optional["MovieModel"] = Relationship(back_populates="reviews")

    def __repr__(self):
        return f"<ReviewModel(id={self.id}, tmdb_id={self.tmdb_id}, author='{self.author}')>"
