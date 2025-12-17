"""
리뷰(Review) 관련 스키마 클래스
"""

from pydantic import BaseModel, Field
from typing import Optional, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.schemas.movie import MovieResponse

import logging
from helper_dev_utils import get_auto_logger

logger = get_auto_logger(log_level=logging.DEBUG)


class ReviewCreate(BaseModel):
    """리뷰 등록 요청 스키마"""

    movie_id: int = Field(..., description="영화 ID")
    author: str = Field(..., max_length=100, description="작성자 이름")
    content: str = Field(..., max_length=2000, description="리뷰 내용")


class ReviewResponse(BaseModel):
    """리뷰 조회 응답 스키마"""

    id: int
    movie_id: int
    author: str
    content: str
    is_positive: Optional[int] = None

    class Config:
        from_attributes = True


class ReviewWithMovie(ReviewResponse):
    """영화 정보 포함 리뷰 조회 응답 스키마"""

    movie: Any  # 순환 참조 방지를 위해 Any 사용
