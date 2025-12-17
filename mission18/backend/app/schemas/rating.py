"""
평점(Rating) 관련 스키마 클래스
"""

from pydantic import BaseModel, Field
import logging
from helper_dev_utils import get_auto_logger

logger = get_auto_logger(log_level=logging.DEBUG)


class MovieRating(BaseModel):
    """영화 평점 응답 스키마"""

    movie_id: int
    title: str
    total_reviews: int = Field(..., description="전체 리뷰 수")
    positive_reviews: int = Field(..., description="긍정 리뷰 수")
    negative_reviews: int = Field(..., description="부정 리뷰 수")
    positive_ratio: float = Field(..., description="긍정 비율 (0.0 ~ 1.0)")
    ai_rating: float = Field(..., description="AI 평점 (긍정 비율 기반 5점 만점)")
