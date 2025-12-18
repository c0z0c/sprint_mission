"""
Schemas 모듈 초기화
"""

from app.schemas.movie import (
    MovieCreate,
    MovieResponse,
    MovieWithReviews,
    MovieWithReviewsAndRating,
    MoviePaginationResponse,
)
from app.schemas.review import (
    ReviewCreate,
    ReviewResponse,
    ReviewWithMovie,
    ReviewPaginationResponse,
)
from app.schemas.rating import MovieRating

__all__ = [
    "MovieCreate",
    "MovieResponse",
    "MovieWithReviews",
    "MovieWithReviewsAndRating",
    "MoviePaginationResponse",
    "ReviewCreate",
    "ReviewResponse",
    "ReviewWithMovie",
    "ReviewPaginationResponse",
    "MovieRating",
]
