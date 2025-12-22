"""
Schemas 모듈 초기화
"""

from app.schemas.movie import (
    MovieCreate,
    MovieResponse,
    MovieWithReviews,
    MovieWithReviewsAndRating,
    MoviePaginationResponse,
    MovieUpdate,
    MoviePatch,
)
from app.schemas.review import (
    ReviewCreate,
    ReviewResponse,
    ReviewWithMovie,
    ReviewPaginationResponse,
    ReviewUpdate,
    ReviewPatch,
)
from app.schemas.rating import MovieRating

__all__ = [
    "MovieCreate",
    "MovieResponse",
    "MovieWithReviews",
    "MovieWithReviewsAndRating",
    "MoviePaginationResponse",
    "MovieUpdate",
    "MoviePatch",
    "ReviewCreate",
    "ReviewResponse",
    "ReviewWithMovie",
    "ReviewPaginationResponse",
    "ReviewUpdate",
    "ReviewPatch",
    "MovieRating",
]
