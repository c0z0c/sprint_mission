"""
Routes 모듈 초기화
"""

from app.routes.MovieRouter import router as movie_router
from app.routes.ReviewRouter import router as review_router

__all__ = ["movie_router", "review_router"]
