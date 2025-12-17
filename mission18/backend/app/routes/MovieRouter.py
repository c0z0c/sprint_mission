"""
영화(Movie) API 라우터
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session
from typing import List

from app.database import get_db
from app.services.MovieService import MovieService
from app.schemas import MovieCreate, MovieResponse, MovieWithReviews

import logging
from helper_dev_utils import get_auto_logger

logger = get_auto_logger(log_level=logging.DEBUG)


class MovieRouter:
    """
    영화 관련 API 엔드포인트를 관리하는 라우터 클래스
    """

    def __init__(self):
        """
        MovieRouter 초기화 및 라우트 설정
        """
        self.router = APIRouter(prefix="/movies", tags=["movies"])
        self._setup_routes()

    def _setup_routes(self):
        """
        라우트 설정
        """
        self.router.add_api_route(
            "/",
            self.create_movie,
            methods=["POST"],
            response_model=MovieResponse,
            status_code=status.HTTP_201_CREATED,
            summary="영화 등록",
            description="새로운 영화를 등록합니다. 포스터 URL이 제공되면 이미지를 다운로드하여 저장합니다.",
        )
        self.router.add_api_route(
            "/",
            self.get_all_movies,
            methods=["GET"],
            response_model=List[MovieResponse],
            summary="전체 영화 목록 조회",
            description="등록된 모든 영화 목록을 조회합니다.",
        )
        self.router.add_api_route(
            "/{movie_id}",
            self.get_movie,
            methods=["GET"],
            response_model=MovieWithReviews,
            summary="특정 영화 조회",
            description="영화 ID로 특정 영화를 조회합니다. 리뷰 정보도 함께 반환됩니다.",
        )
        self.router.add_api_route(
            "/{movie_id}",
            self.delete_movie,
            methods=["DELETE"],
            status_code=status.HTTP_204_NO_CONTENT,
            summary="영화 삭제",
            description="영화 ID로 특정 영화를 삭제합니다.",
        )

    def create_movie(
        self, movie_data: MovieCreate, db: Session = Depends(get_db)
    ) -> MovieResponse:
        """
        영화 등록

        Args:
            movie_data: 영화 등록 데이터
            db: 데이터베이스 세션

        Returns:
            MovieResponse: 등록된 영화 정보

        Raises:
            HTTPException: TMDB ID가 이미 존재하는 경우
        """
        service = MovieService(db)

        # TMDB ID 중복 체크
        existing_movie = service.get_movie_by_tmdb_id(movie_data.tmdb_id)
        if existing_movie:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"TMDB ID {movie_data.tmdb_id}는 이미 등록된 영화입니다.",
            )

        movie = service.create_movie(movie_data)
        return movie

    def get_all_movies(self, db: Session = Depends(get_db)) -> List[MovieResponse]:
        """
        전체 영화 목록 조회

        Args:
            db: 데이터베이스 세션

        Returns:
            List[MovieResponse]: 영화 목록
        """
        service = MovieService(db)
        movies = service.get_all_movies()
        return movies

    def get_movie(
        self, movie_id: int, db: Session = Depends(get_db)
    ) -> MovieWithReviews:
        """
        특정 영화 조회

        Args:
            movie_id: 영화 ID
            db: 데이터베이스 세션

        Returns:
            MovieWithReviews: 영화 정보 (리뷰 포함)

        Raises:
            HTTPException: 영화를 찾을 수 없는 경우
        """
        service = MovieService(db)
        movie = service.get_movie_by_id(movie_id)

        if not movie:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"영화 ID {movie_id}를 찾을 수 없습니다.",
            )

        return movie

    def delete_movie(self, movie_id: int, db: Session = Depends(get_db)) -> None:
        """
        영화 삭제

        Args:
            movie_id: 영화 ID
            db: 데이터베이스 세션

        Raises:
            HTTPException: 영화를 찾을 수 없는 경우
        """
        service = MovieService(db)
        success = service.delete_movie(movie_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"영화 ID {movie_id}를 찾을 수 없습니다.",
            )


# 라우터 인스턴스 생성
movie_router = MovieRouter()
router = movie_router.router
