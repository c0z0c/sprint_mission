"""
영화(Movie) 서비스 클래스
"""

from sqlmodel import Session, select
from typing import List, Optional
import os
import requests
from pathlib import Path

from app.models.MovieModel import MovieModel
from app.schemas import MovieCreate

import logging
from helper_dev_utils import get_auto_logger

logger = get_auto_logger(log_level=logging.DEBUG)


class MovieService:
    """
    영화 관련 비즈니스 로직을 처리하는 서비스 클래스
    """

    def __init__(self, session: Session):
        """
        MovieService 초기화

        Args:
            session: 데이터베이스 세션
        """
        self.session = session

    def create_movie(self, movie_data: MovieCreate) -> MovieModel:
        """
        영화 등록 (포스터 이미지 저장 포함)

        Args:
            movie_data: 영화 등록 데이터

        Returns:
            MovieModel: 등록된 영화 모델
        """
        # 포스터 이미지 다운로드 및 저장
        poster_local_path = None
        if movie_data.poster_url:
            poster_local_path = self._download_poster(
                movie_data.poster_url, movie_data.tmdb_id
            )

        # 영화 모델 생성
        movie = MovieModel(
            tmdb_id=movie_data.tmdb_id,
            title=movie_data.title,
            release_date=movie_data.release_date,
            director=movie_data.director,
            genre=movie_data.genre,
            poster_local_path=poster_local_path,
            tmdb_rating=movie_data.tmdb_rating,
        )

        self.session.add(movie)
        self.session.commit()
        self.session.refresh(movie)

        return movie

    def get_all_movies(self) -> List[MovieModel]:
        """
        전체 영화 목록 조회

        Returns:
            List[MovieModel]: 영화 모델 리스트
        """
        statement = select(MovieModel)
        results = self.session.exec(statement)
        return results.all()

    def get_movie_by_id(self, movie_id: int) -> Optional[MovieModel]:
        """
        특정 영화 조회

        Args:
            movie_id: 영화 ID

        Returns:
            Optional[MovieModel]: 영화 모델 또는 None
        """
        return self.session.get(MovieModel, movie_id)

    def get_movie_by_tmdb_id(self, tmdb_id: int) -> Optional[MovieModel]:
        """
        TMDB ID로 영화 조회

        Args:
            tmdb_id: TMDB 영화 ID

        Returns:
            Optional[MovieModel]: 영화 모델 또는 None
        """
        statement = select(MovieModel).where(MovieModel.tmdb_id == tmdb_id)
        result = self.session.exec(statement).first()
        return result

    def delete_movie(self, movie_id: int) -> bool:
        """
        영화 삭제

        Args:
            movie_id: 영화 ID

        Returns:
            bool: 삭제 성공 여부
        """
        movie = self.session.get(MovieModel, movie_id)
        if not movie:
            return False

        self.session.delete(movie)
        self.session.commit()
        return True

    def _download_poster(self, poster_url: str, tmdb_id: int) -> Optional[str]:
        """
        포스터 이미지 다운로드 및 저장

        Args:
            poster_url: 포스터 이미지 URL
            tmdb_id: TMDB 영화 ID

        Returns:
            Optional[str]: 저장된 로컬 경로 또는 None
        """
        # 저장 디렉토리 생성
        poster_dir = Path("static/posters")
        poster_dir.mkdir(parents=True, exist_ok=True)

        # 파일 경로 설정
        file_extension = poster_url.split(".")[-1].split("?")[0]
        if file_extension not in ["jpg", "jpeg", "png", "webp"]:
            file_extension = "jpg"

        file_path = poster_dir / f"{tmdb_id}.{file_extension}"

        # 이미지 다운로드
        response = requests.get(poster_url, timeout=10)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
            return str(file_path)

        return None
