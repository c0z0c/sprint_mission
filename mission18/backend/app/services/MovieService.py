"""
영화(Movie) 서비스 클래스
"""

from sqlmodel import Session, select
from typing import List, Optional
import os
import requests
import time
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

    def create_movie(
        self, movie_data: MovieCreate, background_tasks=None
    ) -> MovieModel:
        """
        영화 등록 (포스터 다운로드는 백그라운드 처리)

        Args:
            movie_data: 영화 등록 데이터
            background_tasks: FastAPI BackgroundTasks (선택사항)

        Returns:
            MovieModel: 등록된 영화 모델
        """
        logger.debug(
            f"[Service] create_movie started for TMDB ID: {movie_data.tmdb_id}"
        )
        service_start = time.time()

        # 영화 모델 생성 (포스터는 아직 None)
        db_start = time.time()
        movie = MovieModel(
            tmdb_id=movie_data.tmdb_id,
            title=movie_data.title,
            release_date=movie_data.release_date,
            director=movie_data.director,
            genre=movie_data.genre,
            poster_local_path=None,  # 나중에 백그라운드에서 업데이트
            tmdb_rating=movie_data.tmdb_rating,
        )

        self.session.add(movie)
        self.session.commit()
        self.session.refresh(movie)

        db_elapsed = time.time() - db_start
        logger.debug(f"[Service] DB save completed in {db_elapsed:.2f}s")

        # 포스터 다운로드를 백그라운드 작업으로 등록
        if movie_data.poster_url and background_tasks:
            logger.debug(
                f"[Service] Scheduling poster download in background for movie ID: {movie.id}"
            )
            background_tasks.add_task(
                self._download_and_update_poster,
                movie.id,
                movie_data.poster_url,
                movie_data.tmdb_id,
            )

        service_elapsed = time.time() - service_start
        logger.debug(
            f"[Service] Total create_movie time: {service_elapsed:.2f}s (without poster download)"
        )

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

    def get_movies_paginated(
        self, page: int = 1, page_size: int = 10
    ) -> tuple[List[MovieModel], int]:
        """
        페이지네이션된 영화 목록 조회

        Args:
            page: 페이지 번호 (1부터 시작)
            page_size: 페이지당 항목 수

        Returns:
            tuple[List[MovieModel], int]: (영화 목록, 전체 영화 수)
        """
        # 전체 영화 수 조회
        from sqlmodel import func

        count_statement = select(func.count(MovieModel.id))
        total = self.session.exec(count_statement).one()

        # 페이지네이션된 영화 목록 조회
        offset = (page - 1) * page_size
        statement = select(MovieModel).offset(offset).limit(page_size)
        results = self.session.exec(statement)

        return results.all(), total

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

    def get_max_tmdb_id(self) -> int:
        """
        최대 TMDB ID 조회 (효율적인 쿼리)

        Returns:
            int: 최대 TMDB ID (영화가 없으면 0)
        """
        from sqlmodel import func

        statement = select(func.max(MovieModel.tmdb_id))
        result = self.session.exec(statement).one()
        return result if result is not None else 0

    def _download_and_update_poster(
        self, movie_id: int, poster_url: str, tmdb_id: int
    ) -> None:
        """
        백그라운드에서 포스터 다운로드 후 DB 업데이트

        Args:
            movie_id: 영화 ID
            poster_url: 포스터 이미지 URL
            tmdb_id: TMDB 영화 ID
        """
        logger.debug(
            f"[Background] Starting poster download for movie ID: {movie_id}, TMDB ID: {tmdb_id}"
        )
        download_start = time.time()

        try:
            # 포스터 다운로드
            poster_local_path = self._download_poster(poster_url, tmdb_id)

            if poster_local_path:
                # DB 업데이트
                movie = self.session.get(MovieModel, movie_id)
                if movie:
                    movie.poster_local_path = poster_local_path
                    self.session.add(movie)
                    self.session.commit()

                    elapsed = time.time() - download_start
                    logger.info(
                        f"[Background] Poster successfully downloaded and updated for movie ID: {movie_id} in {elapsed:.2f}s"
                    )
                else:
                    logger.error(f"[Background] Movie not found for ID: {movie_id}")
            else:
                logger.warning(
                    f"[Background] Poster download failed for movie ID: {movie_id} - keeping poster_local_path as None"
                )
        except Exception as e:
            logger.error(
                f"[Background] Error downloading poster for movie ID: {movie_id} - {str(e)}"
            )
            # 에러가 발생해도 영화 등록 자체는 유지 (poster_local_path는 None으로 유지)

    def _download_poster(self, poster_url: str, tmdb_id: int) -> Optional[str]:
        """
        포스터 이미지 다운로드 및 저장

        Args:
            poster_url: 포스터 이미지 URL
            tmdb_id: TMDB 영화 ID

        Returns:
            Optional[str]: 저장된 로컬 경로 또는 None
        """
        logger.debug(f"[Service] Starting poster download for TMDB ID: {tmdb_id}")
        download_start = time.time()

        # 저장 디렉토리 생성
        poster_dir = Path("data/posters")
        poster_dir.mkdir(parents=True, exist_ok=True)

        # 파일 경로 설정
        file_extension = poster_url.split(".")[-1].split("?")[0]
        if file_extension not in ["jpg", "jpeg", "png", "webp"]:
            file_extension = "jpg"

        file_path = poster_dir / f"{tmdb_id}.{file_extension}"

        # 이미지 다운로드
        try:
            response = requests.get(poster_url, timeout=10)
            request_elapsed = time.time() - download_start
            logger.debug(
                f"[Service] Poster HTTP request completed in {request_elapsed:.2f}s (status: {response.status_code})"
            )

            if response.status_code == 200:
                write_start = time.time()
                with open(file_path, "wb") as f:
                    f.write(response.content)
                write_elapsed = time.time() - write_start
                total_elapsed = time.time() - download_start

                logger.debug(
                    f"[Service] Poster file write completed in {write_elapsed:.2f}s"
                )
                logger.debug(
                    f"[Service] Total poster download time: {total_elapsed:.2f}s"
                )

                # data 마운트 경로를 제외한 상대 경로 반환 (프론트엔드에서 /data/ 추가)
                return f"data/posters/{tmdb_id}.{file_extension}"
        except Exception as e:
            logger.error(f"[Service] Poster download failed: {str(e)}")

        return None
