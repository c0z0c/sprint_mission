"""
영화(Movie) 서비스 클래스
"""

from sqlmodel import Session, select, func
from sqlalchemy.orm import selectinload
from typing import List, Optional
import os
import requests
import time
from pathlib import Path

from app.models.MovieModel import MovieModel
from app.models.ReviewModel import ReviewModel
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
            # TMDB 추가 필드
            overview=movie_data.overview,
            popularity=movie_data.popularity,
            vote_count=movie_data.vote_count,
            original_title=movie_data.original_title,
            original_language=movie_data.original_language,
            adult=movie_data.adult,
            backdrop_path=movie_data.backdrop_path,
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

        # 페이지네이션된 영화 목록 조회 (리뷰 포함 - Eager Loading)
        offset = (page - 1) * page_size
        statement = (
            select(MovieModel)
            .options(selectinload(MovieModel.reviews))  # N+1 쿼리 방지
            .offset(offset)
            .limit(page_size)
        )
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
        영화 삭제 (포스터 파일도 함께 삭제)

        Args:
            movie_id: 영화 ID

        Returns:
            bool: 삭제 성공 여부
        """
        movie = self.session.get(MovieModel, movie_id)
        if not movie:
            return False

        # 포스터 파일 삭제
        if movie.poster_local_path:
            poster_file = Path(movie.poster_local_path)
            if poster_file.exists():
                try:
                    poster_file.unlink()
                    logger.info(
                        f"[Service] Deleted poster file: {movie.poster_local_path}"
                    )
                except Exception as e:
                    logger.warning(
                        f"[Service] Failed to delete poster file: {movie.poster_local_path} - {str(e)}"
                    )

        self.session.delete(movie)
        self.session.commit()
        return True

    def get_max_tmdb_id(self) -> int:
        """
        최대 TMDB ID 조회 (효율적인 쿼리)

        Returns:
            int: 최대 TMDB ID (영화가 없으면 0)
        """
        statement = select(func.max(MovieModel.tmdb_id))
        result = self.session.exec(statement).one()
        return result if result is not None else 0

    def update_movie_ai_rating(self, tmdb_id: int) -> None:
        """
        영화의 AI 평점 업데이트 (리뷰 기반 감성 분석)

        Args:
            tmdb_id: TMDB 영화 ID
        """
        logger.debug(f"[Service] Updating AI rating for TMDB ID: {tmdb_id}")

        # 영화 조회
        movie = self.get_movie_by_tmdb_id(tmdb_id)
        if not movie:
            logger.warning(f"[Service] Movie not found for TMDB ID: {tmdb_id}")
            return

        # 해당 영화의 모든 리뷰 조회
        statement = select(ReviewModel).where(ReviewModel.tmdb_id == tmdb_id)
        reviews = self.session.exec(statement).all()

        if not reviews:
            # 리뷰가 없으면 ai_rating을 None으로 설정
            movie.ai_rating = None
            logger.debug(
                f"[Service] No reviews found for TMDB ID: {tmdb_id}, setting ai_rating to None"
            )
        else:
            # 긍정 리뷰 개수 계산
            positive_count = sum(1 for r in reviews if r.is_positive == 1)
            total_count = len(reviews)

            # AI 평점 계산 (긍정 비율 * 5점)
            positive_ratio = positive_count / total_count if total_count > 0 else 0.0
            ai_rating = round(positive_ratio * 5.0, 2)

            movie.ai_rating = ai_rating
            logger.debug(
                f"[Service] Updated AI rating for TMDB ID: {tmdb_id} - "
                f"AI Rating: {ai_rating} ({positive_count}/{total_count} positive)"
            )

        self.session.add(movie)
        self.session.commit()
        self.session.refresh(movie)

    def search_movies(
        self, filters: "MovieSearchFilters"
    ) -> tuple[List[MovieModel], int]:
        """
        영화 검색 (복합 필터링, 정렬, 페이지네이션)

        Args:
            filters: 영화 검색 필터

        Returns:
            tuple[List[MovieModel], int]: (영화 목록, 전체 검색 결과 수)
        """
        from app.schemas.movie import MovieSearchFilters

        logger.debug(f"[Service] Searching movies with filters: {filters}")

        # 기본 쿼리
        statement = select(MovieModel)

        # 필터 적용 (AND 조합)
        if filters.title:
            # 대소문자 무시 검색
            statement = statement.where(MovieModel.title.ilike(f"%{filters.title}%"))

        if filters.director:
            # 대소문자 무시 검색
            statement = statement.where(
                MovieModel.director.ilike(f"%{filters.director}%")
            )

        if filters.genre:
            # 대소문자 무시 부분 매칭
            statement = statement.where(MovieModel.genre.ilike(f"%{filters.genre}%"))

        if filters.release_date_from:
            # 문자열 비교 (YYYY-MM-DD 형식이므로 사전순 정렬 가능)
            statement = statement.where(
                MovieModel.release_date >= filters.release_date_from
            )

        if filters.release_date_to:
            # 문자열 비교
            statement = statement.where(
                MovieModel.release_date <= filters.release_date_to
            )

        if filters.tmdb_rating_min is not None:
            statement = statement.where(
                MovieModel.tmdb_rating >= filters.tmdb_rating_min
            )

        if filters.tmdb_rating_max is not None:
            statement = statement.where(
                MovieModel.tmdb_rating <= filters.tmdb_rating_max
            )

        if filters.ai_rating_min is not None:
            statement = statement.where(MovieModel.ai_rating >= filters.ai_rating_min)

        if filters.ai_rating_max is not None:
            statement = statement.where(MovieModel.ai_rating <= filters.ai_rating_max)

        # 전체 검색 결과 수 조회 (필터 적용 후)
        count_statement = select(func.count()).select_from(statement.subquery())
        total = self.session.exec(count_statement).one()

        # 정렬 적용
        sort_field = getattr(MovieModel, filters.sort_by, MovieModel.release_date)
        if filters.sort_order == "desc":
            statement = statement.order_by(sort_field.desc())
        else:
            statement = statement.order_by(sort_field.asc())

        # 페이지네이션 적용
        offset = (filters.page - 1) * filters.page_size
        statement = statement.offset(offset).limit(filters.page_size)

        # 리뷰 Eager Loading 추가 (N+1 쿼리 방지)
        statement = statement.options(selectinload(MovieModel.reviews))

        # 실행
        results = self.session.exec(statement)
        movies = results.all()

        logger.debug(
            f"[Service] Found {total} movies, returning page {filters.page} with {len(movies)} items"
        )

        return movies, total

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

    def update_movie(
        self, movie_id: int, movie_data: "MovieUpdate", background_tasks=None
    ) -> Optional[MovieModel]:
        """
        영화 정보 업데이트 (전체 업데이트 - PUT)

        Args:
            movie_id: 영화 ID
            movie_data: 영화 업데이트 데이터 (MovieUpdate 또는 MoviePatch)
            background_tasks: FastAPI BackgroundTasks (선택사항)

        Returns:
            Optional[MovieModel]: 업데이트된 영화 모델 또는 None
        """
        from app.schemas.movie import MovieUpdate, MoviePatch

        logger.debug(f"[Service] update_movie started for movie ID: {movie_id}")

        # 영화 조회
        movie = self.session.get(MovieModel, movie_id)
        if not movie:
            logger.warning(f"[Service] Movie not found for ID: {movie_id}")
            return None

        # 기존 poster_url 추출 (변경 감지용)
        old_poster_path = movie.poster_local_path

        # 필드 업데이트 (exclude_unset=True로 PATCH 지원)
        update_dict = movie_data.model_dump(exclude_unset=True)

        # poster_url 처리 (변경 시 재다운로드)
        poster_url = update_dict.pop("poster_url", None)
        if poster_url is not None:
            # 기존 포스터 파일 삭제
            if old_poster_path:
                old_file_path = Path(old_poster_path)
                if old_file_path.exists():
                    try:
                        os.remove(old_file_path)
                        logger.info(
                            f"[Service] Deleted old poster file: {old_poster_path}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"[Service] Failed to delete old poster file: {old_poster_path} - {str(e)}"
                        )
                else:
                    logger.warning(
                        f"[Service] Old poster file not found: {old_poster_path}"
                    )

            # 새 포스터 다운로드 준비 (백그라운드)
            movie.poster_local_path = None
            if poster_url and background_tasks:
                logger.debug(
                    f"[Service] Scheduling new poster download for movie ID: {movie_id}"
                )
                background_tasks.add_task(
                    self._download_and_update_poster,
                    movie_id,
                    poster_url,
                    movie.tmdb_id,
                )

        # 나머지 필드 업데이트
        for key, value in update_dict.items():
            setattr(movie, key, value)

        self.session.add(movie)
        self.session.commit()
        self.session.refresh(movie)

        logger.debug(f"[Service] Movie updated successfully: ID={movie_id}")
        return movie

    def bulk_upsert_movies(
        self, movies_data: list[dict], background_tasks=None
    ) -> tuple[int, int, int]:
        """
        영화 데이터 대량 UPSERT (Insert or Update)

        Args:
            movies_data: 영화 데이터 딕셔너리 리스트
            background_tasks: FastAPI BackgroundTasks (포스터 다운로드용)

        Returns:
            tuple[int, int, int]: (신규 등록 수, 업데이트 수, 실패 수)
        """
        inserted_count = 0
        updated_count = 0
        failed_count = 0

        logger.info(f"[Service] Starting bulk upsert for {len(movies_data)} movies...")

        for movie_data in movies_data:
            try:
                tmdb_id = movie_data.get("tmdb_id")
                if not tmdb_id:
                    logger.warning("[Service] Movie data missing tmdb_id, skipping")
                    failed_count += 1
                    continue

                # 기존 영화 조회
                existing_movie = self.get_movie_by_tmdb_id(tmdb_id)

                if existing_movie:
                    # 업데이트
                    for key, value in movie_data.items():
                        if key == "poster_url":
                            # 포스터는 별도 처리 (백그라운드)
                            if value and background_tasks:
                                # 기존 포스터와 다르면 재다운로드
                                background_tasks.add_task(
                                    self._download_and_update_poster,
                                    existing_movie.id,
                                    value,
                                    tmdb_id,
                                )
                        elif hasattr(existing_movie, key) and key != "tmdb_id":
                            setattr(existing_movie, key, value)

                    self.session.add(existing_movie)
                    updated_count += 1
                    logger.debug(f"[Service] Updated movie: {tmdb_id}")

                else:
                    # 신규 등록
                    poster_url = movie_data.pop("poster_url", None)

                    new_movie = MovieModel(**movie_data)
                    self.session.add(new_movie)
                    self.session.flush()  # ID 생성

                    # 포스터 다운로드 (백그라운드)
                    if poster_url and background_tasks:
                        background_tasks.add_task(
                            self._download_and_update_poster,
                            new_movie.id,
                            poster_url,
                            tmdb_id,
                        )

                    inserted_count += 1
                    logger.debug(f"[Service] Inserted movie: {tmdb_id}")

                # 50건마다 커밋 (배치 처리)
                if (inserted_count + updated_count) % 50 == 0:
                    self.session.commit()
                    logger.debug(
                        f"[Service] Batch commit: {inserted_count} inserted, {updated_count} updated"
                    )

            except Exception as e:
                logger.error(
                    f"[Service] Failed to upsert movie {movie_data.get('tmdb_id')}: {str(e)}"
                )
                failed_count += 1
                continue

        # 최종 커밋
        try:
            self.session.commit()
            logger.info(
                f"[Service] Bulk upsert completed: "
                f"{inserted_count} inserted, {updated_count} updated, {failed_count} failed"
            )
        except Exception as e:
            logger.error(f"[Service] Final commit failed: {str(e)}")
            self.session.rollback()

        return inserted_count, updated_count, failed_count
