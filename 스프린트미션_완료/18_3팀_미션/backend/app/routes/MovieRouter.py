"""
영화(Movie) API 라우터
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from sqlmodel import Session
from typing import List, Optional
import math
import time
import asyncio

from app.database import get_db
from app.services.MovieService import MovieService
from app.services.TMDBService import TMDBService
from app.services.SyncStateManager import get_sync_state_manager
from app.constants.tmdb_genres import convert_genre_ids_to_korean
from app.schemas import (
    MovieCreate,
    MovieResponse,
    MovieWithReviews,
    MovieWithReviewsAndRating,
    MoviePaginationResponse,
    MovieUpdate,
    MoviePatch,
)
from app.schemas.tmdb import (
    TMDBSyncRequest,
    SyncStartResponse,
    SyncStateResponse,
    SyncType,
    SyncConfigResponse,
    SyncStatsResponse,
)
from app.config import get_sync_config

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
            "/max-tmdb-id",
            self.get_max_tmdb_id,
            methods=["GET"],
            response_model=dict,
            summary="최대 TMDB ID 조회",
            description="DB에 저장된 최대 TMDB ID를 효율적으로 조회합니다.",
        )
        self.router.add_api_route(
            "/paginated",
            self.get_movies_paginated,
            methods=["GET"],
            response_model=MoviePaginationResponse,
            summary="영화 목록 페이지네이션 조회",
            description="페이지 번호와 페이지 크기를 기반으로 영화 목록을 조회합니다. 리뷰 정보도 함께 반환됩니다.",
        )
        self.router.add_api_route(
            "/search",
            self.search_movies,
            methods=["GET"],
            response_model=MoviePaginationResponse,
            summary="영화 검색 (복합 필터)",
            description="다양한 조건으로 영화를 검색합니다. 모든 필터는 AND 조합으로 작동합니다.",
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
            self.update_movie_put,
            methods=["PUT"],
            response_model=MovieResponse,
            summary="영화 전체 업데이트",
            description="영화 ID로 특정 영화의 모든 필드를 업데이트합니다. poster_url 변경 시 기존 파일 삭제 및 재다운로드가 수행됩니다. tmdb_id는 수정 불가합니다.",
        )
        self.router.add_api_route(
            "/{movie_id}",
            self.update_movie_patch,
            methods=["PATCH"],
            response_model=MovieResponse,
            summary="영화 부분 업데이트",
            description="영화 ID로 특정 영화의 일부 필드만 선택적으로 업데이트합니다. poster_url 변경 시 기존 파일 삭제 및 재다운로드가 수행됩니다. tmdb_id는 수정 불가합니다.",
        )
        self.router.add_api_route(
            "/{movie_id}",
            self.delete_movie,
            methods=["DELETE"],
            status_code=status.HTTP_204_NO_CONTENT,
            summary="영화 삭제",
            description="영화 ID로 특정 영화를 삭제합니다.",
        )

        # TMDB 동기화 엔드포인트
        self.router.add_api_route(
            "/sync/popular",
            self.sync_popular_movies,
            methods=["POST"],
            response_model=SyncStartResponse,
            status_code=status.HTTP_202_ACCEPTED,
            summary="인기 영화 동기화",
            description="TMDB API로부터 인기 영화 목록을 가져와 DB에 저장합니다. 백그라운드에서 실행됩니다.",
        )
        self.router.add_api_route(
            "/sync/latest",
            self.sync_latest_movies,
            methods=["POST"],
            response_model=SyncStartResponse,
            status_code=status.HTTP_202_ACCEPTED,
            summary="최신 영화 동기화",
            description="TMDB API로부터 최신 영화 목록을 가져와 DB에 저장합니다. 백그라운드에서 실행됩니다.",
        )
        self.router.add_api_route(
            "/sync/period",
            self.sync_period_movies,
            methods=["POST"],
            response_model=SyncStartResponse,
            status_code=status.HTTP_202_ACCEPTED,
            summary="기간별 영화 동기화",
            description="TMDB API로부터 특정 기간의 영화 목록을 가져와 DB에 저장합니다. 백그라운드에서 실행됩니다.",
        )
        self.router.add_api_route(
            "/sync/status/{task_id}",
            self.get_sync_status,
            methods=["GET"],
            response_model=SyncStateResponse,
            summary="동기화 상태 조회",
            description="동기화 작업의 진행 상황을 조회합니다.",
        )
        self.router.add_api_route(
            "/sync/config",
            self.get_sync_config_endpoint,
            methods=["GET"],
            response_model=SyncConfigResponse,
            summary="동기화 설정 조회",
            description="현재 동기화 설정 정보를 조회합니다.",
        )
        self.router.add_api_route(
            "/sync/stats",
            self.get_sync_stats,
            methods=["GET"],
            response_model=SyncStatsResponse,
            summary="동기화 통계 조회",
            description="동기화 작업 통계를 조회합니다.",
        )

    def create_movie(
        self,
        movie_data: MovieCreate,
        background_tasks: BackgroundTasks,
        db: Session = Depends(get_db),
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
        logger.debug(
            f"[Router] Movie registration started: {movie_data.title} (TMDB ID: {movie_data.tmdb_id})"
        )
        start_time = time.time()

        service = MovieService(db)

        # TMDB ID 중복 체크
        existing_movie = service.get_movie_by_tmdb_id(movie_data.tmdb_id)
        if existing_movie:
            logger.warning(
                f"Duplicate TMDB ID detected: {movie_data.tmdb_id} - Movie: {existing_movie.title}"
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"TMDB ID {movie_data.tmdb_id}는 이미 등록된 영화입니다. (등록된 영화: {existing_movie.title})",
            )

        # 포스터 다운로드를 백그라운드 작업으로 등록
        has_poster = movie_data.poster_url is not None
        movie = service.create_movie(movie_data, background_tasks)

        elapsed = time.time() - start_time
        logger.info(
            f"[Router] Movie created successfully in {elapsed:.2f}s: {movie.title} (TMDB ID: {movie.tmdb_id})"
            + (" - Poster download scheduled in background" if has_poster else "")
        )
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

    def get_max_tmdb_id(self, db: Session = Depends(get_db)) -> dict:
        """
        최대 TMDB ID 조회

        Args:
            db: 데이터베이스 세션

        Returns:
            dict: {"max_tmdb_id": int}
        """
        service = MovieService(db)
        max_id = service.get_max_tmdb_id()
        return {"max_tmdb_id": max_id}

    def get_movies_paginated(
        self,
        page: int = Query(1, ge=1, description="페이지 번호 (1부터 시작)"),
        page_size: int = Query(
            10, ge=1, le=100, description="페이지당 항목 수 (최대 100)"
        ),
        db: Session = Depends(get_db),
    ) -> MoviePaginationResponse:
        """
        페이지네이션된 영화 목록 조회 (리뷰 및 AI 평점 포함)

        Args:
            page: 페이지 번호 (1부터 시작)
            page_size: 페이지당 항목 수
            db: 데이터베이스 세션

        Returns:
            MoviePaginationResponse: 페이지네이션 정보와 영화 목록 (AI 평점 포함)
        """
        service = MovieService(db)
        movies, total = service.get_movies_paginated(page, page_size)

        # 각 영화의 리뷰 정보 및 AI 평점 계산
        movies_with_data = []
        for movie in movies:
            # 기본 영화 정보
            movie_dict = MovieWithReviews.model_validate(movie).model_dump()
            movie_dict["reviews"] = movie.reviews

            # AI 평점 계산
            total_reviews = len(movie.reviews)
            positive_reviews = sum(
                1 for review in movie.reviews if review.is_positive == 1
            )
            negative_reviews = sum(
                1 for review in movie.reviews if review.is_positive == 0
            )

            if total_reviews > 0:
                positive_ratio = positive_reviews / total_reviews
                ai_rating = positive_ratio * 10.0
            else:
                positive_ratio = 0.0
                ai_rating = 0.0

            # AI 평점 정보 추가
            movie_dict["total_reviews"] = total_reviews
            movie_dict["positive_reviews"] = positive_reviews
            movie_dict["negative_reviews"] = negative_reviews
            movie_dict["positive_ratio"] = round(positive_ratio, 2)
            movie_dict["ai_rating"] = round(ai_rating, 1)

            movies_with_data.append(MovieWithReviewsAndRating(**movie_dict))

        total_pages = math.ceil(total / page_size) if total > 0 else 0

        return MoviePaginationResponse(
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            movies=movies_with_data,
        )

    def search_movies(
        self,
        title: Optional[str] = Query(
            None, description="영화 제목 (부분 검색, 대소문자 무시)"
        ),
        director: Optional[str] = Query(
            None, description="감독 이름 (부분 검색, 대소문자 무시)"
        ),
        genre: Optional[str] = Query(
            None, description="장르 (부분 검색, 대소문자 무시)"
        ),
        release_date_from: Optional[str] = Query(
            None, description="개봉일 시작 (YYYY-MM-DD)"
        ),
        release_date_to: Optional[str] = Query(
            None, description="개봉일 종료 (YYYY-MM-DD)"
        ),
        tmdb_rating_min: Optional[float] = Query(
            None, ge=0, le=10, description="최소 TMDB 평점 (0-10)"
        ),
        tmdb_rating_max: Optional[float] = Query(
            None, ge=0, le=10, description="최대 TMDB 평점 (0-10)"
        ),
        ai_rating_min: Optional[float] = Query(
            None, ge=0, le=10, description="최소 AI 평점 (0-10)"
        ),
        ai_rating_max: Optional[float] = Query(
            None, ge=0, le=10, description="최대 AI 평점 (0-10)"
        ),
        sort_by: str = Query(
            "release_date",
            description="정렬 필드 (release_date, tmdb_rating, title, ai_rating)",
        ),
        sort_order: str = Query("desc", description="정렬 방향 (asc, desc)"),
        page: int = Query(1, ge=1, description="페이지 번호 (1부터 시작)"),
        page_size: int = Query(
            10, ge=1, le=100, description="페이지당 항목 수 (최대 100)"
        ),
        include_reviews: bool = Query(
            False, description="리뷰 포함 여부 (False: 리뷰 개수만, True: 전체 리뷰)"
        ),
        db: Session = Depends(get_db),
    ) -> MoviePaginationResponse:
        """
        영화 검색 (복합 필터, 정렬, 페이지네이션)

        모든 필터는 AND 조합으로 작동합니다.

        Query Parameters:
            - title: 영화 제목 부분 검색 (대소문자 무시)
            - director: 감독 이름 부분 검색 (대소문자 무시)
            - genre: 장르 부분 검색 (대소문자 무시)
            - release_date_from: 개봉일 시작 (YYYY-MM-DD)
            - release_date_to: 개봉일 종료 (YYYY-MM-DD)
            - tmdb_rating_min: 최소 TMDB 평점 (0-10)
            - tmdb_rating_max: 최대 TMDB 평점 (0-10)
            - ai_rating_min: 최소 AI 평점 (0-10)
            - ai_rating_max: 최대 AI 평점 (0-10)
            - sort_by: 정렬 필드 (release_date, tmdb_rating, title, ai_rating)
            - sort_order: 정렬 방향 (asc, desc)
            - page: 페이지 번호
            - page_size: 페이지당 항목 수

        Returns:
            MoviePaginationResponse: 검색 결과 및 페이지네이션 정보
        """
        from app.schemas.movie import MovieSearchFilters

        # 필터 객체 생성
        filters = MovieSearchFilters(
            title=title,
            director=director,
            genre=genre,
            release_date_from=release_date_from,
            release_date_to=release_date_to,
            tmdb_rating_min=tmdb_rating_min,
            tmdb_rating_max=tmdb_rating_max,
            ai_rating_min=ai_rating_min,
            ai_rating_max=ai_rating_max,
            sort_by=sort_by,
            sort_order=sort_order,
            page=page,
            page_size=page_size,
        )

        service = MovieService(db)
        movies, total = service.search_movies(filters)

        # 각 영화의 리뷰 정보 및 AI 평점 추가
        movies_with_data = []
        for movie in movies:
            movie_dict = MovieWithReviews.model_validate(movie).model_dump()

            # include_reviews 파라미터에 따라 리뷰 포함 여부 결정
            if include_reviews:
                movie_dict["reviews"] = movie.reviews
            else:
                movie_dict["reviews"] = []  # 빈 배열 반환 (리뷰 개수만 사용)

            # AI 평점 계산 (캐시된 값 사용 또는 계산)
            total_reviews = len(movie.reviews)
            positive_reviews = sum(
                1 for review in movie.reviews if review.is_positive == 1
            )
            negative_reviews = sum(
                1 for review in movie.reviews if review.is_positive == 0
            )

            if total_reviews > 0:
                positive_ratio = positive_reviews / total_reviews
                ai_rating = (
                    movie.ai_rating
                    if movie.ai_rating is not None
                    else positive_ratio * 10.0
                )
            else:
                positive_ratio = 0.0
                ai_rating = 0.0

            movie_dict["total_reviews"] = total_reviews
            movie_dict["positive_reviews"] = positive_reviews
            movie_dict["negative_reviews"] = negative_reviews
            movie_dict["positive_ratio"] = round(positive_ratio, 2)
            movie_dict["ai_rating"] = round(ai_rating, 1)

            movies_with_data.append(MovieWithReviewsAndRating(**movie_dict))

        total_pages = math.ceil(total / page_size) if total > 0 else 0

        logger.debug(
            f"[Router] Movie search completed: {total} results, page {page}/{total_pages}"
        )

        return MoviePaginationResponse(
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            movies=movies_with_data,
        )

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

    def update_movie_put(
        self,
        movie_id: int,
        movie_data: MovieUpdate,
        background_tasks: BackgroundTasks,
        db: Session = Depends(get_db),
    ) -> MovieResponse:
        """
        영화 전체 업데이트 (PUT)

        Args:
            movie_id: 영화 ID
            movie_data: 영화 업데이트 데이터 (전체 필드)
            background_tasks: 백그라운드 작업
            db: 데이터베이스 세션

        Returns:
            MovieResponse: 업데이트된 영화 정보

        Raises:
            HTTPException: 영화를 찾을 수 없는 경우
        """
        service = MovieService(db)
        updated_movie = service.update_movie(movie_id, movie_data, background_tasks)

        if not updated_movie:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"영화 ID {movie_id}를 찾을 수 없습니다.",
            )

        return updated_movie

    def update_movie_patch(
        self,
        movie_id: int,
        movie_data: MoviePatch,
        background_tasks: BackgroundTasks,
        db: Session = Depends(get_db),
    ) -> MovieResponse:
        """
        영화 부분 업데이트 (PATCH)

        Args:
            movie_id: 영화 ID
            movie_data: 영화 업데이트 데이터 (선택적 필드)
            background_tasks: 백그라운드 작업
            db: 데이터베이스 세션

        Returns:
            MovieResponse: 업데이트된 영화 정보

        Raises:
            HTTPException: 영화를 찾을 수 없는 경우
        """
        service = MovieService(db)
        updated_movie = service.update_movie(movie_id, movie_data, background_tasks)

        if not updated_movie:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"영화 ID {movie_id}를 찾을 수 없습니다.",
            )

        return updated_movie

    # ==================== TMDB 동기화 엔드포인트 ====================

    def sync_popular_movies(
        self,
        request: TMDBSyncRequest = TMDBSyncRequest(),
        background_tasks: BackgroundTasks = BackgroundTasks(),
    ) -> SyncStartResponse:
        """
        인기 영화 동기화 시작

        Args:
            request: 동기화 요청 (max_pages)
            background_tasks: 백그라운드 작업

        Returns:
            SyncStartResponse: 작업 시작 응답
        """
        state_manager = get_sync_state_manager()

        # 이미 실행 중인 작업이 있는지 확인
        if state_manager.has_running_tasks():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="이미 실행 중인 동기화 작업이 있습니다. 완료 후 다시 시도하세요.",
            )

        task_id = state_manager.create_task(SyncType.POPULAR)

        # 백그라운드에서 동기화 실행
        background_tasks.add_task(self._run_sync_popular, task_id, request.max_pages)

        logger.info(
            f"Popular sync started: task_id={task_id}, max_pages={request.max_pages}"
        )

        return SyncStartResponse(
            task_id=task_id,
            sync_type=SyncType.POPULAR,
            message=f"인기 영화 동기화가 시작되었습니다 (최대 {request.max_pages}페이지).",
            status_url=f"/movies/sync/status/{task_id}",
        )

    def sync_latest_movies(
        self,
        request: TMDBSyncRequest = TMDBSyncRequest(),
        background_tasks: BackgroundTasks = BackgroundTasks(),
    ) -> SyncStartResponse:
        """
        최신 영화 동기화 시작

        Args:
            request: 동기화 요청 (max_pages)
            background_tasks: 백그라운드 작업

        Returns:
            SyncStartResponse: 작업 시작 응답
        """
        state_manager = get_sync_state_manager()

        if state_manager.has_running_tasks():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="이미 실행 중인 동기화 작업이 있습니다. 완료 후 다시 시도하세요.",
            )

        task_id = state_manager.create_task(SyncType.LATEST)

        background_tasks.add_task(self._run_sync_latest, task_id, request.max_pages)

        logger.info(
            f"Latest sync started: task_id={task_id}, max_pages={request.max_pages}"
        )

        return SyncStartResponse(
            task_id=task_id,
            sync_type=SyncType.LATEST,
            message=f"최신 영화 동기화가 시작되었습니다 (최대 {request.max_pages}페이지).",
            status_url=f"/movies/sync/status/{task_id}",
        )

    def sync_period_movies(
        self,
        request: TMDBSyncRequest,
        background_tasks: BackgroundTasks = BackgroundTasks(),
    ) -> SyncStartResponse:
        """
        기간별 영화 동기화 시작

        Args:
            request: 동기화 요청 (start_date, end_date, max_pages)
            background_tasks: 백그라운드 작업

        Returns:
            SyncStartResponse: 작업 시작 응답

        Raises:
            HTTPException: start_date가 없는 경우
        """
        if not request.start_date:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="start_date는 필수입니다. (YYYY-MM-DD 형식)",
            )

        state_manager = get_sync_state_manager()

        if state_manager.has_running_tasks():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="이미 실행 중인 동기화 작업이 있습니다. 완료 후 다시 시도하세요.",
            )

        task_id = state_manager.create_task(SyncType.PERIOD)

        background_tasks.add_task(
            self._run_sync_period,
            task_id,
            request.start_date,
            request.end_date,
            request.max_pages,
        )

        logger.info(
            f"Period sync started: task_id={task_id}, "
            f"period={request.start_date} to {request.end_date}, "
            f"max_pages={request.max_pages}"
        )

        return SyncStartResponse(
            task_id=task_id,
            sync_type=SyncType.PERIOD,
            message=f"기간별 영화 동기화가 시작되었습니다 ({request.start_date} ~ {request.end_date}).",
            status_url=f"/movies/sync/status/{task_id}",
        )

    def get_sync_status(self, task_id: str) -> SyncStateResponse:
        """
        동기화 작업 상태 조회

        Args:
            task_id: 작업 ID

        Returns:
            SyncStateResponse: 작업 상태

        Raises:
            HTTPException: 작업을 찾을 수 없는 경우
        """
        state_manager = get_sync_state_manager()
        state = state_manager.get_state(task_id)

        if not state:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"작업 ID {task_id}를 찾을 수 없습니다.",
            )

        return state.to_response()

    def get_sync_config_endpoint(self) -> SyncConfigResponse:
        """
        동기화 설정 조회

        Returns:
            SyncConfigResponse: 설정 정보
        """
        config = get_sync_config()

        return SyncConfigResponse(
            tmdb_base_url=config.get("tmdb.base_url"),
            language=config.get("tmdb.language"),
            region=config.get("tmdb.region"),
            rate_limiting=config.get("tmdb.rate_limiting"),
            initial_sync=config.get("initial_sync"),
            scheduler=config.get("scheduler"),
            poster=config.get("poster"),
            error_handling=config.get("error_handling"),
        )

    def get_sync_stats(self, db: Session = Depends(get_db)) -> SyncStatsResponse:
        """
        동기화 통계 조회

        Args:
            db: 데이터베이스 세션

        Returns:
            SyncStatsResponse: 통계 정보
        """
        state_manager = get_sync_state_manager()
        stats = state_manager.get_statistics()

        # 전체 영화 수 조회
        service = MovieService(db)
        total_movies = len(service.get_all_movies())

        return SyncStatsResponse(
            total_movies=total_movies,
            total_syncs=stats["total_syncs"],
            successful_syncs=stats["successful_syncs"],
            failed_syncs=stats["failed_syncs"],
            last_sync_at=stats["last_sync_at"],
            active_tasks=stats["active_tasks"],
        )

    # ==================== 백그라운드 동기화 작업 ====================

    def _run_sync_popular(self, task_id: str, max_pages: int) -> None:
        """백그라운드에서 인기 영화 동기화 실행"""
        try:
            asyncio.run(self._sync_popular_impl(task_id, max_pages))
        except Exception as e:
            logger.error(f"Popular sync failed: {str(e)}")
            state_manager = get_sync_state_manager()
            state_manager.mark_failed(task_id, str(e))

    def _run_sync_latest(self, task_id: str, max_pages: int) -> None:
        """백그라운드에서 최신 영화 동기화 실행"""
        try:
            config = get_sync_config()
            days_back = config.get("initial_sync.latest.days_back", 7)
            asyncio.run(self._sync_latest_impl(task_id, days_back, max_pages))
        except Exception as e:
            logger.error(f"Latest sync failed: {str(e)}")
            state_manager = get_sync_state_manager()
            state_manager.mark_failed(task_id, str(e))

    def _run_sync_period(
        self, task_id: str, start_date: str, end_date: Optional[str], max_pages: int
    ) -> None:
        """백그라운드에서 기간별 영화 동기화 실행"""
        try:
            asyncio.run(
                self._sync_period_impl(task_id, start_date, end_date, max_pages)
            )
        except Exception as e:
            logger.error(f"Period sync failed: {str(e)}")
            state_manager = get_sync_state_manager()
            state_manager.mark_failed(task_id, str(e))

    async def _sync_popular_impl(self, task_id: str, max_pages: int) -> None:
        """인기 영화 동기화 구현"""
        from app.database import db_connector

        state_manager = get_sync_state_manager()
        state_manager.mark_started(task_id)
        state_manager.update_progress(task_id, total_pages=max_pages)

        try:
            tmdb_service = TMDBService()
            movies = await tmdb_service.fetch_popular_movies(max_pages)

            with db_connector.get_session() as session:
                service = MovieService(session)

                movies_data = []
                for movie in movies:
                    genre_korean = convert_genre_ids_to_korean(movie.genre_ids)

                    movies_data.append(
                        {
                            "tmdb_id": movie.id,
                            "title": movie.title,
                            "release_date": movie.release_date,
                            "genre": genre_korean,
                            "poster_url": tmdb_service.get_poster_url(
                                movie.poster_path
                            ),
                            "tmdb_rating": movie.vote_average,
                            "overview": movie.overview,
                            "popularity": movie.popularity,
                            "vote_count": movie.vote_count,
                            "original_title": movie.original_title,
                            "original_language": movie.original_language,
                            "adult": movie.adult,
                            "backdrop_path": movie.backdrop_path,
                        }
                    )

                inserted, updated, failed = service.bulk_upsert_movies(movies_data)

                state_manager.update_progress(
                    task_id,
                    current_page=max_pages,
                    movies_collected=len(movies),
                    movies_inserted=inserted,
                    movies_updated=updated,
                    movies_failed=failed,
                )

            state_manager.mark_completed(task_id)

        except Exception as e:
            logger.error(f"Popular sync implementation failed: {str(e)}")
            state_manager.mark_failed(task_id, str(e))
            raise

    async def _sync_latest_impl(
        self, task_id: str, days_back: int, max_pages: int
    ) -> None:
        """최신 영화 동기화 구현"""
        from app.database import db_connector

        state_manager = get_sync_state_manager()
        state_manager.mark_started(task_id)
        state_manager.update_progress(task_id, total_pages=max_pages)

        try:
            tmdb_service = TMDBService()
            movies = await tmdb_service.fetch_latest_movies(days_back, max_pages)

            with db_connector.get_session() as session:
                service = MovieService(session)

                movies_data = []
                for movie in movies:
                    genre_korean = convert_genre_ids_to_korean(movie.genre_ids)

                    movies_data.append(
                        {
                            "tmdb_id": movie.id,
                            "title": movie.title,
                            "release_date": movie.release_date,
                            "genre": genre_korean,
                            "poster_url": tmdb_service.get_poster_url(
                                movie.poster_path
                            ),
                            "tmdb_rating": movie.vote_average,
                            "overview": movie.overview,
                            "popularity": movie.popularity,
                            "vote_count": movie.vote_count,
                            "original_title": movie.original_title,
                            "original_language": movie.original_language,
                            "adult": movie.adult,
                            "backdrop_path": movie.backdrop_path,
                        }
                    )

                inserted, updated, failed = service.bulk_upsert_movies(movies_data)

                state_manager.update_progress(
                    task_id,
                    current_page=max_pages,
                    movies_collected=len(movies),
                    movies_inserted=inserted,
                    movies_updated=updated,
                    movies_failed=failed,
                )

            state_manager.mark_completed(task_id)

        except Exception as e:
            logger.error(f"Latest sync implementation failed: {str(e)}")
            state_manager.mark_failed(task_id, str(e))
            raise

    async def _sync_period_impl(
        self,
        task_id: str,
        start_date: str,
        end_date: Optional[str],
        max_pages: int,
    ) -> None:
        """기간별 영화 동기화 구현"""
        from app.database import db_connector

        state_manager = get_sync_state_manager()
        state_manager.mark_started(task_id)
        state_manager.update_progress(task_id, total_pages=max_pages)

        try:
            tmdb_service = TMDBService()
            movies = await tmdb_service.fetch_movies_by_period(
                start_date, end_date, max_pages
            )

            with db_connector.get_session() as session:
                service = MovieService(session)

                movies_data = []
                for movie in movies:
                    genre_korean = convert_genre_ids_to_korean(movie.genre_ids)

                    movies_data.append(
                        {
                            "tmdb_id": movie.id,
                            "title": movie.title,
                            "release_date": movie.release_date,
                            "genre": genre_korean,
                            "poster_url": tmdb_service.get_poster_url(
                                movie.poster_path
                            ),
                            "tmdb_rating": movie.vote_average,
                            "overview": movie.overview,
                            "popularity": movie.popularity,
                            "vote_count": movie.vote_count,
                            "original_title": movie.original_title,
                            "original_language": movie.original_language,
                            "adult": movie.adult,
                            "backdrop_path": movie.backdrop_path,
                        }
                    )

                inserted, updated, failed = service.bulk_upsert_movies(movies_data)

                state_manager.update_progress(
                    task_id,
                    current_page=max_pages,
                    movies_collected=len(movies),
                    movies_inserted=inserted,
                    movies_updated=updated,
                    movies_failed=failed,
                )

            state_manager.mark_completed(task_id)

        except Exception as e:
            logger.error(f"Period sync implementation failed: {str(e)}")
            state_manager.mark_failed(task_id, str(e))
            raise


# 라우터 인스턴스 생성
movie_router = MovieRouter()
router = movie_router.router
