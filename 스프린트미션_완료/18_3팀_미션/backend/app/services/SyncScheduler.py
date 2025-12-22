"""
동기화 스케줄러
"""

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
import asyncio

from app.config import get_sync_config
from app.services.TMDBService import TMDBService
from app.services.SyncStateManager import get_sync_state_manager
from app.schemas.tmdb import SyncType
from app.database import db_connector
from app.services.MovieService import MovieService

import logging
from helper_dev_utils import get_auto_logger

logger = get_auto_logger(log_level=logging.DEBUG)


class SyncScheduler:
    """
    TMDB 동기화 작업 스케줄러
    """

    def __init__(self):
        """SyncScheduler 초기화"""
        self.config = get_sync_config()
        self.scheduler = BackgroundScheduler()
        self.state_manager = get_sync_state_manager()
        self.tmdb_service = TMDBService()

        logger.info("SyncScheduler initialized")

    async def run_initial_sync_if_needed(self) -> None:
        """DB가 비어있으면 초기 동기화 실행"""
        initial_sync_enabled = self.config.get("initial_sync.enabled", False)

        if not initial_sync_enabled:
            logger.info("Initial sync disabled in config.")
            return

        # DB에 영화가 있는지 확인
        with db_connector.get_session() as session:
            from sqlmodel import select, func
            from app.models.MovieModel import MovieModel

            movie_count = session.exec(select(func.count(MovieModel.id))).first()

            if movie_count > 0:
                logger.info(
                    f"Database already has {movie_count} movies. Skipping initial sync."
                )
                return

        logger.info("Database is empty. Starting initial sync...")

        # 인기 영화 동기화
        popular_enabled = self.config.get("initial_sync.popular.enabled", False)
        if popular_enabled:
            max_pages = self.config.get("initial_sync.popular.max_pages", 10)
            logger.info(
                f"Initial sync: Fetching popular movies (max_pages={max_pages})..."
            )

            try:
                await self._sync_popular_movies(max_pages, is_initial=True)
                logger.info(f"Initial popular sync completed successfully.")
            except Exception as e:
                logger.error(f"Initial popular sync failed: {str(e)}")

        # 최신 영화 동기화
        latest_enabled = self.config.get("initial_sync.latest.enabled", False)
        if latest_enabled:
            start_date = self.config.get("initial_sync.latest.start_date", "2020-01-01")
            end_date = self.config.get("initial_sync.latest.end_date", None)
            max_pages = self.config.get("initial_sync.latest.max_pages", 5)

            logger.info(
                f"Initial sync: Fetching latest movies (start_date={start_date}, max_pages={max_pages})..."
            )

            try:
                from datetime import datetime

                if end_date is None:
                    end_date = datetime.now().strftime("%Y-%m-%d")

                movies = await self.tmdb_service.fetch_movies_by_period(
                    start_date, end_date, max_pages
                )

                # DB 저장 로직 (popular sync와 동일)
                with db_connector.get_session() as session:
                    service = MovieService(session)
                    from app.constants.tmdb_genres import convert_genre_ids_to_korean

                    movies_data = []
                    for movie in movies:
                        genre_korean = convert_genre_ids_to_korean(movie.genre_ids)
                        movies_data.append(
                            {
                                "tmdb_id": movie.id,
                                "title": movie.title,
                                "release_date": movie.release_date,
                                "genre": genre_korean,
                                "poster_url": self.tmdb_service.get_poster_url(
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

                    service.bulk_upsert_movies(movies_data)

                # 포스터 다운로드
                logger.info("Starting poster download for initial latest movies...")
                await self._download_posters_for_movies(movies)
                logger.info("Initial latest poster download completed.")

                logger.info(f"Initial latest sync completed successfully.")
            except Exception as e:
                logger.error(f"Initial latest sync failed: {str(e)}")

    def start(self) -> None:
        """스케줄러 시작"""
        scheduler_enabled = self.config.get("scheduler.enabled", False)

        if not scheduler_enabled:
            logger.info("Scheduler disabled in config. Skipping scheduler start.")
            return

        # 인기 영화 동기화 스케줄 등록
        popular_sync_enabled = self.config.get("scheduler.popular_sync.enabled", False)
        if popular_sync_enabled:
            cron = self.config.get("scheduler.popular_sync.cron", "0 2 * * *")
            max_pages = self.config.get("scheduler.popular_sync.max_pages", 5)

            self.scheduler.add_job(
                self._scheduled_popular_sync,
                CronTrigger.from_crontab(cron),
                args=[max_pages],
                id="popular_sync",
                name="Popular Movies Sync",
                replace_existing=True,
            )
            logger.info(f"Scheduled popular sync: {cron} (max_pages={max_pages})")

        # 최신 영화 동기화 스케줄 등록
        latest_sync_enabled = self.config.get("scheduler.latest_sync.enabled", False)
        if latest_sync_enabled:
            cron = self.config.get("scheduler.latest_sync.cron", "0 3 * * 0")
            days_back = self.config.get("scheduler.latest_sync.days_back", 7)
            max_pages = self.config.get("scheduler.latest_sync.max_pages", 3)

            self.scheduler.add_job(
                self._scheduled_latest_sync,
                CronTrigger.from_crontab(cron),
                args=[days_back, max_pages],
                id="latest_sync",
                name="Latest Movies Sync",
                replace_existing=True,
            )
            logger.info(
                f"Scheduled latest sync: {cron} (days_back={days_back}, max_pages={max_pages})"
            )

        self.scheduler.start()
        logger.info("Scheduler started successfully")

    def stop(self) -> None:
        """스케줄러 종료"""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Scheduler stopped")

    def _scheduled_popular_sync(self, max_pages: int) -> None:
        """스케줄된 인기 영화 동기화"""
        logger.info(f"Starting scheduled popular sync (max_pages={max_pages})...")

        try:
            # 비동기 함수를 동기적으로 실행
            asyncio.run(self._sync_popular_movies(max_pages))
        except Exception as e:
            logger.error(f"Scheduled popular sync failed: {str(e)}")

    def _scheduled_latest_sync(self, days_back: int, max_pages: int) -> None:
        """스케줄된 최신 영화 동기화"""
        logger.info(
            f"Starting scheduled latest sync (days_back={days_back}, max_pages={max_pages})..."
        )

        try:
            # 비동기 함수를 동기적으로 실행
            asyncio.run(self._sync_latest_movies(days_back, max_pages))
        except Exception as e:
            logger.error(f"Scheduled latest sync failed: {str(e)}")

    async def _sync_popular_movies(
        self, max_pages: int, is_initial: bool = False
    ) -> None:
        """인기 영화 동기화 실행"""
        task_id = self.state_manager.create_task(SyncType.POPULAR)

        # 초기 동기화인 경우 task_id 저장
        if is_initial:
            self.state_manager.set_initial_sync_task_id(task_id)

        self.state_manager.mark_started(task_id)

        # 전체 페이지 수 설정
        self.state_manager.update_progress(
            task_id,
            total_pages=max_pages,
            current_page=0,
        )

        try:
            # TMDB API로부터 영화 수집
            movies = await self.tmdb_service.fetch_popular_movies(max_pages)

            # DB에 저장
            with db_connector.get_session() as session:
                service = MovieService(session)

                movies_data = []
                for movie in movies:
                    # genre_ids를 한글 텍스트로 변환
                    from app.constants.tmdb_genres import convert_genre_ids_to_korean

                    genre_korean = convert_genre_ids_to_korean(movie.genre_ids)

                    movies_data.append(
                        {
                            "tmdb_id": movie.id,
                            "title": movie.title,
                            "release_date": movie.release_date,
                            "genre": genre_korean,
                            "poster_url": self.tmdb_service.get_poster_url(
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

                self.state_manager.update_progress(
                    task_id,
                    current_page=max_pages,
                    movies_collected=len(movies),
                    movies_inserted=inserted,
                    movies_updated=updated,
                    movies_failed=failed,
                )

            # 포스터 다운로드
            logger.info("Starting poster download for popular movies...")
            await self._download_posters_for_movies(movies, task_id)
            logger.info("Popular movies poster download completed.")

            self.state_manager.mark_completed(task_id)

        except Exception as e:
            logger.error(f"Popular sync failed: {str(e)}")
            self.state_manager.mark_failed(task_id, str(e))

    async def _sync_latest_movies(self, days_back: int, max_pages: int) -> None:
        """최신 영화 동기화 실행"""
        task_id = self.state_manager.create_task(SyncType.LATEST)
        self.state_manager.mark_started(task_id)

        try:
            # TMDB API로부터 영화 수집
            movies = await self.tmdb_service.fetch_latest_movies(days_back, max_pages)

            # DB에 저장
            with db_connector.get_session() as session:
                service = MovieService(session)

                movies_data = []
                for movie in movies:
                    # genre_ids를 한글 텍스트로 변환
                    from app.constants.tmdb_genres import convert_genre_ids_to_korean

                    genre_korean = convert_genre_ids_to_korean(movie.genre_ids)

                    movies_data.append(
                        {
                            "tmdb_id": movie.id,
                            "title": movie.title,
                            "release_date": movie.release_date,
                            "genre": genre_korean,
                            "poster_url": self.tmdb_service.get_poster_url(
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

                self.state_manager.update_progress(
                    task_id,
                    movies_collected=len(movies),
                    movies_inserted=inserted,
                    movies_updated=updated,
                    movies_failed=failed,
                )

            # 포스터 다운로드
            logger.info("Starting poster download for latest movies...")
            await self._download_posters_for_movies(movies)
            logger.info("Latest movies poster download completed.")

            self.state_manager.mark_completed(task_id)

        except Exception as e:
            logger.error(f"Latest sync failed: {str(e)}")
            self.state_manager.mark_failed(task_id, str(e))

    async def _download_posters_for_movies(
        self, movies: list, task_id: str = None
    ) -> None:
        """영화 목록에 대한 포스터 다운로드"""
        import aiohttp
        from pathlib import Path

        poster_dir = Path("data/posters")
        poster_dir.mkdir(parents=True, exist_ok=True)

        downloaded = 0
        failed = 0
        total_posters = len([m for m in movies if m.poster_path])

        async with aiohttp.ClientSession() as session:
            for idx, movie in enumerate(movies):
                try:
                    if not movie.poster_path:
                        continue

                    poster_url = self.tmdb_service.get_poster_url(movie.poster_path)
                    file_name = f"{movie.id}.jpg"
                    file_path = poster_dir / file_name

                    # 이미 다운로드된 경우 스킵
                    if file_path.exists():
                        continue

                    async with session.get(poster_url) as resp:
                        if resp.status == 200:
                            content = await resp.read()
                            file_path.write_bytes(content)

                            # DB 업데이트
                            with db_connector.get_session() as db_session:
                                service = MovieService(db_session)
                                db_movie = service.get_movie_by_tmdb_id(movie.id)
                                if db_movie:
                                    db_movie.poster_local_path = (
                                        f"data/posters/{file_name}"
                                    )
                                    db_session.commit()

                            downloaded += 1
                            logger.debug(f"Downloaded poster for movie {movie.id}")

                            # 진행률 업데이트
                            if task_id:
                                self.state_manager.update_progress(
                                    task_id,
                                    posters_downloaded=downloaded,
                                )
                        else:
                            failed += 1
                            logger.warning(
                                f"Failed to download poster for movie {movie.id}: HTTP {resp.status}"
                            )

                            # 실패 진행률 업데이트
                            if task_id:
                                self.state_manager.update_progress(
                                    task_id,
                                    posters_failed=failed,
                                )

                except Exception as e:
                    failed += 1
                    logger.error(
                        f"Error downloading poster for movie {movie.id}: {str(e)}"
                    )

                    # 실패 진행률 업데이트
                    if task_id:
                        self.state_manager.update_progress(
                            task_id,
                            posters_failed=failed,
                        )

        logger.info(
            f"Poster download completed: {downloaded} downloaded, {failed} failed"
        )


# 전역 스케줄러 인스턴스
_scheduler: SyncScheduler = None


def get_sync_scheduler() -> SyncScheduler:
    """
    전역 SyncScheduler 인스턴스 반환

    Returns:
        SyncScheduler: 스케줄러 인스턴스
    """
    global _scheduler
    if _scheduler is None:
        _scheduler = SyncScheduler()
    return _scheduler
