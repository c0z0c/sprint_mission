"""
TMDB API 서비스
"""

import os
import httpx
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from app.config import get_sync_config
from app.constants.tmdb_genres import convert_genre_ids_to_korean
from app.schemas.tmdb import TMDBMovieResponse

import logging
from helper_dev_utils import get_auto_logger

logger = get_auto_logger(log_level=logging.DEBUG)


class TMDBService:
    """
    TMDB API 호출을 담당하는 서비스 클래스
    """

    def __init__(self):
        """TMDBService 초기화"""
        self.config = get_sync_config()
        self.base_url = self.config.get("tmdb.base_url")
        self.language = self.config.get("tmdb.language", "ko-KR")
        self.region = self.config.get("tmdb.region", "KR")

        # Rate Limiting 설정
        self.requests_per_second = self.config.get(
            "tmdb.rate_limiting.requests_per_second", 40
        )
        self.burst_size = self.config.get("tmdb.rate_limiting.burst_size", 10)

        # 환경변수에서 TMDB 토큰 가져오기
        self.access_token = os.getenv("TMDB_ACCESS_TOKEN")
        if not self.access_token:
            logger.warning("TMDB_ACCESS_TOKEN not found in environment variables")

        # Rate Limiting을 위한 Semaphore
        self._semaphore = asyncio.Semaphore(self.burst_size)
        self._last_request_time = 0.0
        self._request_interval = 1.0 / self.requests_per_second  # 초당 요청 간격

    def _get_headers(self) -> Dict[str, str]:
        """
        TMDB API 요청 헤더 생성

        Returns:
            Dict[str, str]: 헤더 딕셔너리
        """
        return {
            "accept": "application/json",
            "Authorization": f"Bearer {self.access_token}",
        }

    async def _rate_limited_request(
        self, client: httpx.AsyncClient, url: str, params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Rate Limiting이 적용된 HTTP 요청

        Args:
            client: httpx AsyncClient
            url: 요청 URL
            params: 요청 파라미터

        Returns:
            Optional[Dict]: 응답 JSON 또는 None (실패 시)
        """
        async with self._semaphore:
            # Rate Limiting: 이전 요청과의 최소 간격 보장
            now = asyncio.get_event_loop().time()
            time_since_last_request = now - self._last_request_time

            if time_since_last_request < self._request_interval:
                await asyncio.sleep(self._request_interval - time_since_last_request)

            self._last_request_time = asyncio.get_event_loop().time()

            try:
                response = await client.get(
                    url, headers=self._get_headers(), params=params, timeout=10.0
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(
                        f"TMDB API request failed: {response.status_code} - {url}"
                    )
                    return None

            except Exception as e:
                logger.error(f"TMDB API request error: {str(e)} - {url}")
                return None

    async def fetch_popular_movies(
        self, max_pages: int = 10
    ) -> List[TMDBMovieResponse]:
        """
        TMDB 인기 영화 목록 수집

        Args:
            max_pages: 수집할 최대 페이지 수 (페이지당 20건)

        Returns:
            List[TMDBMovieResponse]: 수집된 영화 목록
        """
        logger.info(f"Fetching popular movies (max {max_pages} pages)...")

        url = f"{self.base_url}/movie/popular"
        params_base = {
            "language": self.language,
            "region": self.region,
        }

        all_movies = []

        async with httpx.AsyncClient() as client:
            for page in range(1, max_pages + 1):
                params = {**params_base, "page": page}

                data = await self._rate_limited_request(client, url, params)

                if not data:
                    logger.warning(f"Failed to fetch page {page}, stopping...")
                    break

                movies = data.get("results", [])
                if not movies:
                    logger.info(f"No more movies at page {page}")
                    break

                # 장르 ID → 한글 변환
                for movie in movies:
                    genre_ids = movie.get("genre_ids")
                    movie["genre_korean"] = convert_genre_ids_to_korean(genre_ids)

                all_movies.extend(movies)
                logger.debug(
                    f"Fetched page {page}/{max_pages}: {len(movies)} movies (total: {len(all_movies)})"
                )

        logger.info(f"Fetched {len(all_movies)} popular movies from {page} pages")
        return [self._parse_movie(movie) for movie in all_movies]

    async def fetch_movies_by_period(
        self, start_date: str, end_date: Optional[str] = None, max_pages: int = 10
    ) -> List[TMDBMovieResponse]:
        """
        특정 기간의 영화 목록 수집

        Args:
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD, None이면 오늘)
            max_pages: 수집할 최대 페이지 수

        Returns:
            List[TMDBMovieResponse]: 수집된 영화 목록
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(
            f"Fetching movies from {start_date} to {end_date} (max {max_pages} pages)..."
        )

        url = f"{self.base_url}/discover/movie"
        params_base = {
            "language": self.language,
            "region": self.region,
            "sort_by": "primary_release_date.desc",
            "primary_release_date.gte": start_date,
            "primary_release_date.lte": end_date,
            "include_adult": "false",
        }

        all_movies = []

        async with httpx.AsyncClient() as client:
            for page in range(1, max_pages + 1):
                params = {**params_base, "page": page}

                data = await self._rate_limited_request(client, url, params)

                if not data:
                    logger.warning(f"Failed to fetch page {page}, stopping...")
                    break

                movies = data.get("results", [])
                if not movies:
                    logger.info(f"No more movies at page {page}")
                    break

                # 장르 ID → 한글 변환
                for movie in movies:
                    genre_ids = movie.get("genre_ids")
                    movie["genre_korean"] = convert_genre_ids_to_korean(genre_ids)

                all_movies.extend(movies)
                logger.debug(
                    f"Fetched page {page}/{max_pages}: {len(movies)} movies (total: {len(all_movies)})"
                )

        logger.info(
            f"Fetched {len(all_movies)} movies from {start_date} to {end_date} ({page} pages)"
        )
        return [self._parse_movie(movie) for movie in all_movies]

    async def fetch_latest_movies(
        self, days_back: int = 7, max_pages: int = 5
    ) -> List[TMDBMovieResponse]:
        """
        최근 N일간의 영화 목록 수집

        Args:
            days_back: 과거 며칠까지 조회할지 (기본 7일)
            max_pages: 수집할 최대 페이지 수

        Returns:
            List[TMDBMovieResponse]: 수집된 영화 목록
        """
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        return await self.fetch_movies_by_period(start_date, end_date, max_pages)

    def _parse_movie(self, raw_movie: Dict[str, Any]) -> TMDBMovieResponse:
        """
        TMDB API 응답을 TMDBMovieResponse로 파싱

        Args:
            raw_movie: TMDB API 원본 응답

        Returns:
            TMDBMovieResponse: 파싱된 영화 데이터
        """
        # backdrop_path를 전체 URL로 변환 (프론트엔드에서 사용)
        backdrop_path = raw_movie.get("backdrop_path")
        if backdrop_path:
            backdrop_path = f"https://image.tmdb.org/t/p/w1280{backdrop_path}"

        return TMDBMovieResponse(
            id=raw_movie.get("id"),
            title=raw_movie.get("title"),
            original_title=raw_movie.get("original_title"),
            overview=raw_movie.get("overview"),
            release_date=raw_movie.get("release_date"),
            vote_average=raw_movie.get("vote_average"),
            vote_count=raw_movie.get("vote_count"),
            popularity=raw_movie.get("popularity"),
            genre_ids=raw_movie.get("genre_ids"),
            original_language=raw_movie.get("original_language"),
            poster_path=raw_movie.get("poster_path"),
            backdrop_path=backdrop_path,
            adult=raw_movie.get("adult", False),
            video=raw_movie.get("video", False),
        )

    def get_poster_url(self, poster_path: Optional[str]) -> Optional[str]:
        """
        포스터 경로를 전체 URL로 변환

        Args:
            poster_path: TMDB 포스터 경로 (예: /abc123.jpg)

        Returns:
            Optional[str]: 전체 URL 또는 None
        """
        if not poster_path:
            return None
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
