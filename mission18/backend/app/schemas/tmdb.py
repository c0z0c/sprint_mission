"""
TMDB API 관련 스키마
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum

import logging
from helper_dev_utils import get_auto_logger

logger = get_auto_logger(log_level=logging.DEBUG)


# ==================== TMDB API 요청/응답 스키마 ====================


class TMDBMovieResponse(BaseModel):
    """
    TMDB API 영화 응답 스키마

    Note:
        TMDB API /movie/popular, /discover/movie 엔드포인트 응답
    """

    id: int = Field(..., description="TMDB 영화 ID")
    title: str = Field(..., description="영화 제목 (한글)")
    original_title: Optional[str] = Field(None, description="원제")
    overview: Optional[str] = Field(None, description="줄거리/개요")
    release_date: Optional[str] = Field(None, description="개봉일 (YYYY-MM-DD)")

    # 평점 및 인기도
    vote_average: Optional[float] = Field(None, description="TMDB 평점 (0-10)")
    vote_count: Optional[int] = Field(None, description="투표 수")
    popularity: Optional[float] = Field(None, description="인기도")

    # 장르 및 언어
    genre_ids: Optional[List[int]] = Field(None, description="장르 ID 배열")
    original_language: Optional[str] = Field(None, description="원어 (ISO 639-1)")

    # 이미지
    poster_path: Optional[str] = Field(None, description="포스터 경로")
    backdrop_path: Optional[str] = Field(None, description="배경 이미지 경로")

    # 기타
    adult: Optional[bool] = Field(False, description="성인 영화 여부")
    video: Optional[bool] = Field(False, description="비디오 포함 여부")


class TMDBSyncRequest(BaseModel):
    """
    TMDB 동기화 요청 스키마
    """

    max_pages: Optional[int] = Field(
        10, ge=1, le=100, description="수집할 최대 페이지 수 (1-100)"
    )
    start_date: Optional[str] = Field(
        None, description="시작 날짜 (YYYY-MM-DD, 기간별 수집 시 사용)"
    )
    end_date: Optional[str] = Field(
        None, description="종료 날짜 (YYYY-MM-DD, 기간별 수집 시 사용)"
    )


# ==================== 동기화 상태 스키마 ====================


class SyncStatus(str, Enum):
    """동기화 작업 상태"""

    PENDING = "pending"  # 대기 중
    RUNNING = "running"  # 실행 중
    COMPLETED = "completed"  # 완료
    FAILED = "failed"  # 실패
    CANCELLED = "cancelled"  # 취소됨


class SyncType(str, Enum):
    """동기화 작업 유형"""

    POPULAR = "popular"  # 인기 영화
    LATEST = "latest"  # 최신 영화
    PERIOD = "period"  # 기간별 영화


class SyncProgress(BaseModel):
    """
    동기화 진행 상황
    """

    current_page: int = Field(0, description="현재 처리 중인 페이지")
    total_pages: int = Field(0, description="전체 페이지 수")
    movies_collected: int = Field(0, description="수집된 영화 수")
    movies_inserted: int = Field(0, description="신규 등록된 영화 수")
    movies_updated: int = Field(0, description="업데이트된 영화 수")
    movies_failed: int = Field(0, description="실패한 영화 수")
    posters_downloaded: int = Field(0, description="다운로드된 포스터 수")
    posters_failed: int = Field(0, description="실패한 포스터 다운로드 수")


class SyncStateResponse(BaseModel):
    """
    동기화 작업 상태 응답
    """

    task_id: str = Field(..., description="작업 ID")
    sync_type: SyncType = Field(..., description="동기화 유형")
    status: SyncStatus = Field(..., description="작업 상태")
    progress: SyncProgress = Field(..., description="진행 상황")

    started_at: Optional[datetime] = Field(None, description="시작 시각")
    completed_at: Optional[datetime] = Field(None, description="완료 시각")
    error_message: Optional[str] = Field(None, description="에러 메시지")

    # 추가 정보
    elapsed_seconds: Optional[float] = Field(None, description="경과 시간 (초)")
    estimated_remaining_seconds: Optional[float] = Field(
        None, description="예상 남은 시간 (초)"
    )


class SyncStartResponse(BaseModel):
    """
    동기화 시작 응답
    """

    task_id: str = Field(..., description="작업 ID")
    sync_type: SyncType = Field(..., description="동기화 유형")
    message: str = Field(..., description="응답 메시지")
    status_url: str = Field(..., description="상태 조회 URL")


# ==================== 설정 스키마 ====================


class SyncConfigResponse(BaseModel):
    """
    동기화 설정 조회 응답
    """

    tmdb_base_url: str = Field(..., description="TMDB API 기본 URL")
    language: str = Field(..., description="언어 설정")
    region: str = Field(..., description="지역 설정")

    rate_limiting: dict = Field(..., description="Rate Limiting 설정")
    initial_sync: dict = Field(..., description="초기 동기화 설정")
    scheduler: dict = Field(..., description="스케줄러 설정")
    poster: dict = Field(..., description="포스터 다운로드 설정")
    error_handling: dict = Field(..., description="에러 처리 설정")


class SyncStatsResponse(BaseModel):
    """
    동기화 통계 응답
    """

    total_movies: int = Field(..., description="전체 영화 수")
    total_syncs: int = Field(..., description="전체 동기화 실행 횟수")
    successful_syncs: int = Field(..., description="성공한 동기화 횟수")
    failed_syncs: int = Field(..., description="실패한 동기화 횟수")
    last_sync_at: Optional[datetime] = Field(None, description="마지막 동기화 시각")

    # 활성 작업
    active_tasks: List[str] = Field([], description="현재 실행 중인 작업 ID 목록")
