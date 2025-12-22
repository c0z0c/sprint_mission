"""
동기화 상태 관리자
"""

import threading
import uuid
from typing import Dict, Optional
from datetime import datetime

from app.schemas.tmdb import (
    SyncStatus,
    SyncType,
    SyncProgress,
    SyncStateResponse,
)

import logging
from helper_dev_utils import get_auto_logger

logger = get_auto_logger(log_level=logging.DEBUG)


class SyncState:
    """
    개별 동기화 작업 상태
    """

    def __init__(self, task_id: str, sync_type: SyncType):
        self.task_id = task_id
        self.sync_type = sync_type
        self.status = SyncStatus.PENDING
        self.progress = SyncProgress()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error_message: Optional[str] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """작업 시작"""
        with self._lock:
            self.status = SyncStatus.RUNNING
            self.started_at = datetime.now()
            logger.info(f"Sync task started: {self.task_id} ({self.sync_type})")

    def complete(self) -> None:
        """작업 완료"""
        with self._lock:
            self.status = SyncStatus.COMPLETED
            self.completed_at = datetime.now()
            logger.info(
                f"Sync task completed: {self.task_id} - "
                f"{self.progress.movies_collected} movies collected, "
                f"{self.progress.movies_inserted} inserted, "
                f"{self.progress.movies_updated} updated"
            )

    def fail(self, error_message: str) -> None:
        """작업 실패"""
        with self._lock:
            self.status = SyncStatus.FAILED
            self.completed_at = datetime.now()
            self.error_message = error_message
            logger.error(f"Sync task failed: {self.task_id} - {error_message}")

    def cancel(self) -> None:
        """작업 취소"""
        with self._lock:
            self.status = SyncStatus.CANCELLED
            self.completed_at = datetime.now()
            logger.warning(f"Sync task cancelled: {self.task_id}")

    def update_progress(
        self,
        current_page: Optional[int] = None,
        total_pages: Optional[int] = None,
        movies_collected: Optional[int] = None,
        movies_inserted: Optional[int] = None,
        movies_updated: Optional[int] = None,
        movies_failed: Optional[int] = None,
        posters_downloaded: Optional[int] = None,
        posters_failed: Optional[int] = None,
    ) -> None:
        """진행 상황 업데이트"""
        with self._lock:
            if current_page is not None:
                self.progress.current_page = current_page
            if total_pages is not None:
                self.progress.total_pages = total_pages
            if movies_collected is not None:
                self.progress.movies_collected = movies_collected
            if movies_inserted is not None:
                self.progress.movies_inserted = movies_inserted
            if movies_updated is not None:
                self.progress.movies_updated = movies_updated
            if movies_failed is not None:
                self.progress.movies_failed = movies_failed
            if posters_downloaded is not None:
                self.progress.posters_downloaded = posters_downloaded
            if posters_failed is not None:
                self.progress.posters_failed = posters_failed

    def to_response(self) -> SyncStateResponse:
        """응답 스키마로 변환"""
        with self._lock:
            elapsed_seconds = None
            estimated_remaining_seconds = None

            if self.started_at:
                elapsed_seconds = (datetime.now() - self.started_at).total_seconds()

                # 진행 중일 때만 예상 시간 계산
                if self.status == SyncStatus.RUNNING and self.progress.current_page > 0:
                    avg_time_per_page = elapsed_seconds / self.progress.current_page
                    remaining_pages = (
                        self.progress.total_pages - self.progress.current_page
                    )
                    estimated_remaining_seconds = avg_time_per_page * remaining_pages

            return SyncStateResponse(
                task_id=self.task_id,
                sync_type=self.sync_type,
                status=self.status,
                progress=self.progress,
                started_at=self.started_at,
                completed_at=self.completed_at,
                error_message=self.error_message,
                elapsed_seconds=elapsed_seconds,
                estimated_remaining_seconds=estimated_remaining_seconds,
            )


class SyncStateManager:
    """
    동기화 작업 상태를 관리하는 싱글톤 클래스
    """

    _instance: Optional["SyncStateManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """초기화"""
        self._states: Dict[str, SyncState] = {}
        self._states_lock = threading.Lock()
        self._total_syncs = 0
        self._successful_syncs = 0
        self._failed_syncs = 0
        self._last_sync_at: Optional[datetime] = None
        self._initial_sync_task_id: Optional[str] = None
        logger.info("SyncStateManager initialized")

    def create_task(self, sync_type: SyncType) -> str:
        """
        새로운 동기화 작업 생성

        Args:
            sync_type: 동기화 유형

        Returns:
            str: 생성된 작업 ID
        """
        task_id = str(uuid.uuid4())
        with self._states_lock:
            self._states[task_id] = SyncState(task_id, sync_type)
            self._total_syncs += 1
        logger.debug(f"Created sync task: {task_id} ({sync_type})")
        return task_id

    def get_state(self, task_id: str) -> Optional[SyncState]:
        """
        작업 상태 조회

        Args:
            task_id: 작업 ID

        Returns:
            Optional[SyncState]: 작업 상태 또는 None
        """
        with self._states_lock:
            return self._states.get(task_id)

    def get_all_states(self) -> Dict[str, SyncState]:
        """
        전체 작업 상태 조회

        Returns:
            Dict[str, SyncState]: 작업 ID별 상태 딕셔너리
        """
        with self._states_lock:
            return self._states.copy()

    def get_active_tasks(self) -> list[str]:
        """
        현재 실행 중인 작업 ID 목록 조회

        Returns:
            list[str]: 실행 중인 작업 ID 목록
        """
        with self._states_lock:
            return [
                task_id
                for task_id, state in self._states.items()
                if state.status == SyncStatus.RUNNING
            ]

    def has_running_tasks(self) -> bool:
        """
        실행 중인 작업이 있는지 확인

        Returns:
            bool: 실행 중인 작업 존재 여부
        """
        return len(self.get_active_tasks()) > 0

    def mark_started(self, task_id: str) -> None:
        """작업 시작 표시"""
        if state := self.get_state(task_id):
            state.start()

    def mark_completed(self, task_id: str) -> None:
        """작업 완료 표시"""
        if state := self.get_state(task_id):
            state.complete()
            with self._states_lock:
                self._successful_syncs += 1
                self._last_sync_at = datetime.now()

    def mark_failed(self, task_id: str, error_message: str) -> None:
        """작업 실패 표시"""
        if state := self.get_state(task_id):
            state.fail(error_message)
            with self._states_lock:
                self._failed_syncs += 1
                self._last_sync_at = datetime.now()

    def update_progress(self, task_id: str, **kwargs) -> None:
        """진행 상황 업데이트"""
        if state := self.get_state(task_id):
            state.update_progress(**kwargs)

    def remove_task(self, task_id: str) -> None:
        """
        작업 상태 제거

        Args:
            task_id: 작업 ID
        """
        with self._states_lock:
            if task_id in self._states:
                del self._states[task_id]
                logger.debug(f"Removed sync task: {task_id}")

    def get_statistics(self) -> dict:
        """
        동기화 통계 조회

        Returns:
            dict: 통계 정보
        """
        with self._states_lock:
            return {
                "total_syncs": self._total_syncs,
                "successful_syncs": self._successful_syncs,
                "failed_syncs": self._failed_syncs,
                "last_sync_at": self._last_sync_at,
                "active_tasks": self.get_active_tasks(),
            }

    def set_initial_sync_task_id(self, task_id: str) -> None:
        """
        초기 동기화 task_id 설정

        Args:
            task_id: 초기 동기화 작업 ID
        """
        with self._states_lock:
            self._initial_sync_task_id = task_id
            logger.info(f"Initial sync task ID set: {task_id}")

    def get_initial_sync_status(self) -> dict:
        """
        초기 동기화 상태 조회

        Returns:
            dict: 초기 동기화 상태 정보
                - in_progress: 진행 중 여부
                - current: 현재 수집된 영화 수
                - total: 예상 전체 영화 수
                - sync_type: 동기화 유형
        """
        with self._states_lock:
            if not self._initial_sync_task_id:
                return {
                    "in_progress": False,
                    "current": 0,
                    "total": 0,
                    "sync_type": None,
                }

            state = self._states.get(self._initial_sync_task_id)
            if not state:
                return {
                    "in_progress": False,
                    "current": 0,
                    "total": 0,
                    "sync_type": None,
                }

            in_progress = state.status == SyncStatus.RUNNING

            # 전체 진행률 = 영화 수집(50%) + 포스터 다운로드(50%)
            movies_collected = state.progress.movies_collected
            total_movies = (
                state.progress.total_pages * 20 if state.progress.total_pages else 0
            )
            posters_downloaded = state.progress.posters_downloaded

            # 영화 수집 진행률 (50%)
            movie_progress = movies_collected if total_movies > 0 else 0
            # 포스터 다운로드 진행률 (50%)
            poster_progress = posters_downloaded
            # 전체 진행률
            current = movie_progress + poster_progress
            total = total_movies * 2 if total_movies > 0 else 0  # 영화 + 포스터

            return {
                "in_progress": in_progress,
                "current": current,
                "total": total,
                "sync_type": state.sync_type.value if state.sync_type else None,
                "movies_collected": movies_collected,
                "posters_downloaded": posters_downloaded,
            }


# 전역 싱글톤 인스턴스
def get_sync_state_manager() -> SyncStateManager:
    """
    SyncStateManager 싱글톤 인스턴스 반환

    Returns:
        SyncStateManager: 상태 관리자 인스턴스
    """
    return SyncStateManager()
