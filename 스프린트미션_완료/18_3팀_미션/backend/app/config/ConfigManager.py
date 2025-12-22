"""
설정 관리자 클래스
"""

import os
import yaml
from pathlib import Path
from typing import Any, Optional, Dict
from functools import lru_cache

import logging
from helper_dev_utils import get_auto_logger

logger = get_auto_logger(log_level=logging.DEBUG)


class ConfigManager:
    """
    YAML 설정 파일을 로드하고 관리하는 클래스
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        ConfigManager 초기화

        Args:
            config_path: YAML 설정 파일 경로 (None이면 기본 경로 사용)
        """
        if config_path is None:
            # 기본 경로: backend/config/sync_config.yaml
            backend_dir = Path(__file__).resolve().parent.parent.parent
            config_path = backend_dir / "config" / "sync_config.yaml"

        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """
        YAML 설정 파일 로드
        """
        try:
            if not self.config_path.exists():
                logger.warning(
                    f"Config file not found: {self.config_path}. Using default settings."
                )
                self.config = self._get_default_config()
                return

            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}

            logger.info(f"Configuration loaded from: {self.config_path}")
            self._apply_env_overrides()

        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}. Using default settings.")
            self.config = self._get_default_config()

    def _apply_env_overrides(self) -> None:
        """
        환경변수로 설정 오버라이드

        예: SYNC_SCHEDULER_ENABLED=true → scheduler.enabled = true
        """
        # 스케줄러 활성화 여부
        if scheduler_enabled := os.getenv("SYNC_SCHEDULER_ENABLED"):
            self.config.setdefault("scheduler", {})["enabled"] = (
                scheduler_enabled.lower() == "true"
            )

        # 초기 동기화 활성화 여부
        if initial_sync_enabled := os.getenv("SYNC_INITIAL_ENABLED"):
            self.config.setdefault("initial_sync", {})["enabled"] = (
                initial_sync_enabled.lower() == "true"
            )

        # Rate Limiting
        if requests_per_second := os.getenv("SYNC_RATE_LIMIT"):
            self.config.setdefault("tmdb", {}).setdefault("rate_limiting", {})[
                "requests_per_second"
            ] = int(requests_per_second)

    def _get_default_config(self) -> Dict[str, Any]:
        """
        기본 설정 반환

        Returns:
            Dict[str, Any]: 기본 설정 딕셔너리
        """
        return {
            "tmdb": {
                "base_url": "https://api.themoviedb.org/3",
                "language": "ko-KR",
                "region": "KR",
                "rate_limiting": {"requests_per_second": 40, "burst_size": 10},
            },
            "initial_sync": {
                "enabled": True,
                "popular": {"enabled": True, "max_pages": 10},
                "latest": {
                    "enabled": True,
                    "start_date": "2020-01-01",
                    "end_date": None,
                    "max_pages": 5,
                },
            },
            "scheduler": {
                "enabled": False,
                "popular_sync": {"enabled": False, "cron": "0 2 * * *", "max_pages": 5},
                "latest_sync": {
                    "enabled": False,
                    "cron": "0 3 * * 0",
                    "days_back": 7,
                    "max_pages": 3,
                },
            },
            "poster": {
                "download_on_sync": True,
                "download_timeout": 10,
                "retry_count": 3,
                "concurrent_downloads": 5,
            },
            "error_handling": {
                "skip_on_error": True,
                "log_errors": True,
                "max_consecutive_errors": 10,
            },
            "logging": {
                "level": "INFO",
                "sync_progress": True,
                "detailed_stats": True,
            },
        }

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        중첩된 키 경로로 설정값 가져오기

        Args:
            key_path: 점으로 구분된 키 경로 (예: "tmdb.rate_limiting.requests_per_second")
            default: 키가 없을 때 반환할 기본값

        Returns:
            Any: 설정값 또는 기본값

        Examples:
            >>> config.get("tmdb.rate_limiting.requests_per_second")
            40
            >>> config.get("scheduler.enabled", False)
            False
        """
        keys = key_path.split(".")
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_all(self) -> Dict[str, Any]:
        """
        전체 설정 반환

        Returns:
            Dict[str, Any]: 전체 설정 딕셔너리
        """
        return self.config.copy()

    def reload(self) -> None:
        """
        설정 파일 재로드
        """
        logger.info("Reloading configuration...")
        self._load_config()


# 전역 설정 인스턴스 (싱글톤 패턴)
_config_manager: Optional[ConfigManager] = None


@lru_cache(maxsize=1)
def get_sync_config() -> ConfigManager:
    """
    전역 ConfigManager 인스턴스 반환 (싱글톤)

    Returns:
        ConfigManager: 설정 관리자 인스턴스
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
