"""
API 클라이언트 헬퍼 모듈
requests 호출을 래핑하여 자동으로 로그를 출력합니다.
"""

import requests
import logging
from typing import Optional, Dict, Any
from helper_dev_utils import get_auto_logger

logger = get_auto_logger(log_level=logging.DEBUG)


def api_get(
    url: str, params: Optional[Dict[str, Any]] = None, **kwargs
) -> requests.Response:
    """
    GET 요청 with logging

    Args:
        url: 요청 URL
        params: 쿼리 파라미터
        **kwargs: requests.get에 전달할 추가 인자

    Returns:
        requests.Response 객체
    """
    logger.debug(f"[API GET] URL: {url}, params: {params}")
    try:
        response = requests.get(url, params=params, **kwargs)
        logger.debug(f"[API GET] Response: {response.status_code}, URL: {url}")
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"[API GET] Error: {str(e)}, URL: {url}")
        raise


def api_post(
    url: str, json: Optional[Dict[str, Any]] = None, **kwargs
) -> requests.Response:
    """
    POST 요청 with logging

    Args:
        url: 요청 URL
        json: JSON 데이터
        **kwargs: requests.post에 전달할 추가 인자

    Returns:
        requests.Response 객체
    """
    logger.debug(f"[API POST] URL: {url}, json: {json}")
    try:
        response = requests.post(url, json=json, **kwargs)
        logger.debug(f"[API POST] Response: {response.status_code}, URL: {url}")
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"[API POST] Error: {str(e)}, URL: {url}")
        raise


def api_put(
    url: str, json: Optional[Dict[str, Any]] = None, **kwargs
) -> requests.Response:
    """
    PUT 요청 with logging

    Args:
        url: 요청 URL
        json: JSON 데이터
        **kwargs: requests.put에 전달할 추가 인자

    Returns:
        requests.Response 객체
    """
    logger.debug(f"[API PUT] URL: {url}, json: {json}")
    try:
        response = requests.put(url, json=json, **kwargs)
        logger.debug(f"[API PUT] Response: {response.status_code}, URL: {url}")
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"[API PUT] Error: {str(e)}, URL: {url}")
        raise


def api_delete(url: str, **kwargs) -> requests.Response:
    """
    DELETE 요청 with logging

    Args:
        url: 요청 URL
        **kwargs: requests.delete에 전달할 추가 인자

    Returns:
        requests.Response 객체
    """
    logger.debug(f"[API DELETE] URL: {url}")
    try:
        response = requests.delete(url, **kwargs)
        logger.debug(f"[API DELETE] Response: {response.status_code}, URL: {url}")
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"[API DELETE] Error: {str(e)}, URL: {url}")
        raise


def api_patch(
    url: str, json: Optional[Dict[str, Any]] = None, **kwargs
) -> requests.Response:
    """
    PATCH 요청 with logging

    Args:
        url: 요청 URL
        json: JSON 데이터
        **kwargs: requests.patch에 전달할 추가 인자

    Returns:
        requests.Response 객체
    """
    logger.debug(f"[API PATCH] URL: {url}, json: {json}")
    try:
        response = requests.patch(url, json=json, **kwargs)
        logger.debug(f"[API PATCH] Response: {response.status_code}, URL: {url}")
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"[API PATCH] Error: {str(e)}, URL: {url}")
        raise


def get_health_status(
    base_url: str = "http://localhost:8000", timeout: tuple = (1, 2)
) -> Optional[Dict[str, Any]]:
    """
    서버 헬스 체크 및 초기 동기화 상태 조회

    Args:
        base_url: API 서버 주소
        timeout: 요청 타임아웃 (초)

    Returns:
        Optional[Dict[str, Any]]: 헬스 체크 결과
            - status: "ok"
            - ready: 서버 준비 완료 여부
            - initial_sync: 초기 동기화 상태
                - in_progress: 진행 중 여부
                - current: 현재 수집된 영화 수
                - total: 예상 전체 영화 수
                - sync_type: 동기화 유형
        None: 서버 연결 실패
    """
    try:
        response = requests.get(f"{base_url}/health", timeout=timeout)
        if response.status_code == 200:
            return response.json()
        logger.warning(f"[Health Check] Unexpected status code: {response.status_code}")
        return None
    except requests.exceptions.Timeout:
        logger.warning(f"[Health Check] Timeout after {timeout}s")
        return None
    except requests.exceptions.ConnectionError:
        logger.warning(f"[Health Check] Connection refused to {base_url}")
        return None
    except Exception as e:
        logger.error(f"[Health Check] Unexpected error: {str(e)}")
        return None


def get_visitor_count(
    base_url: str = "http://localhost:8000", timeout: tuple = (1, 2)
) -> Optional[Dict[str, Any]]:
    """
    방문자 수 조회 (총 방문자, 고유 방문자, 서버 시작 시간)

    Args:
        base_url: API 서버 주소
        timeout: 요청 타임아웃 (초)

    Returns:
        Optional[Dict[str, Any]]: 방문자 통계
            - total_visitors: 총 방문자 수
            - unique_visitors: 고유 방문자 수 (IP 해시 기준)
            - server_start_time: 서버 시작 시간 (ISO 8601)
        None: 서버 연결 실패
    """
    try:
        response = requests.get(f"{base_url}/visitors/count", timeout=timeout)
        if response.status_code == 200:
            return response.json()
        logger.warning(
            f"[Visitor Count] Unexpected status code: {response.status_code}"
        )
        return None
    except requests.exceptions.RequestException as e:
        logger.warning(f"[Visitor Count] Error: {str(e)}")
        return None
