"""
개발 유틸리티 헬퍼 함수
"""

import logging
import sys


def get_auto_logger(log_level=logging.INFO):
    """
    자동으로 로거를 생성하고 반환합니다.

    Args:
        log_level: 로그 레벨 (기본값: logging.INFO)

    Returns:
        logging.Logger: 설정된 로거
    """
    # 호출한 모듈의 이름을 가져옵니다
    frame = sys._getframe(1)
    module_name = frame.f_globals.get("__name__", "__main__")

    # 로거 생성
    logger = logging.getLogger(module_name)
    logger.setLevel(log_level)

    # 핸들러가 없으면 추가
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(log_level)

        # 포맷 설정
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
