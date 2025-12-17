"""
데이터베이스 연결 및 세션 관리 클래스
"""

from sqlmodel import create_engine, Session, SQLModel
from typing import Generator

import logging
from helper_dev_utils import get_auto_logger

logger = get_auto_logger(log_level=logging.DEBUG)


class DatabaseConnector:
    """
    SQLModel 기반 데이터베이스 연결 클래스
    """

    def __init__(self, database_url: str = "sqlite:///./movie_review.db"):
        """
        데이터베이스 연결 초기화

        Args:
            database_url: 데이터베이스 연결 URL
        """
        self.database_url = database_url
        self.engine = create_engine(
            database_url,
            connect_args=(
                {"check_same_thread": False} if "sqlite" in database_url else {}
            ),
            echo=True,  # 디버깅을 위한 SQL 쿼리 로그 출력
        )

    def create_tables(self) -> None:
        """
        데이터베이스 테이블 생성
        """
        SQLModel.metadata.create_all(self.engine)

    def get_session(self) -> Generator[Session, None, None]:
        """
        데이터베이스 세션 생성 및 반환

        Yields:
            Session: SQLModel 세션 객체
        """
        with Session(self.engine) as session:
            yield session


# 전역 데이터베이스 커넥터 인스턴스
db_connector = DatabaseConnector()


# FastAPI Depends용 헬퍼 함수
def get_db() -> Generator[Session, None, None]:
    """
    FastAPI Depends에서 사용할 DB 세션 생성 함수

    Yields:
        Session: SQLModel 세션 객체
    """
    return db_connector.get_session()
