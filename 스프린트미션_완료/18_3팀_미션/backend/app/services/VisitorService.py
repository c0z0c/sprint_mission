"""방문자 로깅 및 통계 서비스

IP 해시 기반 중복 방지 및 일별 통계 제공
"""

import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from sqlmodel import Session, select, func

from app.models.VisitorModel import VisitorModel

logger = logging.getLogger(__name__)


class VisitorService:
    """방문자 로깅 및 통계 서비스

    - IP 주소 SHA256 해시 처리
    - 5분 이내 동일 IP 중복 방지
    - 일별 방문자 통계 제공
    """

    DUPLICATE_WINDOW_MINUTES = 5

    def __init__(self, session: Session):
        self.session = session

    @staticmethod
    def hash_ip(ip_address: str) -> str:
        """IP 주소를 SHA256으로 해시 처리

        Args:
            ip_address: 원본 IP 주소

        Returns:
            str: SHA256 해시값 (64자 hex)
        """
        return hashlib.sha256(ip_address.encode()).hexdigest()

    def should_log_visit(self, ip_hash: str) -> bool:
        """방문 로그 기록 여부 판단 (5분 중복 체크)

        Args:
            ip_hash: IP 해시값

        Returns:
            bool: True이면 로그 기록, False이면 중복으로 스킵
        """
        cutoff_time = datetime.now() - timedelta(minutes=self.DUPLICATE_WINDOW_MINUTES)

        statement = select(VisitorModel).where(
            VisitorModel.ip_hash == ip_hash, VisitorModel.visit_time > cutoff_time
        )
        recent_visit = self.session.exec(statement).first()

        return recent_visit is None

    def log_visit(self, ip_address: str) -> Optional[VisitorModel]:
        """방문 로그 기록 (중복 체크 후)

        Args:
            ip_address: 방문자 IP 주소

        Returns:
            VisitorModel: 기록된 로그 (중복 시 None)
        """
        ip_hash = self.hash_ip(ip_address)

        if not self.should_log_visit(ip_hash):
            logger.debug(f"중복 방문 스킵: {ip_hash[:8]}... (5분 이내)")
            return None

        visitor_log = VisitorModel(ip_hash=ip_hash)
        self.session.add(visitor_log)
        self.session.commit()
        self.session.refresh(visitor_log)

        logger.debug(f"방문 로그 기록: {ip_hash[:8]}... at {visitor_log.visit_time}")
        return visitor_log

    def get_total_count(self) -> int:
        """총 방문자 수 조회

        Returns:
            int: 총 방문 로그 수
        """
        statement = select(func.count(VisitorModel.id))
        return self.session.exec(statement).one()

    def get_unique_visitors_count(self) -> int:
        """고유 방문자 수 조회 (IP 해시 기준)

        Returns:
            int: 고유 방문자 수
        """
        statement = select(func.count(func.distinct(VisitorModel.ip_hash)))
        return self.session.exec(statement).one()

    def get_daily_stats(self, days: int = 7) -> List[Dict]:
        """일별 방문자 통계 조회

        Args:
            days: 조회할 일수 (기본 7일)

        Returns:
            List[Dict]: 일별 통계 리스트
                - date: 날짜 (YYYY-MM-DD)
                - total_visits: 총 방문 수
                - unique_visitors: 고유 방문자 수
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        # SQLite date() 함수 사용
        statement = (
            select(
                func.date(VisitorModel.visit_time).label("date"),
                func.count(VisitorModel.id).label("total_visits"),
                func.count(func.distinct(VisitorModel.ip_hash)).label(
                    "unique_visitors"
                ),
            )
            .where(VisitorModel.visit_time >= cutoff_date)
            .group_by(func.date(VisitorModel.visit_time))
            .order_by(func.date(VisitorModel.visit_time).desc())
        )

        results = self.session.exec(statement).all()

        return [
            {
                "date": row.date,
                "total_visits": row.total_visits,
                "unique_visitors": row.unique_visitors,
            }
            for row in results
        ]
