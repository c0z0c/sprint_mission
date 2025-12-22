"""방문자 통계 API 라우터

방문자 수 조회 및 일별 통계 제공
"""

import logging
from typing import Dict, List
from fastapi import APIRouter, Depends
from sqlmodel import Session

from app.database.DatabaseConnector import get_db
from app.services.VisitorService import VisitorService

logger = logging.getLogger(__name__)


class VisitorRouter:
    """방문자 통계 라우터

    Endpoints:
        GET /visitors/count: 총 방문자 수 및 서버 시작 시간 조회
        GET /visitors/stats/daily: 일별 방문자 통계 조회 (향후 확장용)
    """

    def __init__(self):
        self.router = APIRouter(prefix="/visitors", tags=["visitors"])
        self._setup_routes()

    def _setup_routes(self):
        """라우트 설정"""
        self.router.add_api_route(
            "/count",
            self.get_visitor_count,
            methods=["GET"],
            response_model=Dict,
            summary="총 방문자 수 조회",
            description="서버 시작 이후 총 방문자 수 및 서버 시작 시간 반환",
        )

        self.router.add_api_route(
            "/stats/daily",
            self.get_daily_stats,
            methods=["GET"],
            response_model=List[Dict],
            summary="일별 방문자 통계 조회",
            description="최근 N일간 일별 방문자 통계 (향후 프론트엔드 확장용)",
        )

    def get_visitor_count(self, db: Session = Depends(get_db)) -> Dict:
        """총 방문자 수 및 서버 시작 시간 조회

        Args:
            db: 데이터베이스 세션

        Returns:
            Dict:
                - total_visitors: 총 방문자 수
                - unique_visitors: 고유 방문자 수
                - server_start_time: 서버 시작 시간 (ISO 8601)
        """
        # 순환 임포트 방지를 위한 지연 임포트
        from app.main import SERVER_START_TIME

        visitor_service = VisitorService(db)
        total_count = visitor_service.get_total_count()
        unique_count = visitor_service.get_unique_visitors_count()

        logger.debug(f"방문자 통계 조회: total={total_count}, unique={unique_count}")

        return {
            "total_visitors": total_count,
            "unique_visitors": unique_count,
            "server_start_time": (
                SERVER_START_TIME.isoformat() if SERVER_START_TIME else None
            ),
        }

    def get_daily_stats(
        self, days: int = 7, db: Session = Depends(get_db)
    ) -> List[Dict]:
        """일별 방문자 통계 조회

        Args:
            days: 조회할 일수 (기본 7일)
            db: 데이터베이스 세션

        Returns:
            List[Dict]: 일별 통계 리스트
                - date: 날짜 (YYYY-MM-DD)
                - total_visits: 총 방문 수
                - unique_visitors: 고유 방문자 수
        """
        visitor_service = VisitorService(db)
        stats = visitor_service.get_daily_stats(days)

        logger.debug(f"일별 통계 조회: {len(stats)}일")

        return stats


# 라우터 인스턴스 생성
visitor_router_instance = VisitorRouter()
visitor_router = visitor_router_instance.router
