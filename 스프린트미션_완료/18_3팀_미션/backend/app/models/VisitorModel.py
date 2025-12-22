"""방문자 로그 모델

방문자 IP 해시 기반 로그 저장 및 통계 조회
"""

from datetime import datetime
from typing import Optional
from sqlmodel import Field, SQLModel, Index


class VisitorModel(SQLModel, table=True):
    """방문자 로그 테이블

    Attributes:
        id: 고유 ID
        ip_hash: IP 주소 SHA256 해시값 (익명화)
        visit_time: 방문 시간 (자동 생성)
    """

    __tablename__ = "visitor_logs"

    __table_args__ = (Index("idx_visitor_ip_time", "ip_hash", "visit_time"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    ip_hash: str = Field(max_length=64, nullable=False, index=True)
    visit_time: datetime = Field(
        default_factory=datetime.now, nullable=False, index=True
    )
