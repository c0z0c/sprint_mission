"""
리뷰(Review) 관련 스키마 클래스
"""

from pydantic import BaseModel, Field
from typing import Optional, TYPE_CHECKING, Any
from datetime import datetime
from typing import List

if TYPE_CHECKING:
    from app.schemas.movie import MovieResponse

import logging
from helper_dev_utils import get_auto_logger

logger = get_auto_logger(log_level=logging.DEBUG)


class ReviewCreate(BaseModel):
    """
    리뷰 등록 요청 스키마

    Attributes:
        tmdb_id: TMDB 영화 ID
        author: 작성자 이름
        content: 리뷰 내용
    """

    tmdb_id: int = Field(..., description="TMDB 영화 ID")
    author: str = Field(..., max_length=100, description="작성자 이름")
    content: str = Field(..., max_length=2000, description="리뷰 내용")


class ReviewResponse(BaseModel):
    """
    리뷰 조회 응답 스키마

    Attributes:
        id: 리뷰 ID
        tmdb_id: TMDB 영화 ID
        author: 작성자 이름
        content: 리뷰 내용
        is_positive: 감성 분석 결과 (1: 긍정, 0: 부정, None: 미분석)
        created_at: 생성 시간
        updated_at: 수정 시간
    """

    id: int = Field(..., description="리뷰 ID")
    tmdb_id: int = Field(..., description="TMDB 영화 ID")
    author: str = Field(..., description="작성자 이름")
    content: str = Field(..., description="리뷰 내용")
    is_positive: Optional[int] = Field(
        None, description="감성 분석 결과 (1: 긍정, 0: 부정, None: 미분석)"
    )
    created_at: datetime = Field(..., description="생성 시간")
    updated_at: datetime = Field(..., description="수정 시간")

    class Config:
        from_attributes = True


class ReviewWithMovie(ReviewResponse):
    """
    영화 정보 포함 리뷰 조회 응답 스키마

    Attributes:
        movie: 영화 정보
    """

    movie: Any = Field(..., description="영화 정보")  # 순환 참조 방지를 위해 Any 사용


class ReviewPaginationResponse(BaseModel):
    """
    리뷰 목록 페이지네이션 응답 스키마

    Attributes:
        total: 전체 리뷰 수
        page: 현재 페이지 번호
        page_size: 페이지당 항목 수
        total_pages: 전체 페이지 수
        reviews: 리뷰 목록 (영화 정보 포함)
    """

    total: int = Field(..., description="전체 리뷰 수")
    page: int = Field(..., description="현재 페이지 번호")
    page_size: int = Field(..., description="페이지당 항목 수")
    total_pages: int = Field(..., description="전체 페이지 수")
    reviews: List[ReviewWithMovie] = Field(
        ..., description="리뷰 목록 (영화 정보 포함)"
    )


class ReviewSearchFilters(BaseModel):
    """
    리뷰 검색 필터 스키마

    Attributes:
        author: 작성자 이름 (부분 검색, 대소문자 무시)
        content: 리뷰 내용 (부분 검색, 대소문자 무시)
        sentiment: 감성 필터 (positive, negative, all)
        movie_title: 영화 제목 (부분 검색, 대소문자 무시)
        tmdb_id: TMDB 영화 ID
        created_from: 생성일 시작 (ISO 8601 형식)
        created_to: 생성일 종료 (ISO 8601 형식)
        sort_by: 정렬 필드 (created_at, author)
        sort_order: 정렬 방향 (asc, desc)
        page: 페이지 번호
        page_size: 페이지당 항목 수
    """

    author: Optional[str] = Field(
        None, description="작성자 이름 (부분 검색, 대소문자 무시)"
    )
    content: Optional[str] = Field(
        None, description="리뷰 내용 (부분 검색, 대소문자 무시)"
    )
    sentiment: str = Field("all", description="감성 필터 (positive, negative, all)")
    movie_title: Optional[str] = Field(
        None, description="영화 제목 (부분 검색, 대소문자 무시)"
    )
    tmdb_id: Optional[int] = Field(None, description="TMDB 영화 ID")
    created_from: Optional[datetime] = Field(
        None, description="생성일 시작 (ISO 8601 형식)"
    )
    created_to: Optional[datetime] = Field(
        None, description="생성일 종료 (ISO 8601 형식)"
    )
    sort_by: str = Field("created_at", description="정렬 필드 (created_at, author)")
    sort_order: str = Field("desc", description="정렬 방향 (asc, desc)")
    page: int = Field(1, ge=1, description="페이지 번호")
    page_size: int = Field(10, ge=1, le=100, description="페이지당 항목 수")
