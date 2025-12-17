"""
FastAPI 메인 애플리케이션
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pathlib import Path

from app.database import db_connector
from app.routes import movie_router, review_router

import logging
from helper_dev_utils import get_auto_logger

logger = get_auto_logger(log_level=logging.DEBUG)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    애플리케이션 시작/종료 시 실행되는 이벤트

    Args:
        app: FastAPI 애플리케이션 인스턴스
    """
    # 시작 시: 데이터베이스 테이블 생성
    logger.debug("애플리케이션 시작: 데이터베이스 테이블 생성 중...")
    db_connector.create_tables()
    logger.debug("데이터베이스 테이블 생성 완료")

    yield

    # 종료 시 정리 작업 (필요시 추가)
    logger.debug("애플리케이션 종료")


# FastAPI 애플리케이션 생성
app = FastAPI(
    title="Movie Review Sentiment Analysis API",
    description="영화 리뷰 감성 분석 서비스 백엔드 API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 설정 (프론트엔드 연동을 위해)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 구체적인 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 디렉토리 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

# 정적 파일 서빙 (포스터 이미지)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
else:
    logger.warning(f"Static directory not found: {STATIC_DIR}")

# 라우터 등록
app.include_router(movie_router)
app.include_router(review_router)


@app.get("/", tags=["health"])
def health_check():
    """
    서버 상태 확인 엔드포인트

    Returns:
        dict: 서버 상태 정보
    """
    return {
        "status": "healthy",
        "message": "Movie Review Sentiment Analysis API is running",
        "version": "1.0.0",
    }


@app.get("/health", tags=["health"])
def health():
    """
    헬스체크 엔드포인트

    Returns:
        dict: 상태 정보
    """
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 개발 모드: 코드 변경 시 자동 재시작
    )
