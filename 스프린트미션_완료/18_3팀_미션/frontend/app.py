"""
Streamlit 멀티페이지 애플리케이션 메인 엔트리
"""

import streamlit as st
import logging
import os
import time
from helper_dev_utils import get_auto_logger
from utils import st_style_page_margin_hidden, st_sidebar_show, st_div_divider
from utils.api_client import get_health_status
from utils.api_client import get_visitor_count
from datetime import datetime

logger = get_auto_logger(log_level=logging.DEBUG)

st_style_page_margin_hidden()

# 페이지 설정
st.set_page_config(
    page_title="영화 리뷰 감성 분석",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 백엔드 API URL
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
logger.info(f"API_BASE_URL: {API_BASE_URL}")

# 서버 준비 대기
placeholder = st.empty()
max_attempts = 300  # 최대 5분 (1초 * 300)
attempt = 0

while attempt < max_attempts:
    health = get_health_status(base_url=API_BASE_URL, timeout=(1, 2))

    if health is None:
        # 서버 연결 실패
        with placeholder.container():
            st.error("🔧 서버 점검 중입니다...")
            st.caption(f"서버 연결 시도 중... ({attempt + 1}/{max_attempts})")
    elif not health.get("ready", True):
        # 초기 동기화 진행 중
        initial_sync = health.get("initial_sync", {})
        current = initial_sync.get("current", 0)
        total = initial_sync.get("total", 1)
        sync_type = initial_sync.get("sync_type", "unknown")
        movies_collected = initial_sync.get("movies_collected", 0)
        posters_downloaded = initial_sync.get("posters_downloaded", 0)

        progress = current / total if total > 0 else 0

        with placeholder.container():
            st.info("⏳ 데이터베이스 초기화 중...")
            st.progress(progress)
            st.caption(
                f"영화 데이터: {movies_collected}개 수집 | "
                f"포스터: {posters_downloaded}개 다운로드 | "
                f"전체 진행: {current}/{total} ({sync_type})"
            )
    else:
        # 서버 준비 완료
        break

    time.sleep(1)
    attempt += 1

# 타임아웃
if attempt >= max_attempts:
    with placeholder.container():
        st.error("⚠️ 서버 연결 시간 초과")
        st.caption("서버가 응답하지 않습니다. 관리자에게 문의하세요.")
    st.stop()

# 메시지 제거
placeholder.empty()

# 페이지 정의
movie_list_page = st.Page(
    "pages/movie_list.py",
    title="영화 목록",
    icon="🎥",
    default=True,
)

management_page = st.Page(
    "pages/management.py",
    title="영화 관리",
    icon="🎬",
)

review_edit_page = st.Page(
    "pages/review_edit.py",
    title="리뷰 작성",
    icon="✍️",
)

review_list_page = st.Page(
    "pages/review_list.py",
    title="리뷰 목록",
    icon="📝",
)

# 네비게이션 설정
pg = st.navigation(
    {
        "메인": [movie_list_page, management_page, review_edit_page, review_list_page],
    }
)

# 사이드바 정보
with st.sidebar:
    st.write("##### 🎬 영화 리뷰 감성 분석")

    st_div_divider()

    st.info(
        """**프로젝트 정보**
        
개발기간 :

25.12.17 ~ 25.12.19

* TMDB 연동
* AI가 평점
    - 영화 정보 등록
    - 리뷰 작성 
    - 감성 분석
    - AI 기반 평점 시각화
"""
    )

    st_div_divider()

    visitor_stats = get_visitor_count(base_url=API_BASE_URL, timeout=(1, 2))

    if visitor_stats:
        server_start = visitor_stats.get("server_start_time")
        total_visitors = visitor_stats.get("total_visitors", 0)

        # 서버 시작 시간 포맷팅
        if server_start:
            try:
                start_dt = datetime.fromisoformat(server_start)
                formatted_time = start_dt.strftime("%Y-%m-%d %H:%M")
            except:
                formatted_time = "N/A"
        else:
            formatted_time = "N/A"

        st.info(f"**서버시작**:{formatted_time}")
        st.markdown(f"**방문자**: {total_visitors:,}명")

    st.divider()
    st.caption("Powered by FastAPI + Streamlit")

# 페이지 실행
pg.run()
