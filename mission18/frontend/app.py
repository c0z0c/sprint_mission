"""
Streamlit 멀티페이지 애플리케이션 메인 엔트리
"""

import streamlit as st
import logging
from helper_dev_utils import get_auto_logger
from utils import (
    st_style_page_margin_hidden,
    st_sidebar_show,
)

logger = get_auto_logger(log_level=logging.DEBUG)

st_style_page_margin_hidden()

# 페이지 설정
st.set_page_config(
    page_title="영화 리뷰 감성 분석",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

board_page = st.Page(
    "pages/board.py",
    title="리뷰 게시판",
    icon="💬",
)

# 네비게이션 설정
pg = st.navigation(
    {
        "메인": [movie_list_page, management_page, board_page],
    }
)

# 사이드바 정보
with st.sidebar:
    st.title("🎬 영화 리뷰 감성 분석")
    st.divider()
    st.info(
        """
        **프로젝트 정보**
        
        이 애플리케이션은 영화 리뷰의 감성을 
        AI가 자동으로 분석하여 평점을 제공합니다.
        
        - 영화 정보 등록
        - 리뷰 작성 및 감성 분석
        - AI 기반 평점 시각화
        """
    )
    st.divider()
    st.caption("Powered by FastAPI + Streamlit")

# 페이지 실행
pg.run()
