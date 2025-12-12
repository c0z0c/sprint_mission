import streamlit as st
import logging
from helper_dev_utils import get_auto_logger

logger = get_auto_logger(log_level=logging.DEBUG)


def hidden_page_top_margin() -> None:
    """상단 여백을 제거하면서 사이드바 토글 버튼은 유지합니다."""
    hide_deploy = """
    <style>
    [data-testid="stHeader"] {
        display: none !important;
    }
    [data-testid="stToolbar"] {
        display: none !important;
    }
    .stAppHeader {
        display: none !important;
    }
    .stMainBlockContainer {
        padding-top: 0 !important;
    }
    .stVerticalBlock:first-of-type {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    .st-emotion-cache-6c7yup {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    h1, h2, h3, h4, h5, h6, p, div {
        margin-bottom: 0 !important;
        margin-top: 0 !important;
        padding-bottom: 0 !important;
        padding-top: 0 !important;
    }
    hr.compact {
        display: none !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    /* 사이드바 토글 버튼 영역 유지 */
    [data-testid="stSidebarNav"] {
        display: block !important;
    }
    </style>
    <hr class="compact">
    """
    st.markdown(hide_deploy, unsafe_allow_html=True)


def custom_sidebar_toggle() -> None:
    """커스텀 사이드바 토글 버튼을 추가합니다.

    상단 여백 제거로 숨겨진 기본 토글 버튼을 대신하는 커스텀 버튼을
    페이지 좌상단에 표시합니다.
    """
    toggle_html = """
    <style>
    .custom-sidebar-toggle {
        position: fixed;
        top: 10px;
        left: 10px;
        z-index: 10000;
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 8px 12px;
        cursor: pointer;
        font-size: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 40px;
        height: 40px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .custom-sidebar-toggle:hover {
        background: #f5f5f5;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    </style>
    <div class="custom-sidebar-toggle" onclick="document.querySelector('[data-testid=stSidebarNav]').click()" title="사이드바 토글">
        ☰
    </div>
    """
    st.markdown(toggle_html, unsafe_allow_html=True)


def minimal_divider() -> None:
    """여백 최소화 가로선을 렌더링합니다.

    기본 st.markdown("---")의 과도한 여백을 제거합니다.
    """
    st.markdown(
        '<div style="height: 1px; background-color: #ddd; margin: 0; padding: 0;"></div>',
        unsafe_allow_html=True,
    )
