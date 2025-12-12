# -*- coding: utf-8 -*-
"""Streamlit 유틸리티 함수 모음

이 모듈은 Streamlit 앱의 UI 스타일을 커스터마이징하는 유틸리티 함수들을 제공합니다.

주요 기능:
    - 페이지 상단 여백 제거
    - 컴팩트한 구분선 생성
    - 사이드바 토글 유지
"""

import logging

import streamlit as st
from helper_dev_utils import get_auto_logger

logger = get_auto_logger(log_level=logging.DEBUG)


def hidden_page_top_margin() -> None:
    """상단 여백을 제거하면서 사이드바 토글 버튼은 유지합니다.

    Streamlit의 기본 헤더/툴바를 숨기고 페이지 상단 여백을 제거하여
    더 많은 화면 공간을 확보합니다. 단, 사이드바 토글 버튼은 유지합니다.

    적용되는 스타일:
        - Streamlit 헤더/툴바 숨김
        - 메인 컨테이너 상단 여백 제거
        - 텍스트 요소 여백 최소화
        - 사이드바 토글 버튼 유지
    """
    hide_deploy = []

    # 스타일 태그 시작
    hide_deploy.append("<style>")

    # Streamlit 기본 헤더/툴바 숨김
    hide_deploy.append('[data-testid="stHeader"] { display: none !important; }')
    hide_deploy.append('[data-testid="stToolbar"] { display: none !important; }')
    hide_deploy.append(".stAppHeader { display: none !important; }")

    # 메인 컨테이너 여백 제거
    hide_deploy.append(".stMainBlockContainer { padding-top: 0 !important; }")
    hide_deploy.append(
        ".stVerticalBlock:first-of-type { margin-top: 0 !important; padding-top: 0 !important; }"
    )
    hide_deploy.append(
        ".st-emotion-cache-6c7yup { margin-top: 0 !important; padding-top: 0 !important; }"
    )

    # 텍스트 요소 여백 제거
    hide_deploy.append(
        "h1, h2, h3, h4, h5, h6, p, div { margin-bottom: 0 !important; margin-top: 0 !important; padding-bottom: 0 !important; padding-top: 0 !important; }"
    )

    # 컴팩트 구분선 숨김
    hide_deploy.append(
        "hr.compact { display: none !important; margin: 0 !important; padding: 0 !important; }"
    )

    # 사이드바 토글 버튼 영역 유지
    hide_deploy.append('[data-testid="stSidebarNav"] { display: block !important; }')

    # 스타일 태그 종료
    hide_deploy.append("</style>")

    # HTML 구분선 (선택적)
    hide_deploy.append('<hr class="compact">')

    # 리스트를 개행으로 결합하여 마크다운 렌더링
    hide_deploy_str = "\n".join(hide_deploy)
    st.markdown(hide_deploy_str, unsafe_allow_html=True)


def minimal_divider() -> None:
    """여백 최소화 가로선을 렌더링합니다.

    기본 st.markdown("---")의 과도한 여백을 제거하고
    얇은 회색 구분선만 표시합니다.

    사용 예:
        minimal_divider()  # 1px 회색 구분선 표시
    """
    st.markdown(
        '<div style="height: 1px; background-color: #ddd; margin: 0; padding: 0;"></div>',
        unsafe_allow_html=True,
    )
