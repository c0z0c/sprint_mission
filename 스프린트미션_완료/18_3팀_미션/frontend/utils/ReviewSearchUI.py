"""
검색 UI 공통 모듈
영화 검색 및 리뷰 검색 UI 컴포넌트
"""

import streamlit as st
from typing import Dict, Optional, Tuple
import logging
from helper_dev_utils import get_auto_logger

logger = get_auto_logger(log_level=logging.DEBUG)


class ReviewSearchUI:
    """
    리뷰 검색 UI 컴포넌트
    """

    def render(self) -> Tuple[bool, Optional[Dict]]:
        """
        리뷰 검색 UI 렌더링

        Returns:
            (search_triggered, filters): 검색 실행 여부와 필터 딕셔너리
            - search_triggered: True(검색 버튼 클릭), False(초기화 버튼 클릭), None(버튼 미클릭)
            - filters: 검색 필터 딕셔너리 (None 값 제거됨)
        """
        # 세션 상태 초기화
        if "review_search_params" not in st.session_state:
            st.session_state["review_search_params"] = {}

        with st.expander("🔍 리뷰 검색", expanded=False):
            # 작성자 및 영화 제목 검색
            col1, col2 = st.columns(2)
            with col1:
                search_author = st.text_input(
                    "작성자",
                    value=st.session_state["review_search_params"].get("author", ""),
                    key="search_author",
                    placeholder="예: 홍길동",
                )
            with col2:
                search_movie_title = st.text_input(
                    "영화 제목",
                    value=st.session_state["review_search_params"].get(
                        "movie_title", ""
                    ),
                    key="search_movie_title",
                    placeholder="예: 인셉션",
                )

            # 리뷰 내용 및 감성 검색
            col1, col2 = st.columns(2)
            with col1:
                search_content = st.text_input(
                    "리뷰 내용",
                    value=st.session_state["review_search_params"].get("content", ""),
                    key="search_content",
                    placeholder="키워드 검색",
                )
            with col2:
                search_sentiment = st.selectbox(
                    "감성",
                    options=["all", "positive", "negative"],
                    format_func=lambda x: {
                        "all": "전체",
                        "positive": "긍정",
                        "negative": "부정",
                    }[x],
                    index=["all", "positive", "negative"].index(
                        st.session_state["review_search_params"].get("sentiment", "all")
                    ),
                    key="search_sentiment",
                )

            # 작성일 범위
            col1, col2 = st.columns(2)
            with col1:
                search_created_from = st.text_input(
                    "작성일 시작",
                    value=st.session_state["review_search_params"].get(
                        "created_from", ""
                    ),
                    key="search_created_from",
                    placeholder="YYYY-MM-DD",
                )
            with col2:
                search_created_to = st.text_input(
                    "작성일 종료",
                    value=st.session_state["review_search_params"].get(
                        "created_to", ""
                    ),
                    key="search_created_to",
                    placeholder="YYYY-MM-DD",
                )

            # 정렬 옵션
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                sort_by = st.selectbox(
                    "정렬 기준",
                    options=["created_at", "author"],
                    format_func=lambda x: {"created_at": "작성일", "author": "작성자"}[
                        x
                    ],
                    index=["created_at", "author"].index(
                        st.session_state["review_search_params"].get(
                            "sort_by", "created_at"
                        )
                    ),
                    key="review_sort_by",
                )
            with col2:
                sort_order = st.selectbox(
                    "정렬 방향",
                    options=["desc", "asc"],
                    format_func=lambda x: "내림차순" if x == "desc" else "오름차순",
                    index=["desc", "asc"].index(
                        st.session_state["review_search_params"].get(
                            "sort_order", "desc"
                        )
                    ),
                    key="review_sort_order",
                )

            # 검색 및 초기화 버튼
            col1, col2 = st.columns(2)
            with col1:
                search_button = st.button(
                    "🔍 검색",
                    type="primary",
                    key="btn_review_search",
                    use_container_width=True,
                )
            with col2:
                reset_button = st.button(
                    "🔄 초기화",
                    type="secondary",
                    key="btn_review_reset",
                    use_container_width=True,
                )

        # 버튼 클릭 처리
        if reset_button:
            # 초기화
            st.session_state["review_search_params"] = {}

            widget_keys = [
                "search_author",
                "search_movie_title",
                "search_content",
                "search_sentiment",
                "search_created_from",
                "search_created_to",
                "review_sort_by",
                "review_sort_order",
            ]
            for key in widget_keys:
                if key in st.session_state:
                    del st.session_state[key]

            return False, None

        if search_button:
            # 검색 필터 구성
            filters = {
                "author": search_author if search_author.strip() else None,
                "movie_title": (
                    search_movie_title if search_movie_title.strip() else None
                ),
                "content": search_content if search_content.strip() else None,
                "sentiment": search_sentiment if search_sentiment != "all" else None,
                "created_from": (
                    search_created_from if search_created_from.strip() else None
                ),
                "created_to": search_created_to if search_created_to.strip() else None,
                "sort_by": sort_by,
                "sort_order": sort_order,
            }

            # None 값 제거
            filters = {k: v for k, v in filters.items() if v is not None}

            # 세션 상태 업데이트
            st.session_state["review_search_params"] = {
                "author": search_author,
                "movie_title": search_movie_title,
                "content": search_content,
                "sentiment": search_sentiment,
                "created_from": search_created_from,
                "created_to": search_created_to,
                "sort_by": sort_by,
                "sort_order": sort_order,
            }

            return True, filters

        return None, None
