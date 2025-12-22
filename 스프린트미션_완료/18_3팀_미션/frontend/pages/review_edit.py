"""
리뷰 작성 페이지 (review_edit.py)
독립적인 리뷰 작성 페이지
"""

import streamlit as st
import requests
import plotly.graph_objects as go
from typing import List, Dict, Optional
from datetime import datetime
import os
import random
import logging
from helper_dev_utils import get_auto_logger
from utils import *
from utils.search_ui import MovieSearchUI
from utils.search_ui import MovieSearchUI

logger = get_auto_logger(log_level=logging.DEBUG)

# 백엔드 API URL
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


class ReviewEditManager:
    """
    리뷰 작성 관리 클래스
    """

    def __init__(self):
        """ReviewEditManager 초기화"""
        self.api_url = API_BASE_URL

    def render(self):
        """리뷰 작성 페이지 렌더링 (독립 페이지)"""

        if st.button(
            "###### ✍️ 리뷰 작성", key="toggle_sidebar_review_edit", help="사이드바"
        ):
            st_sidebar_show()

        # 영화 검색 섹션
        self._render_movie_search()

        st_div_divider()

        # 리뷰 작성 폼
        self._render_review_form()

    def _render_review_form(self):
        """리뷰 작성 폼 렌더링"""
        movies = st.session_state.get("searched_movies", [])

        if not movies:
            st.warning("영화를 먼저 검색해주세요.")
            return

        # 영화 선택 (폼 외부에서 처리하여 AI 평점과 동기화)
        movie_options = {
            f"{m['title']} ({m.get('release_date') or '개봉일 미정'}) - {m.get('director', '감독 미상')}": m[
                "tmdb_id"
            ]
            for m in movies
        }
        movie_names = list(movie_options.keys())

        # 세션 상태 초기화: 첫 번째 영화를 기본값으로 설정
        if (
            "selected_movie_for_review" not in st.session_state
            or st.session_state["selected_movie_for_review"] not in movie_names
        ):
            st.session_state["selected_movie_for_review"] = (
                movie_names[0] if movie_names else None
            )

        if not movie_names:
            return

        # 영화 선택 드롭다운
        with st.container():
            clos = st.columns([1, 7, 2])
            with clos[0]:
                st_label("영화선택")
            with clos[1]:
                selected_movie = st.selectbox(
                    "영화 선택 *",
                    options=movie_names,
                    key="selected_movie_for_review",
                    help="검색된 영화 중에서 선택하세요",
                    label_visibility="collapsed",
                )
            with clos[2]:
                # 랜덤값 설정 버튼
                if st.button(
                    "🎲 랜덤 리뷰 생성", key="set_random_text", use_container_width=True
                ):
                    from utils.random_text import random_name, random_review_text

                    st.session_state["random_author"] = random_name()
                    st.session_state["random_content"] = random_review_text()
                    st.rerun()

        # 선택된 영화 TMDB ID 가져오기
        selected_tmdb_id = movie_options[selected_movie]

        with st.form("review_form"):
            # 작성자 이름
            cols1 = st.columns([1, 9])
            with cols1[0]:
                st_label("작성자")

            with cols1[1]:
                author = st.text_input(
                    "작성자 이름 *",
                    placeholder="예: 홍길동",
                    value=st.session_state.get("random_author", ""),
                    label_visibility="collapsed",
                )

            cols2 = st.columns([1, 9])
            with cols2[0]:
                st_label("리뷰<br/>내용")
            with cols2[1]:
                content = st.text_area(
                    "리뷰 내용 *",
                    placeholder="영화에 대한 리뷰를 작성해주세요...",
                    height=70,
                    value=st.session_state.get("random_content", ""),
                    label_visibility="collapsed",
                )

            cols3 = st.columns([1, 9])
            with cols3[0]:
                st_label("작성일")
            with cols3[1]:
                cols4 = st.columns([3, 2])
                with cols4[0]:
                    cols5 = st.columns([1, 1])
                    with cols5[0]:
                        created_datetime = st.text_input(
                            "작성 시간",
                            value=datetime.now().strftime("%Y/%m/%d %H:%M"),
                            help="리뷰 작성 시간",
                            key="review_created_datetime",
                            label_visibility="collapsed",
                        )
                    with cols5[1]:
                        updated_datetime = st.text_input(
                            "수정 시간",
                            value=datetime.now().strftime("%Y/%m/%d %H:%M"),
                            help="리뷰 수정 시간",
                            key="review_updated_datetime",
                            label_visibility="collapsed",
                        )
                with cols4[1]:
                    submitted = st.form_submit_button(
                        "리뷰 등록", use_container_width=True
                    )

            if submitted:
                # 세션 상태 초기화
                if "random_author" in st.session_state:
                    del st.session_state["random_author"]
                if "random_content" in st.session_state:
                    del st.session_state["random_content"]

                if not author or not content:
                    st.error("작성자 이름과 리뷰 내용은 필수 입력 항목입니다.")
                else:
                    # datetime 객체 생성
                    created_datetime = datetime.strptime(
                        created_datetime, "%Y/%m/%d %H:%M"
                    )
                    updated_datetime = datetime.strptime(
                        updated_datetime, "%Y/%m/%d %H:%M"
                    )
                    self._register_review(
                        selected_tmdb_id,
                        author,
                        content,
                        created_datetime,
                        updated_datetime,
                    )

        # 선택된 영화의 AI 평점 표시
        st_div_divider()

        self._render_movie_rating_section(selected_tmdb_id, movies)

    def _render_movie_rating_section(self, movie_id: int, movies: List[Dict]):
        """영화 AI 평점 섹션 렌더링"""
        st.write("##### 영화 AI 평점")

        # 선택된 영화의 평점 데이터 캐싱
        cache_key = f"cached_rating_{movie_id}"
        if cache_key not in st.session_state:
            st.session_state[cache_key] = self._get_movie_rating(movie_id)

        rating_data = st.session_state[cache_key]

        if rating_data:
            col1, col2 = st.columns([2, 1])

            with col1:
                # Gauge Chart
                self._render_gauge_chart(rating_data)

            with col2:
                # 통계 정보
                st.metric("총 리뷰 수", rating_data["total_reviews"])
                st.metric(
                    "긍정 리뷰",
                    rating_data["positive_reviews"],
                    delta_color="normal",
                )
                st.metric(
                    "부정 리뷰",
                    rating_data["negative_reviews"],
                    delta_color="inverse",
                )
                st.metric("AI 평점", f"{rating_data['ai_rating']}/10.0")
        else:
            st.info("해당 영화에 대한 리뷰가 아직 없습니다.")

    def _render_gauge_chart(self, rating_data: Dict):
        """Gauge Chart 렌더링"""
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=rating_data["ai_rating"],
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "AI 평점 (10점 만점)", "font": {"size": 24}},
                delta={"reference": 5.0, "increasing": {"color": "green"}},
                gauge={
                    "axis": {
                        "range": [None, 10],
                        "tickwidth": 1,
                        "tickcolor": "darkblue",
                    },
                    "bar": {"color": "darkblue"},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "bordercolor": "gray",
                    "steps": [
                        {"range": [0, 2], "color": "#ffcccc"},
                        {"range": [2, 4], "color": "#ffddcc"},
                        {"range": [4, 6], "color": "#ffffcc"},
                        {"range": [6, 8], "color": "#ddffcc"},
                        {"range": [8, 10], "color": "#ccffcc"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 9.0,
                    },
                },
            )
        )

        fig.update_layout(
            paper_bgcolor="white",
            font={"color": "darkblue", "family": "Arial"},
            height=400,
        )

        st.plotly_chart(fig, width="content")

    def _get_movie_rating(self, tmdb_id: int) -> Optional[Dict]:
        """영화 평점 조회"""
        try:
            response = api_get(f"{self.api_url}/reviews/movie/{tmdb_id}/rating")
            if response.status_code == 200:
                return response.json()
        except requests.exceptions.RequestException:
            pass
        return None

    def _register_review(
        self,
        tmdb_id: int,
        author: str,
        content: str,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
    ):
        """리뷰 등록"""

        review_data = {
            "tmdb_id": tmdb_id,
            "author": author,
            "content": content,
            "created_at": created_at.isoformat() if created_at else None,
            "updated_at": updated_at.isoformat() if updated_at else None,
        }

        try:
            response = api_post(f"{self.api_url}/reviews/", json=review_data)

            if response.status_code == 201:
                review = response.json()
                is_positive = review.get("is_positive") == 1
                sentiment = "긍정" if is_positive else "부정"
                sentiment_color = "green" if is_positive else "red"

                # 생성 날짜 포맷팅
                created_at_str = ""
                if review.get("created_at"):
                    try:
                        dt = datetime.fromisoformat(
                            review["created_at"].replace("Z", "+00:00")
                        )
                        created_at_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        created_at_str = review["created_at"]

                # 캐시 무효화
                if "loaded_reviews" in st.session_state:
                    st.session_state["loaded_reviews"] = []
                    st.session_state["reviews_current_page"] = 1
                    st.session_state["reviews_has_more"] = True
                # 평점 캐시 무효화
                cache_key = f"cached_rating_{tmdb_id}"
                if cache_key in st.session_state:
                    del st.session_state[cache_key]

                success_msg = f"✅ 리뷰가 등록되었습니다!\n\n"
                success_msg += f"**작성자:** {author}\n\n"
                success_msg += f"**리뷰 내용:**\n{content}\n\n"
                success_msg += f"- AI 분석 결과: :{sentiment_color}[**{sentiment}**]\n"
                if created_at_str:
                    success_msg += f"- 작성일시: {created_at_str}"
                st.success(success_msg)
            elif response.status_code == 409:
                st.warning("⚠️ 중복 입력: 이미 동일한 리뷰가 등록되어 있습니다.")
            else:
                error_detail = response.json().get("detail", "알 수 없는 오류")
                st.error(f"리뷰 등록 실패: {error_detail}")
        except requests.exceptions.RequestException as e:
            st.error(f"API 연결 오류: {str(e)}")

    def _render_movie_search(self):
        """영화 검색 섹션 렌더링"""
        # 영화 검색 UI
        search_ui = MovieSearchUI(show_ai_rating=False, show_sort_options=True)
        search_triggered, filters = search_ui.render()

        if search_triggered is False:
            # 초기화
            st.session_state["searched_movies"] = self._get_recent_movies()
            st.session_state["search_performed"] = False
            st.rerun()
        elif search_triggered is True:
            # 검색
            search_params = {}
            if "title" in filters:
                search_params["title"] = filters["title"]
            if "director" in filters:
                search_params["director"] = filters["director"]
            if "genre" in filters:
                search_params["genre"] = filters["genre"]
            if "release_date_from" in filters:
                search_params["release_date_from"] = filters["release_date_from"]
            if "release_date_to" in filters:
                search_params["release_date_to"] = filters["release_date_to"]
            if "tmdb_rating_min" in filters:
                search_params["tmdb_rating_min"] = filters["tmdb_rating_min"]
            if "tmdb_rating_max" in filters:
                search_params["tmdb_rating_max"] = filters["tmdb_rating_max"]
            if "sort_by" in filters:
                search_params["sort_by"] = filters["sort_by"]
            if "sort_order" in filters:
                search_params["sort_order"] = filters["sort_order"]

            searched_movies = self._search_movies(search_params)
            st.session_state["searched_movies"] = searched_movies
            st.session_state["search_performed"] = True
            st.rerun()

        # 초기 로드: searched_movies가 없으면 최신 영화 로드
        if "searched_movies" not in st.session_state:
            st.session_state["searched_movies"] = self._get_recent_movies()
            st.session_state["search_performed"] = False

        # 검색 결과 정보 표시
        movies = st.session_state.get("searched_movies", [])
        if movies:
            if st.session_state.get("search_performed", False):
                st.info(f"📋 검색된 영화: {len(movies)}개")
            else:
                st.info(f"📋 최신 영화: {len(movies)}개")
        else:
            if st.session_state.get("search_performed", False):
                st.warning(
                    "검색 결과가 없습니다. 검색 조건을 변경하거나 초기화 버튼을 눌러주세요."
                )
            else:
                st.warning("등록된 영화가 없습니다. 먼저 영화를 등록해주세요.")

    def _get_recent_movies(self) -> List[Dict]:
        """최신 영화 10개 가져오기 (개봉일 기준 내림차순)"""
        try:
            search_params = {
                "page_size": 10,
                "page": 1,
                "sort_by": "release_date",
                "sort_order": "desc",
            }

            response = api_get(f"{self.api_url}/movies/search", params=search_params)

            if response.status_code == 200:
                data = response.json()
                return data.get("movies", [])
            else:
                logger.error(f"Failed to get recent movies: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get recent movies: {str(e)}")

        return []

    def _search_movies(self, search_params: dict) -> List[Dict]:
        """영화 검색"""
        try:
            search_params["page_size"] = 100
            search_params["page"] = 1

            response = api_get(f"{self.api_url}/movies/search", params=search_params)

            if response.status_code == 200:
                data = response.json()
                return data.get("movies", [])
            else:
                logger.error(f"Movie search failed: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to search movies: {str(e)}")

        return []


# 페이지 실행
manager = ReviewEditManager()
manager.render()
