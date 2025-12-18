"""
리뷰 게시판 페이지 (board.py)
board_edit.py와 board_list.py를 통합하는 메인 페이지
"""

import sys
from pathlib import Path

# pages 디렉토리를 sys.path에 추가
pages_dir = Path(__file__).parent
if str(pages_dir) not in sys.path:
    sys.path.insert(0, str(pages_dir))

import streamlit as st
import requests
import os
from typing import List, Dict
import logging
from helper_dev_utils import get_auto_logger
from board_edit import ReviewEditManager
from board_list import ReviewListManager
from utils import *

logger = get_auto_logger(log_level=logging.DEBUG)

# 백엔드 API URL
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


class ReviewManager:
    """
    리뷰 관리 통합 클래스
    """

    def __init__(self):
        """ReviewManager 초기화"""
        self.api_url = API_BASE_URL
        self.edit_manager = ReviewEditManager()
        self.list_manager = ReviewListManager()

    def render(self):
        """리뷰 게시판 페이지 렌더링"""

        if st.button("###### 리뷰 게시판", key="toggle_sidebar", help="사이드바"):
            st_sidebar_show()

        # 영화 검색 섹션 (공통)
        self._render_movie_search()

        st_div_divider()

        # 탭 구성
        tab1, tab2 = st.tabs(["리뷰 작성", "리뷰 목록"])

        with tab1:
            self.edit_manager.render()

        with tab2:
            self.list_manager.render()

    def _render_movie_search(self):
        """영화 검색 섹션 렌더링 (공통)"""
        # URL 쿼리 파라미터에서 검색 조건 로드
        query_params = st.query_params

        # 세션 상태에 검색 조건 저장 (다른 페이지 갔다 와도 유지)
        if "board_search_params" not in st.session_state:
            st.session_state["board_search_params"] = {}

        # 초기 로드 플래그: URL 파라미터를 세션에 로드했는지 확인
        if "board_search_params_loaded" not in st.session_state:
            st.session_state["board_search_params_loaded"] = False

        # URL 쿼리 파라미터가 있고 아직 로드하지 않았으면 세션에 저장 (최초 1회만)
        if not st.session_state["board_search_params_loaded"] and any(
            key in query_params
            for key in [
                "title",
                "director",
                "genre",
                "date_from",
                "date_to",
                "tmdb_min",
                "tmdb_max",
            ]
        ):
            st.session_state["board_search_params"] = {
                "title": query_params.get("title", ""),
                "director": query_params.get("director", ""),
                "genre": query_params.get("genre", ""),
                "date_from": query_params.get("date_from", ""),
                "date_to": query_params.get("date_to", ""),
                "tmdb_min": query_params.get("tmdb_min", "0.0"),
                "tmdb_max": query_params.get("tmdb_max", "10.0"),
            }
            st.session_state["board_search_params_loaded"] = True

        # 영화 검색 섹션
        # st.write("**🔍 영화 검색**")

        with st.expander("🔍 영화 검색", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                search_title = st.text_input(
                    "제목",
                    value=st.session_state["board_search_params"].get("title", ""),
                    placeholder="예: Dark Knight",
                    help="영화 제목 검색 (부분 검색, 대소문자 무시)",
                )
                search_director = st.text_input(
                    "감독",
                    value=st.session_state["board_search_params"].get("director", ""),
                    placeholder="예: Nolan",
                    help="감독 이름 검색 (부분 검색, 대소문자 무시)",
                )
                search_genre = st.text_input(
                    "장르",
                    value=st.session_state["board_search_params"].get("genre", ""),
                    placeholder="예: Action",
                    help="장르 검색 (부분 검색, 대소문자 무시)",
                )

            with col2:
                col2_1, col2_2 = st.columns(2)
                with col2_1:
                    search_date_from = st.text_input(
                        "개봉일 시작",
                        value=st.session_state["board_search_params"].get(
                            "date_from", ""
                        ),
                        placeholder="YYYY-MM-DD",
                        help="예: 2020-01-01",
                    )
                with col2_2:
                    search_date_to = st.text_input(
                        "개봉일 종료",
                        value=st.session_state["board_search_params"].get(
                            "date_to", ""
                        ),
                        placeholder="YYYY-MM-DD",
                        help="예: 2023-12-31",
                    )

                col2_3, col2_4 = st.columns(2)
                with col2_3:
                    search_tmdb_min = st.number_input(
                        "최소 TMDB 평점",
                        min_value=0.0,
                        max_value=10.0,
                        value=float(
                            st.session_state["board_search_params"].get(
                                "tmdb_min", "0.0"
                            )
                        ),
                        step=0.5,
                        help="0~10점",
                    )
                with col2_4:
                    search_tmdb_max = st.number_input(
                        "최대 TMDB 평점",
                        min_value=0.0,
                        max_value=10.0,
                        value=float(
                            st.session_state["board_search_params"].get(
                                "tmdb_max", "10.0"
                            )
                        ),
                        step=0.5,
                        help="0~10점",
                    )

            col3_1, col3_2, col3_3 = st.columns([2, 2, 6])
            with col3_1:
                search_button = st.button("🔍 검색", type="primary", width="content")
            with col3_2:
                reset_button = st.button("🔄 초기화", width="content")

        # 검색 또는 초기화 버튼 클릭 시 처리
        if search_button or reset_button:
            if reset_button:
                # 초기화: URL 쿼리 파라미터 및 세션 상태 제거
                st.query_params.clear()
                st.session_state["board_search_params"] = {}
                st.session_state["board_search_params_loaded"] = False
                # 초기화: 최신 영화 10개 로드
                st.session_state["searched_movies"] = self._get_recent_movies()
                st.session_state["search_performed"] = False
                # 리뷰 목록 초기화
                self._reset_review_list()
                st.rerun()
            else:
                # 검색 실행
                search_params = {}
                new_query_params = {}

                if search_title:
                    search_params["title"] = search_title
                    new_query_params["title"] = search_title
                if search_director:
                    search_params["director"] = search_director
                    new_query_params["director"] = search_director
                if search_genre:
                    search_params["genre"] = search_genre
                    new_query_params["genre"] = search_genre
                if search_date_from:
                    search_params["release_date_from"] = search_date_from
                    new_query_params["date_from"] = search_date_from
                if search_date_to:
                    search_params["release_date_to"] = search_date_to
                    new_query_params["date_to"] = search_date_to
                if search_tmdb_min > 0:
                    search_params["tmdb_rating_min"] = search_tmdb_min
                    new_query_params["tmdb_min"] = str(search_tmdb_min)
                if search_tmdb_max < 10:
                    search_params["tmdb_rating_max"] = search_tmdb_max
                    new_query_params["tmdb_max"] = str(search_tmdb_max)

                # 세션 상태에 검색 조건 저장
                st.session_state["board_search_params"] = {
                    "title": search_title,
                    "director": search_director,
                    "genre": search_genre,
                    "date_from": search_date_from,
                    "date_to": search_date_to,
                    "tmdb_min": str(search_tmdb_min),
                    "tmdb_max": str(search_tmdb_max),
                }

                # URL 쿼리 파라미터 업데이트
                st.query_params.update(new_query_params)

                # 검색 API 호출
                searched_movies = self._search_movies(search_params)
                st.session_state["searched_movies"] = searched_movies
                st.session_state["search_performed"] = True

                # 리뷰 목록 초기화 (새로운 영화 목록에 맞춰)
                self._reset_review_list()

                st.session_state["board_search_params"] = {
                    "title": search_title,
                    "director": search_director,
                    "genre": search_genre,
                    "date_from": search_date_from,
                    "date_to": search_date_to,
                    "tmdb_min": str(search_tmdb_min),
                    "tmdb_max": str(search_tmdb_max),
                }

                # URL 쿼리 파라미터 업데이트
                st.query_params.update(new_query_params)

                # 검색 API 호출
                searched_movies = self._search_movies(search_params)
                st.session_state["searched_movies"] = searched_movies
                st.session_state["search_performed"] = True

                # 검색 결과 정보 표시 됨으로 여기에서는 생략
                # 검색 결과 메시지
                # if not searched_movies:
                # st.success(f"✅ {len(searched_movies)}개의 영화를 찾았습니다.")
                # else:
                #    st.warning("⚠️ 검색 결과가 없습니다. 검색 조건을 변경해보세요.")

        # 검색된 영화 목록 또는 최신 영화 10개 목록 사용
        if "searched_movies" not in st.session_state:
            # URL 쿼리 파라미터가 있으면 검색 실행
            if any(
                key in query_params
                for key in [
                    "title",
                    "director",
                    "genre",
                    "date_from",
                    "date_to",
                    "tmdb_min",
                    "tmdb_max",
                ]
            ):
                search_params = {}
                if "title" in query_params:
                    search_params["title"] = query_params["title"]
                if "director" in query_params:
                    search_params["director"] = query_params["director"]
                if "genre" in query_params:
                    search_params["genre"] = query_params["genre"]
                if "date_from" in query_params:
                    search_params["release_date_from"] = query_params["date_from"]
                if "date_to" in query_params:
                    search_params["release_date_to"] = query_params["date_to"]
                if "tmdb_min" in query_params:
                    search_params["tmdb_rating_min"] = float(query_params["tmdb_min"])
                if "tmdb_max" in query_params:
                    search_params["tmdb_rating_max"] = float(query_params["tmdb_max"])

                st.session_state["searched_movies"] = self._search_movies(search_params)
                st.session_state["search_performed"] = True
            else:
                st.session_state["searched_movies"] = self._get_recent_movies()
                st.session_state["search_performed"] = False

        movies = st.session_state.get("searched_movies", [])

        # 검색 결과 정보 표시
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

    def _reset_review_list(self):
        """리뷰 목록 초기화"""
        if "loaded_reviews" in st.session_state:
            st.session_state["loaded_reviews"] = []
        if "reviews_current_page" in st.session_state:
            st.session_state["reviews_current_page"] = 1
        if "reviews_has_more" in st.session_state:
            st.session_state["reviews_has_more"] = True

    def _get_recent_movies(self) -> List[Dict]:
        """최신 영화 10개 가져오기 (개봉일 기준 내림차순)"""
        try:
            search_params = {
                "page_size": 10,
                "page": 1,
                "sort_by": "release_date",
                "sort_order": "desc",
            }

            response = requests.get(
                f"{self.api_url}/movies/search", params=search_params
            )

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

            response = requests.get(
                f"{self.api_url}/movies/search", params=search_params
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("movies", [])
            else:
                logger.error(f"Movie search failed: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to search movies: {str(e)}")

        return []


# 페이지 실행
if __name__ == "__main__":
    manager = ReviewManager()
    manager.render()
