"""
영화 수정/삭제 모듈 (movie_edit.py)
"""

import streamlit as st
import requests
from typing import List, Dict, Optional
import os
import logging
from helper_dev_utils import get_auto_logger
from utils import *

logger = get_auto_logger(log_level=logging.DEBUG)

# 백엔드 API URL
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


class MovieEditManager:
    """
    영화 수정/삭제 관리 클래스
    """

    def __init__(self):
        """MovieEditManager 초기화"""
        self.api_url = API_BASE_URL

    def render(self):
        """영화 수정/삭제 UI 렌더링"""
        st.write("###### 영화 검색 및 수정/삭제")

        # 세션 상태 초기화
        if "loaded_movies_edit" not in st.session_state:
            st.session_state["loaded_movies_edit"] = []
            st.session_state["edit_current_page"] = 1
            st.session_state["edit_has_more"] = True
            st.session_state["edit_page_size"] = 10  # 10개씩 로드

        # 수정 중인 영화 ID 추적
        if "editing_movie_id" not in st.session_state:
            st.session_state["editing_movie_id"] = None

        # 삭제 확인 중인 영화 ID 추적
        if "deleting_movie_id" not in st.session_state:
            st.session_state["deleting_movie_id"] = None

        # 검색 모드 여부 (검색어가 있으면 true)
        if "edit_search_mode" not in st.session_state:
            st.session_state["edit_search_mode"] = False

        # 검색 UI
        self._render_search_ui()

        st_div_divider()

        # 초기 로드: 검색 모드가 아니고 영화가 없으면 자동 로드
        if (
            not st.session_state["edit_search_mode"]
            and not st.session_state["loaded_movies_edit"]
            and st.session_state["edit_has_more"]
        ):
            self._load_more_movies()

        # 영화 목록 표시
        if st.session_state["loaded_movies_edit"]:
            self._render_movie_list()
        else:
            if st.session_state["edit_search_mode"]:
                st.info("🔍 검색 결과가 없습니다.")
            else:
                st.info("📭 등록된 영화가 없습니다.")

    def _render_search_ui(self):
        """영화 검색 UI 렌더링"""

        cols1 = st.columns([4, 1])
        with cols1[0]:
            cols2 = st.columns([1, 4, 1, 4])
            with cols2[0]:
                st_label("제목")
            with cols2[1]:
                search_title = st.text_input(
                    "영화 제목으로 검색",
                    key="search_title_edit",
                    placeholder="예: 인셉션",
                    label_visibility="collapsed",
                )
            with cols2[2]:
                st_label("TMDB ID")
            with cols2[3]:
                search_tmdb_id = st.text_input(
                    "TMDB ID로 검색",
                    key="search_tmdb_id_edit",
                    placeholder="예: 27205",
                    label_visibility="collapsed",
                )
        with cols1[1]:
            cols2 = st.columns([1, 1])
            with cols2[0]:
                if st.button("🔍 검색", key="search_movies_edit", type="primary"):
                    self._search_movies(search_title, search_tmdb_id)

            with cols2[1]:
                if st.button("🔄 초기화", key="reset_search_edit"):
                    st.session_state["loaded_movies_edit"] = []
                    st.session_state["edit_current_page"] = 1
                    st.session_state["edit_has_more"] = True
                    st.session_state["editing_movie_id"] = None
                    st.session_state["deleting_movie_id"] = None
                    st.session_state["edit_search_mode"] = False
                    # 검색 필드 초기화 - 위젯 key를 삭제하여 초기화
                    if "search_title_edit" in st.session_state:
                        del st.session_state["search_title_edit"]
                    if "search_tmdb_id_edit" in st.session_state:
                        del st.session_state["search_tmdb_id_edit"]
                    st.rerun()

    def _search_movies(self, title: str = "", tmdb_id: str = ""):
        """영화 검색"""
        try:
            # 검색어가 없으면 무시
            if not title.strip() and not tmdb_id.strip():
                st.warning("⚠️ 검색어를 입력해주세요.")
                return

            params = {
                "page": 1,  # 검색 시 항상 첫 페이지부터
                "page_size": st.session_state["edit_page_size"],
            }

            if title.strip():
                params["title"] = title.strip()

            if tmdb_id.strip():
                try:
                    params["tmdb_id"] = int(tmdb_id.strip())
                except ValueError:
                    st.error("❌ TMDB ID는 숫자여야 합니다.")
                    return

            response = requests.get(f"{self.api_url}/movies/search", params=params)

            if response.status_code == 200:
                data = response.json()
                movies = data.get("movies", [])
                total_pages = data.get("total_pages", 0)

                # 검색 모드로 전환
                st.session_state["edit_search_mode"] = True
                st.session_state["loaded_movies_edit"] = movies
                st.session_state["edit_current_page"] = 1
                st.session_state["edit_has_more"] = 1 < total_pages

                if not movies:
                    st.warning("🔍 검색 결과가 없습니다.")
                else:
                    st.success(f"✅ {len(movies)}개의 영화를 찾았습니다.")
                    st.rerun()
            else:
                st.error("❌ 영화 검색 실패")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ API 연결 오류: {str(e)}")

    def _load_more_movies(self):
        """다음 페이지의 영화를 로드하여 누적 목록에 추가 (최신순)"""
        try:
            params = {
                "page": st.session_state["edit_current_page"],
                "page_size": st.session_state["edit_page_size"],
            }

            # 검색 모드인 경우 검색 조건 추가
            if st.session_state["edit_search_mode"]:
                title = st.session_state.get("search_title_edit", "")
                tmdb_id = st.session_state.get("search_tmdb_id_edit", "")

                if title.strip():
                    params["title"] = title.strip()

                if tmdb_id.strip():
                    try:
                        params["tmdb_id"] = int(tmdb_id.strip())
                    except ValueError:
                        pass

                response = requests.get(f"{self.api_url}/movies/search", params=params)
            else:
                # 검색 모드가 아니면 paginated API 사용 (최신순)
                response = requests.get(
                    f"{self.api_url}/movies/paginated", params=params
                )

            if response.status_code == 200:
                data = response.json()
                movies = data.get("movies", [])
                total_pages = data.get("total_pages", 0)

                if movies:
                    # 기존 목록에 새 영화 추가
                    st.session_state["loaded_movies_edit"].extend(movies)
                    st.session_state["edit_current_page"] += 1

                    # 더 이상 로드할 페이지가 없는지 확인
                    if st.session_state["edit_current_page"] > total_pages:
                        st.session_state["edit_has_more"] = False
                else:
                    st.session_state["edit_has_more"] = False
            else:
                st.session_state["edit_has_more"] = False
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to load more movies: {str(e)}")
            st.session_state["edit_has_more"] = False

    def _render_movie_list(self):
        """영화 목록 렌더링 (무한 스크롤 방식)"""
        # 영화 개수 표시
        mode_text = "검색 결과" if st.session_state["edit_search_mode"] else "전체 영화"
        st.write(f"###### {mode_text}: {len(st.session_state['loaded_movies_edit'])}개")

        # 영화 카드 표시
        for movie in st.session_state["loaded_movies_edit"]:
            self._render_movie_card(movie)

        # "더 불러오기" 버튼
        if st.session_state["edit_has_more"]:
            _, col2, _ = st.columns([1, 8, 1])
            with col2:
                if st.button("📥 더 불러오기", key="load_more_edit", type="primary"):
                    self._load_more_movies()
                    st.rerun()
        else:
            st.info("✅ 모든 영화를 불러왔습니다.")

    def _render_movie_card(self, movie: dict):
        """영화 카드 렌더링"""
        movie_id = movie["id"]
        tmdb_id = movie["tmdb_id"]
        is_editing = st.session_state.get("editing_movie_id") == movie_id
        is_deleting = st.session_state.get("deleting_movie_id") == movie_id

        with st.container(border=True):
            cols_main = st.columns([1, 3])

            # 포스터 이미지
            with cols_main[0]:
                if movie.get("poster_local_path"):
                    poster_path = f"{self.api_url}/{movie['poster_local_path']}"
                else:
                    poster_path = "https://via.placeholder.com/300x450?text=No+Poster"

                st.markdown(
                    f'<div style="width: 100%; height: 350px; overflow: hidden; border-radius: 8px; background-color: #f0f0f0; display: flex; align-items: center; justify-content: center;"><img src="{poster_path}" alt="poster" style="width: 100%; height: 100%; object-fit: cover;"></div>',
                    unsafe_allow_html=True,
                )

            # 영화 정보 및 버튼
            with cols_main[1]:
                release_date = movie.get("release_date") or "개봉일 미정"
                st.write(f"###### 🎬 {movie['title']}")
                st.caption(f"📅 개봉일정 {release_date}")

                if movie.get("director"):
                    st.caption(f"🎥 감독: {movie['director']}")

                if movie.get("genre"):
                    st.caption(f"🎭 장르: {movie['genre']}")

                # TMDB 평점 및 AI 평점
                cols_rating = st.columns([1, 1, 6])
                with cols_rating[0]:
                    st.caption("평점:")
                    if movie.get("tmdb_rating") is not None:
                        st_label(
                            f"⭐ {movie['tmdb_rating']:.1f}",
                            color="orange",
                            font_weight="bold",
                        )

                with cols_rating[1]:
                    st.caption("AI 평점:")
                    if movie.get("ai_rating") is not None:
                        st_label(
                            f"🤖 {movie['ai_rating']:.1f}",
                            color="blue",
                            font_weight="bold",
                        )

                # 버튼 영역
                cols_btn = st.columns([1, 1, 1, 5])

                with cols_btn[0]:
                    # 수정 버튼 (다른 영화 수정 중이거나 삭제 확인 중이면 비활성화)
                    disabled = (
                        st.session_state.get("editing_movie_id") not in [None, movie_id]
                        or st.session_state.get("deleting_movie_id") is not None
                    )
                    if st.button(
                        "✏️ 수정" if not is_editing else "❌ 취소",
                        key=f"edit_movie_{movie_id}",
                        disabled=disabled,
                    ):
                        if is_editing:
                            st.session_state["editing_movie_id"] = None
                        else:
                            st.session_state["editing_movie_id"] = movie_id
                            st.session_state["deleting_movie_id"] = None
                        st.rerun()

                with cols_btn[1]:
                    # 삭제 버튼 (다른 영화 수정 중이거나 삭제 확인 중이면 비활성화)
                    disabled = st.session_state.get(
                        "editing_movie_id"
                    ) is not None or st.session_state.get("deleting_movie_id") not in [
                        None,
                        movie_id,
                    ]
                    if st.button(
                        "🗑️ 삭제",
                        key=f"delete_movie_{movie_id}",
                        disabled=disabled,
                    ):
                        st.session_state["deleting_movie_id"] = movie_id
                        st.session_state["editing_movie_id"] = None
                        st.rerun()

                # with cols_btn[2]:
                #     st.caption(f"ID: {movie_id} | TMDB: {tmdb_id}")

            # 수정 폼 (수정 모드일 때)
            if is_editing:
                st_div_divider()
                with st.expander("✏️ 영화 정보 수정", expanded=True):
                    self._render_edit_form(movie)

            # 삭제 확인 UI (삭제 확인 중일 때)
            if is_deleting:
                st_div_divider()
                self._render_delete_confirmation(movie)

    def _render_edit_form(self, movie: dict):
        """영화 수정 폼 렌더링"""
        movie_id = movie["id"]

        with st.form(key=f"edit_form_{movie_id}"):
            # TMDB ID (읽기 전용)
            st.text_input(
                "TMDB ID",
                value=str(movie["tmdb_id"]),
                disabled=True,
                help="TMDB ID는 수정할 수 없습니다.",
            )

            # 제목
            new_title = st.text_input(
                "영화 제목 *",
                value=movie["title"],
                max_chars=200,
                key=f"edit_title_{movie_id}",
            )

            # 개봉일
            new_release_date = st.text_input(
                "개봉일 (YYYY-MM-DD)",
                value=movie.get("release_date") or "",
                max_chars=10,
                key=f"edit_release_date_{movie_id}",
                placeholder="예: 2010-07-16",
            )

            # 감독
            new_director = st.text_input(
                "감독",
                value=movie.get("director") or "",
                max_chars=100,
                key=f"edit_director_{movie_id}",
            )

            # 장르
            new_genre = st.text_input(
                "장르",
                value=movie.get("genre") or "",
                max_chars=100,
                key=f"edit_genre_{movie_id}",
            )

            # TMDB 평점
            new_tmdb_rating = st.number_input(
                "TMDB 평점",
                min_value=0.0,
                max_value=10.0,
                value=float(movie.get("tmdb_rating") or 0.0),
                step=0.1,
                key=f"edit_tmdb_rating_{movie_id}",
            )

            # 포스터 URL
            new_poster_url = st.text_input(
                "포스터 URL",
                value=movie.get("poster_url") or "",
                max_chars=500,
                key=f"edit_poster_url_{movie_id}",
                placeholder="https://image.tmdb.org/t/p/w500/...",
            )

            # 포스터 미리보기
            if new_poster_url and new_poster_url != movie.get("poster_url"):
                st.caption("🖼️ 새 포스터 미리보기:")
                try:
                    st.image(new_poster_url, width=200)
                except Exception as e:
                    st.warning(f"⚠️ 포스터 미리보기 실패: {str(e)}")

            st.write("#### 추가 정보 (선택사항)")

            # 새 필드들 추가
            col1, col2 = st.columns(2)

            with col1:
                new_overview = st.text_area(
                    "줄거리",
                    value=movie.get("overview") or "",
                    key=f"edit_overview_{movie_id}",
                    height=100,
                )
                new_original_title = st.text_input(
                    "원제 (Original Title)",
                    value=movie.get("original_title") or "",
                    key=f"edit_original_title_{movie_id}",
                )
                new_original_language = st.text_input(
                    "원어 (Original Language)",
                    value=movie.get("original_language") or "",
                    key=f"edit_original_language_{movie_id}",
                    max_chars=10,
                )
                new_adult = st.checkbox(
                    "성인 영화",
                    value=movie.get("adult") or False,
                    key=f"edit_adult_{movie_id}",
                )

            with col2:
                new_popularity = st.number_input(
                    "인기도 (Popularity)",
                    min_value=0.0,
                    value=float(movie.get("popularity") or 0.0),
                    step=0.1,
                    key=f"edit_popularity_{movie_id}",
                )
                new_vote_count = st.number_input(
                    "투표 수 (Vote Count)",
                    min_value=0,
                    value=int(movie.get("vote_count") or 0),
                    key=f"edit_vote_count_{movie_id}",
                )
                new_backdrop_path = st.text_input(
                    "배경 이미지 URL",
                    value=movie.get("backdrop_path") or "",
                    key=f"edit_backdrop_path_{movie_id}",
                )

            # 버튼들
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                submit_button = st.form_submit_button("💾 저장", type="primary")
            with col2:
                cancel_button = st.form_submit_button("❌ 취소")

            # 폼 제출 처리
            if submit_button:
                if not new_title or not new_title.strip():
                    st.error("❌ 영화 제목을 입력해주세요.")
                else:
                    # 변경된 필드만 수집
                    update_data = {}

                    if new_title and new_title != movie["title"]:
                        update_data["title"] = new_title.strip()

                    if new_release_date != (movie.get("release_date") or ""):
                        update_data["release_date"] = (
                            new_release_date.strip()
                            if new_release_date.strip()
                            else None
                        )

                    if new_director != (movie.get("director") or ""):
                        update_data["director"] = (
                            new_director.strip() if new_director.strip() else None
                        )

                    if new_genre != (movie.get("genre") or ""):
                        update_data["genre"] = (
                            new_genre.strip() if new_genre.strip() else None
                        )

                    if new_tmdb_rating != (movie.get("tmdb_rating") or 0.0):
                        update_data["tmdb_rating"] = new_tmdb_rating

                    if new_poster_url != (movie.get("poster_url") or ""):
                        update_data["poster_url"] = (
                            new_poster_url.strip() if new_poster_url.strip() else None
                        )

                    # 새 필드들 처리
                    if new_overview != (movie.get("overview") or ""):
                        update_data["overview"] = (
                            new_overview.strip() if new_overview.strip() else None
                        )

                    if new_original_title != (movie.get("original_title") or ""):
                        update_data["original_title"] = (
                            new_original_title.strip()
                            if new_original_title.strip()
                            else None
                        )

                    if new_original_language != (movie.get("original_language") or ""):
                        update_data["original_language"] = (
                            new_original_language.strip()
                            if new_original_language.strip()
                            else None
                        )

                    if new_adult != (movie.get("adult") or False):
                        update_data["adult"] = new_adult

                    if new_popularity != (movie.get("popularity") or 0.0):
                        update_data["popularity"] = new_popularity

                    if new_vote_count != (movie.get("vote_count") or 0):
                        update_data["vote_count"] = new_vote_count

                    if new_backdrop_path != (movie.get("backdrop_path") or ""):
                        update_data["backdrop_path"] = (
                            new_backdrop_path.strip()
                            if new_backdrop_path.strip()
                            else None
                        )

                    if update_data:
                        self._update_movie(movie_id, update_data)
                    else:
                        st.info("ℹ️ 변경된 내용이 없습니다.")

            if cancel_button:
                st.session_state["editing_movie_id"] = None
                st.rerun()

    def _render_delete_confirmation(self, movie: dict):
        """영화 삭제 확인 UI 렌더링"""
        movie_id = movie["id"]

        # 영화의 리뷰 개수 조회
        review_count = self._get_review_count(movie_id)

        with st.container(border=True):
            st.warning("⚠️ 정말로 이 영화를 삭제하시겠습니까?")
            st.write(f"**영화 제목:** {movie['title']}")
            st.write(f"**TMDB ID:** {movie['tmdb_id']}")

            if review_count is not None:
                if review_count > 0:
                    st.error(
                        f"🔔 이 영화에 등록된 **리뷰 {review_count}개가 함께 삭제**됩니다!"
                    )
                else:
                    st.info("ℹ️ 이 영화에는 등록된 리뷰가 없습니다.")

            cols = st.columns([1, 1, 6])
            with cols[0]:
                if st.button(
                    "🗑️ 삭제 확인",
                    key=f"confirm_delete_{movie_id}",
                    type="primary",
                ):
                    self._delete_movie(movie_id)

            with cols[1]:
                if st.button("❌ 취소", key=f"cancel_delete_{movie_id}"):
                    st.session_state["deleting_movie_id"] = None
                    st.rerun()

    def _get_review_count(self, movie_id: int) -> Optional[int]:
        """영화의 리뷰 개수 조회"""
        try:
            response = requests.get(f"{self.api_url}/movies/{movie_id}")
            if response.status_code == 200:
                movie_data = response.json()
                reviews = movie_data.get("reviews", [])
                return len(reviews)
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch review count: {str(e)}")
        return None

    def _update_movie(self, movie_id: int, update_data: dict):
        """영화 업데이트 (PATCH 방식)"""
        try:
            response = requests.patch(
                f"{self.api_url}/movies/{movie_id}", json=update_data
            )

            if response.status_code == 200:
                # 캐시 무효화
                st.session_state["editing_movie_id"] = None

                # movie_list.py의 캐시 무효화
                if "loaded_movies" in st.session_state:
                    st.session_state["loaded_movies"] = []
                    st.session_state["current_page"] = 1
                    st.session_state["has_more"] = True

                # movie_edit.py의 캐시 무효화
                st.session_state["loaded_movies_edit"] = []
                st.session_state["edit_current_page"] = 1
                st.session_state["edit_has_more"] = True

                # 모든 평점 캐시 무효화
                keys_to_delete = [
                    k
                    for k in st.session_state.keys()
                    if isinstance(k, str) and k.startswith("cached_rating_")
                ]
                for key in keys_to_delete:
                    del st.session_state[key]

                st.success("✅ 영화 정보가 수정되었습니다.")
                st.rerun()
            elif response.status_code == 400:
                error_detail = response.json().get("detail", "영화 수정 실패")
                st.error(f"❌ {error_detail}")
            elif response.status_code == 404:
                st.error("❌ 영화를 찾을 수 없습니다.")
            else:
                st.error(f"❌ 영화 수정 실패 (상태 코드: {response.status_code})")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ API 연결 오류: {str(e)}")

    def _delete_movie(self, movie_id: int):
        """영화 삭제"""
        try:
            response = requests.delete(f"{self.api_url}/movies/{movie_id}")

            if response.status_code == 204:
                # 캐시 무효화
                st.session_state["deleting_movie_id"] = None
                st.session_state["editing_movie_id"] = None

                # movie_list.py의 캐시 무효화
                if "loaded_movies" in st.session_state:
                    st.session_state["loaded_movies"] = []
                    st.session_state["current_page"] = 1
                    st.session_state["has_more"] = True

                # movie_edit.py의 캐시 무효화
                st.session_state["loaded_movies_edit"] = []
                st.session_state["edit_current_page"] = 1
                st.session_state["edit_has_more"] = True

                # 모든 평점 캐시 무효화
                keys_to_delete = [
                    k
                    for k in st.session_state.keys()
                    if isinstance(k, str) and k.startswith("cached_rating_")
                ]
                for key in keys_to_delete:
                    del st.session_state[key]

                st.success("✅ 영화가 삭제되었습니다.")
                st.rerun()
            elif response.status_code == 404:
                st.error("❌ 영화를 찾을 수 없습니다.")
            else:
                st.error(f"❌ 영화 삭제 실패 (상태 코드: {response.status_code})")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ API 연결 오류: {str(e)}")
