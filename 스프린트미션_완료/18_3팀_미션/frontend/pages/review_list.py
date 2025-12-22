"""
리뷰 목록 페이지 (review_list.py)
독립적인 리뷰 목록 페이지
"""

import streamlit as st
import requests
from typing import List, Dict, Optional
import os
import logging
from datetime import datetime
from helper_dev_utils import get_auto_logger
from utils import *
from utils.search_ui import MovieSearchUI

logger = get_auto_logger(log_level=logging.DEBUG)

# 백엔드 API URL
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


class ReviewListManager:
    """
    리뷰 목록 관리 클래스
    """

    def __init__(self):
        """ReviewListManager 초기화"""
        self.api_url = API_BASE_URL

    def render(self):
        """리뷰 목록 페이지 렌더링 (독립 페이지)"""

        if st.button(
            "###### 📝 리뷰 목록", key="toggle_sidebar_review_list", help="사이드바"
        ):
            st_sidebar_show()

        # 영화 검색 섹션
        self._render_movie_search()

        st_div_divider()

        # 리뷰 목록
        self._render_review_list()

    def _render_review_list(self):
        """리뷰 목록 렌더링 (무한 스크롤 방식)"""
        # 세션 상태 초기화
        if "loaded_reviews" not in st.session_state:
            st.session_state["loaded_reviews"] = []
            st.session_state["reviews_current_page"] = 1
            st.session_state["reviews_has_more"] = True

        # 수정 중인 리뷰 ID 추적
        if "editing_review_id" not in st.session_state:
            st.session_state["editing_review_id"] = None

        # 페이지 크기 고정 (32개)
        if "reviews_page_size" not in st.session_state:
            st.session_state["reviews_page_size"] = 32

        # 새로고침 버튼 및 리뷰 개수 표시
        cols = st.columns([2, 6])
        with cols[0]:
            if st.button("🔄 새로고침", key="refresh_reviews"):
                st.session_state["loaded_reviews"] = []
                st.session_state["reviews_current_page"] = 1
                st.session_state["reviews_has_more"] = True
                st.session_state["editing_review_id"] = None
                st.rerun()

        with cols[1]:
            # 초기 로드 후 리뷰 개수 표시
            review_count_placeholder = st_label("")

        # 초기 로드: 첫 페이지 자동 로드
        if (
            not st.session_state["loaded_reviews"]
            and st.session_state["reviews_has_more"]
        ):
            self._load_more_reviews()

        # 리뷰 개수 표시 업데이트
        st_label(
            review_count_placeholder,
            value=(
                f"{len(st.session_state['loaded_reviews'])}개의 리뷰"
                if st.session_state["loaded_reviews"]
                else "📭 등록된 리뷰가 없습니다."
            ),
            color="blue",
        )

        # 로드된 리뷰가 없으면 종료
        if not st.session_state["loaded_reviews"]:
            return

        st_div_divider()

        # 누적된 모든 리뷰 표시
        for review in st.session_state["loaded_reviews"]:
            self._render_review_card(review)

        # "더 불러오기" 버튼
        if st.session_state["reviews_has_more"]:
            _, col2, _ = st.columns([1, 8, 1])
            with col2:
                if st.button("📥 더 불러오기", width="content", type="primary"):
                    self._load_more_reviews()
                    st.rerun()
        else:
            st.info("✅ 모든 리뷰를 불러왔습니다.")

    def _render_review_card(self, review: dict):
        """리뷰 카드 렌더링"""
        review_id = review["id"]
        is_editing = st.session_state.get("editing_review_id") == review_id

        with st.container(border=True):
            # 영화 정보
            if "movie" in review and review["movie"]:
                movie = review["movie"]
                release_date = movie.get("release_date") or "개봉일 미정"
                st.write(f"##### 🎬 {movie['title']} ({release_date})")
            else:
                st.write(f"##### 🎬 TMDB ID: {review['tmdb_id']}")

            cols2 = st.columns([2, 2, 3, 3])
            with cols2[0]:
                # 작성자
                st.caption(f"✍️ {review['author']}")

            with cols2[1]:
                # 작성시간 및 수정시간
                if review.get("created_at"):
                    try:
                        created_at = datetime.fromisoformat(
                            review["created_at"].replace("Z", "+00:00")
                        )
                        time_text = f"📅 {created_at.strftime('%Y-%m-%d %H:%M')}"

                        # 수정 여부 확인
                        if review.get("updated_at"):
                            updated_at = datetime.fromisoformat(
                                review["updated_at"].replace("Z", "+00:00")
                            )
                            if updated_at > created_at:
                                time_text += f" (수정됨: {updated_at.strftime('%Y-%m-%d %H:%M')})"

                        st.caption(time_text)
                    except Exception as e:
                        logger.debug(f"Failed to parse datetime: {e}")
            with cols2[2]:
                cols4 = st.columns([1, 1, 1])
                with cols4[0]:
                    # 감성 분석 결과
                    if review.get("is_positive") is not None:
                        if review["is_positive"] == 1:
                            st_label("긍정", color="green", font_weight="bold")
                        else:
                            st_label("부정", color="red", font_weight="bold")
                    else:
                        st_label("분석중", color="gray", font_weight="bold")

                with cols4[1]:
                    # 수정 버튼
                    if st.button(
                        "수정" if not is_editing else "취소",
                        key=f"edit_review_{review_id}",
                    ):
                        if is_editing:
                            st.session_state["editing_review_id"] = None
                        else:
                            st.session_state["editing_review_id"] = review_id
                        st.rerun()

                with cols4[2]:
                    # 삭제 버튼
                    if st.button("삭제", key=f"delete_review_{review_id}"):
                        self._delete_review(review_id)

            # 리뷰 내용 (수정 모드가 아닐 때)
            if not is_editing:
                with st.container(border=True):
                    st.write(review["content"])

            # 수정 폼 (expander로 표시)
            if is_editing:
                with st.expander("✏️ 리뷰 수정", expanded=True):
                    self._render_edit_form(review)

    def _render_edit_form(self, review: dict):
        """리뷰 수정 폼 렌더링"""
        review_id = review["id"]

        with st.form(key=f"edit_form_{review_id}"):
            # 작성자 입력
            new_author = st.text_input(
                "작성자",
                value=review["author"],
                max_chars=100,
                key=f"edit_author_{review_id}",
            )

            # 리뷰 내용 입력
            new_content = st.text_area(
                "리뷰 내용",
                value=review["content"],
                max_chars=2000,
                height=150,
                key=f"edit_content_{review_id}",
            )

            # 버튼들
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                submit_button = st.form_submit_button("💾 저장", type="primary")
            with col2:
                cancel_button = st.form_submit_button("❌ 취소")

            # 폼 제출 처리
            if submit_button:
                if not new_author.strip():
                    st.error("작성자를 입력해주세요.")
                elif not new_content.strip():
                    st.error("리뷰 내용을 입력해주세요.")
                else:
                    self._update_review(
                        review_id, new_author.strip(), new_content.strip()
                    )

            if cancel_button:
                st.session_state["editing_review_id"] = None
                st.rerun()

    def _update_review(self, review_id: int, author: str, content: str):
        """리뷰 업데이트"""
        try:
            update_data = {"author": author, "content": content}

            response = api_put(f"{self.api_url}/reviews/{review_id}", json=update_data)

            if response.status_code == 200:
                # 캐시 무효화
                st.session_state["loaded_reviews"] = []
                st.session_state["reviews_current_page"] = 1
                st.session_state["reviews_has_more"] = True
                st.session_state["editing_review_id"] = None

                # 모든 평점 캐시 무효화
                keys_to_delete = [
                    k for k in st.session_state.keys() if k.startswith("cached_rating_")
                ]
                for key in keys_to_delete:
                    del st.session_state[key]

                st.success("리뷰가 수정되었습니다.")
                st.rerun()
            elif response.status_code == 400:
                error_detail = response.json().get("detail", "리뷰 수정 실패")
                st.error(f"❌ {error_detail}")
            elif response.status_code == 404:
                st.error("❌ 리뷰를 찾을 수 없습니다.")
            else:
                st.error(f"❌ 리뷰 수정 실패 (상태 코드: {response.status_code})")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ API 연결 오류: {str(e)}")

    def _load_more_reviews(self):
        """다음 페이지의 리뷰를 로드하여 누적 목록에 추가"""
        # 검색된 영화 목록 가져오기 (None: 전체 리뷰, list: 필터링된 리뷰)
        searched_movies = st.session_state.get("searched_movies", None)

        # 검색된 영화의 tmdb_id 목록 생성
        # searched_movies가 None이거나 빈 리스트면 None (전체 리뷰 조회)
        if searched_movies:
            tmdb_ids = [movie["tmdb_id"] for movie in searched_movies]
        else:
            tmdb_ids = None

        pagination_data = self._get_reviews_paginated(
            st.session_state["reviews_current_page"],
            st.session_state["reviews_page_size"],
            tmdb_ids,
        )

        if pagination_data:
            reviews = pagination_data.get("reviews", [])
            total_pages = pagination_data.get("total_pages", 0)

            if reviews:
                # 기존 목록에 새 리뷰 추가
                st.session_state["loaded_reviews"].extend(reviews)
                st.session_state["reviews_current_page"] += 1

                # 더 이상 로드할 페이지가 없는지 확인
                if st.session_state["reviews_current_page"] > total_pages:
                    st.session_state["reviews_has_more"] = False
            else:
                st.session_state["reviews_has_more"] = False
        else:
            st.session_state["reviews_has_more"] = False

    def _get_reviews_paginated(
        self, page: int, page_size: int, tmdb_ids: List[int] = None
    ) -> Optional[Dict]:
        """페이지네이션된 리뷰 목록 가져오기 (영화 필터링 포함)"""
        try:
            params = {"page": page, "page_size": page_size}

            # tmdb_ids가 제공되고 비어있지 않으면 필터링 파라미터 추가
            # None이거나 빈 리스트면 전체 리뷰 조회
            if tmdb_ids is not None and len(tmdb_ids) > 0:
                # 리스트를 쉼표로 구분된 문자열로 변환
                params["tmdb_ids"] = ",".join(map(str, tmdb_ids))

            response = api_get(
                f"{self.api_url}/reviews/paginated",
                params=params,
            )
            if response.status_code == 200:
                return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch paginated reviews: {str(e)}")
        return None

    def _delete_review(self, review_id: int):
        """
        리뷰 삭제 (최적화 전략)
        1. 로컬 리스트에서 삭제 (즉시 UI 반영)
        2. 서버에 삭제 요청 (백그라운드)
        3. 성공 확인만, 리스트 갱신 없음 (시간 절약)
        """
        try:
            # 삭제할 리뷰의 tmdb_id 저장 (평점 캐시 무효화용)
            deleted_tmdb_id = None
            if "loaded_reviews" in st.session_state:
                for review in st.session_state["loaded_reviews"]:
                    if review["id"] == review_id:
                        deleted_tmdb_id = review.get("movie", {}).get("tmdb_id")
                        break

            # 1. 로컬 리스트에서 먼저 제거 (즉시 반영)
            if "loaded_reviews" in st.session_state:
                st.session_state["loaded_reviews"] = [
                    review
                    for review in st.session_state["loaded_reviews"]
                    if review["id"] != review_id
                ]

            # 2. 서버에 삭제 요청
            response = api_delete(f"{self.api_url}/reviews/{review_id}")

            if response.status_code == 204:
                # 3. 성공 확인만, 해당 영화의 평점 캐시만 무효화
                if deleted_tmdb_id:
                    cache_key = f"cached_rating_{deleted_tmdb_id}"
                    if cache_key in st.session_state:
                        del st.session_state[cache_key]

                st.success("리뷰가 삭제되었습니다.")
                st.rerun()
            else:
                # 삭제 실패 시 리스트 복원 (롤백)
                st.error("리뷰 삭제 실패")
                st.session_state["loaded_reviews"] = []
                st.session_state["reviews_current_page"] = 1
                st.session_state["reviews_has_more"] = True
                st.rerun()
        except requests.exceptions.RequestException as e:
            st.error(f"API 연결 오류: {str(e)}")
            # 네트워크 오류 시 복원
            st.session_state["loaded_reviews"] = []
            st.session_state["reviews_current_page"] = 1
            st.session_state["reviews_has_more"] = True
            st.rerun()

    def _render_movie_search(self):
        """영화 검색 섹션 렌더링"""
        # 영화 검색 UI
        search_ui = MovieSearchUI(show_ai_rating=False, show_sort_options=True)
        search_triggered, filters = search_ui.render()

        if search_triggered is False:
            # 초기화
            st.session_state["searched_movies"] = None
            st.session_state["search_performed"] = False
            self._reset_review_list()
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
            self._reset_review_list()
            st.rerun()

        # 초기 로드: searched_movies가 없으면 None (전체 리뷰)
        if "searched_movies" not in st.session_state:
            st.session_state["searched_movies"] = None
            st.session_state["search_performed"] = False

        # 검색 결과 정보 표시
        movies = st.session_state.get("searched_movies")
        if movies is not None:
            if len(movies) > 0:
                if st.session_state.get("search_performed", False):
                    st.info(f"📋 검색된 영화: {len(movies)}개의 리뷰")
                else:
                    st.info(f"📋 전체 리뷰 표시")
            else:
                st.warning(
                    "검색 결과가 없습니다. 검색 조건을 변경하거나 초기화 버튼을 눌러주세요."
                )
        else:
            st.info("📋 전체 리뷰 표시")

    def _reset_review_list(self):
        """리뷰 목록 초기화"""
        if "loaded_reviews" in st.session_state:
            st.session_state["loaded_reviews"] = []
        if "reviews_current_page" in st.session_state:
            st.session_state["reviews_current_page"] = 1
        if "reviews_has_more" in st.session_state:
            st.session_state["reviews_has_more"] = True

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
if __name__ == "__main__":
    manager = ReviewListManager()
    manager.render()
