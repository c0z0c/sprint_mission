"""
리뷰 목록 모듈 (board_list.py)
"""

import streamlit as st
import requests
from typing import List, Dict, Optional
import os
import logging
from datetime import datetime
from helper_dev_utils import get_auto_logger
from utils import *

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
        """리뷰 목록 렌더링 (무한 스크롤 방식)"""
        st.write("##### 최근 리뷰 목록")

        # 세션 상태 초기화
        if "loaded_reviews" not in st.session_state:
            st.session_state["loaded_reviews"] = []
            st.session_state["reviews_current_page"] = 1
            st.session_state["reviews_has_more"] = True

        # URL query params에서 페이지 크기 로드
        if "reviews_page_size" not in st.session_state:
            st.session_state["reviews_page_size"] = st_query_param_get("page_size", 10)
            logger.debug(f"reviews_page_size: {st.session_state['reviews_page_size']}")

        # 페이지 크기 선택 및 새로고침 버튼
        cols = st.columns([2, 2, 4, 2, 1])
        with cols[0]:
            if st.button("🔄 새로고침", key="refresh_reviews"):
                st.session_state["loaded_reviews"] = []
                st.session_state["reviews_current_page"] = 1
                st.session_state["reviews_has_more"] = True
                st.rerun()

        with cols[1]:
            # 초기 로드 후 리뷰 개수 표시
            review_count_placeholder = st_label("")

        with cols[3]:
            st_label("한 번에 불러올 리뷰 개수:")

        with cols[4]:
            new_page_size = st.slider(
                "한 번에 불러올 리뷰 개수",
                min_value=5,
                max_value=50,
                value=int(st.session_state["reviews_page_size"]),
                step=5,
                key="reviews_page_size_slider",
                label_visibility="collapsed",
            )
            logger.debug(f"Selected new_page_size: {new_page_size}")
            # 페이지 크기가 변경되면 URL에 저장
            if new_page_size != st.session_state["reviews_page_size"]:
                st.session_state["reviews_page_size"] = new_page_size
                st_query_param_set("page_size", new_page_size)

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
                    if st.button("수정", key=f"edit_review_{review['id']}"):
                        # self._edit_review(review["id"])
                        pass

                with cols4[2]:
                    # 삭제 버튼
                    if st.button("삭제", key=f"delete_review_{review['id']}"):
                        self._delete_review(review["id"])

            # 리뷰 내용
            with st.container(border=True):
                st.write(review["content"])

    def _load_more_reviews(self):
        """다음 페이지의 리뷰를 로드하여 누적 목록에 추가"""
        # 검색된 영화 목록 가져오기
        searched_movies = st.session_state.get("searched_movies", [])

        # 검색된 영화의 tmdb_id 목록 생성
        if searched_movies:
            tmdb_ids = [movie["tmdb_id"] for movie in searched_movies]
        else:
            tmdb_ids = []

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

            # tmdb_ids가 제공되면 필터링 파라미터 추가
            if tmdb_ids:
                # 리스트를 쉼표로 구분된 문자열로 변환
                params["tmdb_ids"] = ",".join(map(str, tmdb_ids))

            response = requests.get(
                f"{self.api_url}/reviews/paginated",
                params=params,
            )
            if response.status_code == 200:
                return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch paginated reviews: {str(e)}")
        return None

    def _delete_review(self, review_id: int):
        """리뷰 삭제"""
        try:
            response = requests.delete(f"{self.api_url}/reviews/{review_id}")

            if response.status_code == 204:
                # 캐시 무효화
                if "loaded_reviews" in st.session_state:
                    st.session_state["loaded_reviews"] = []
                    st.session_state["reviews_current_page"] = 1
                    st.session_state["reviews_has_more"] = True
                # 모든 평점 캐시 무효화
                keys_to_delete = [
                    k for k in st.session_state.keys() if k.startswith("cached_rating_")
                ]
                for key in keys_to_delete:
                    del st.session_state[key]

                st.success("리뷰가 삭제되었습니다.")
                st.rerun()
            else:
                st.error("리뷰 삭제 실패")
        except requests.exceptions.RequestException as e:
            st.error(f"API 연결 오류: {str(e)}")
