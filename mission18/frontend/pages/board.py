"""
리뷰 게시판 페이지
"""

import streamlit as st
import requests
import plotly.graph_objects as go
from typing import List, Dict, Optional
import os
import random
import logging
from datetime import datetime
from helper_dev_utils import get_auto_logger
from utils import *

logger = get_auto_logger(log_level=logging.DEBUG)


# 백엔드 API URL
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
TEST_MODE = os.getenv("TEST_MODE", "False") == "True"

logger.debug(f"API_BASE_URL: {API_BASE_URL}")
logger.debug(f"TEST_MODE: {TEST_MODE}")


def random_text(text: str) -> str:
    """텍스트에 랜덤 한글 자음 추가"""
    consonants = "가나다라마바사아자차카타파하거너더러머버서어저처커터퍼허고노도로모보소오조초코토포호구누두루무부수우주추쿠투푸후그느드르므브스으즈츠크트프흐기니디리미비시이지치키티피히"

    consonant = random.choice(consonants)
    result = text + consonant
    logger.debug(f"Generated random text: {result}")
    return result


def st_text_input(label, **kwargs):
    """테스트 모드에서 랜덤 한글 글자 추가된 텍스트 입력"""
    if TEST_MODE and "value" in kwargs:
        kwargs["value"] = random_text(kwargs["value"])
    if TEST_MODE and not "value" in kwargs:
        kwargs["value"] = random_text("테스트_")
    return st.text_input(label, **kwargs)


def st_text_review_memo(label, **kwargs):
    """테스트 모드에서 랜덤 리뷰 문장 추가된 텍스트 입력"""
    if TEST_MODE:
        review_sentences = [
            "그럼에도 불구하고 너무나 재미 있어요 ㅋㅋㅋ",
            "그래서 너무 재미 없어요ㅠㅠ",
        ]
        random_review = random.choice(review_sentences)

        if "value" in kwargs:
            kwargs["value"] = kwargs["value"] + " " + random_review
        else:
            kwargs["value"] = "심야 영화를 봤는데 " + random_review

    return st.text_area(label, **kwargs)


class ReviewManager:
    """
    리뷰 관리 클래스 - 리뷰 작성 및 AI 평점 시각화
    """

    def __init__(self):
        """
        ReviewManager 초기화
        """
        self.api_url = API_BASE_URL

    def render(self):
        """
        리뷰 게시판 페이지 렌더링
        """
        st.write("##### 리뷰 게시판")
        st.write("영화 리뷰를 작성하고 AI 감성 분석 결과를 확인할 수 있습니다.")

        # 탭 구성
        tab1, tab2 = st.tabs(["리뷰 작성", "리뷰 목록"])

        with tab1:
            self._render_review_form()

        with tab2:
            self._render_review_list()

    def _render_review_form(self):
        """
        리뷰 작성 폼 렌더링
        """
        st.write("##### 리뷰 작성")

        # 영화 목록 캐싱 (중복 API 호출 방지)
        if "cached_movies" not in st.session_state:
            st.session_state["cached_movies"] = self._get_movies()

        movies = st.session_state["cached_movies"]

        if not movies:
            st.warning("등록된 영화가 없습니다. 먼저 영화를 등록해주세요.")
            return

        # 영화 선택 (폼 외부에서 처리하여 AI 평점과 동기화)
        movie_options = {
            f"{m['title']} ({m.get('release_date') or '개봉일 미정'})": m["tmdb_id"]
            for m in movies
        }
        movie_names = list(movie_options.keys())

        # 세션 상태 초기화: 첫 번째 영화를 기본값으로 설정
        if "selected_movie_for_review" not in st.session_state:
            st.session_state["selected_movie_for_review"] = movie_names[0]

        # 영화 선택 드롭다운
        selected_movie = st.selectbox(
            "영화 선택 *", options=movie_names, key="selected_movie_for_review"
        )

        # 선택된 영화 TMDB ID 가져오기
        selected_tmdb_id = movie_options[selected_movie]

        with st.form("review_form"):
            # 작성자 이름
            author = st_text_input(
                "작성자 이름 *", placeholder="예: 홍길동", value="작성자"
            )

            # 리뷰 내용
            content = st_text_review_memo(
                "리뷰 내용 *",
                placeholder="영화에 대한 리뷰를 작성해주세요...",
                height=100,
                value="이 영화는 정말 ",
            )

            submitted = st.form_submit_button("리뷰 등록", width="content")

            if submitted:
                if not author or not content:
                    st.error("작성자 이름과 리뷰 내용은 필수 입력 항목입니다.")
                else:
                    self._register_review(selected_tmdb_id, author, content)

        # 선택된 영화의 AI 평점 표시
        st.divider()
        self._render_movie_rating_section(selected_tmdb_id, movies)

    def _render_movie_rating_section(self, movie_id: int, movies: List[Dict]):
        """
        영화 AI 평점 섹션 렌더링

        Args:
            movie_id: 선택된 영화 ID
            movies: 영화 목록
        """
        st.header("영화 AI 평점")

        # 선택된 영화의 평점 데이터 캐싱 (중복 API 호출 방지)
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
                st.metric("AI 평점", f"{rating_data['ai_rating']}/5.0")
        else:
            st.info("해당 영화에 대한 리뷰가 아직 없습니다.")

    def _render_gauge_chart(self, rating_data: Dict):
        """
        Gauge Chart 렌더링

        Args:
            rating_data: 평점 데이터
        """
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=rating_data["ai_rating"],
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "AI 평점 (5점 만점)", "font": {"size": 24}},
                delta={"reference": 2.5, "increasing": {"color": "green"}},
                gauge={
                    "axis": {
                        "range": [None, 5],
                        "tickwidth": 1,
                        "tickcolor": "darkblue",
                    },
                    "bar": {"color": "darkblue"},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "bordercolor": "gray",
                    "steps": [
                        {"range": [0, 1], "color": "#ffcccc"},
                        {"range": [1, 2], "color": "#ffddcc"},
                        {"range": [2, 3], "color": "#ffffcc"},
                        {"range": [3, 4], "color": "#ddffcc"},
                        {"range": [4, 5], "color": "#ccffcc"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 4.5,
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

    def _render_review_list(self):
        """
        리뷰 목록 렌더링 (무한 스크롤 방식)
        """
        st.write("##### 최근 리뷰 목록")

        # 세션 상태 초기화
        if "loaded_reviews" not in st.session_state:
            st.session_state["loaded_reviews"] = []
            st.session_state["reviews_current_page"] = 1
            st.session_state["reviews_has_more"] = True

        # URL query params에서 페이지 크기 로드 (기본값 10)
        if "reviews_page_size" not in st.session_state:
            st.session_state["reviews_page_size"] = st_query_param_get("page_size", 10)
            logger.debug(f"reviews_page_size: {st.session_state['reviews_page_size']}")

        # 페이지 크기 선택 및 새로고침 버튼
        cols = st.columns([1.5, 1, 5, 2, 1])
        with cols[0]:
            if st.button("🔄 새로고침", key="refresh_reviews"):
                st.session_state["loaded_reviews"] = []
                st.session_state["reviews_current_page"] = 1
                st.session_state["reviews_has_more"] = True
                st.rerun()

        with cols[1]:
            # 초기 로드 후 리뷰 개수 표시 (placeholder로 먼저 생성)
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

        # 페이지 크기 로깅
        logger.debug(f"reviews_page_size: {st.session_state['reviews_page_size']}")

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

        # 로드된 리뷰가 없으면 여기서 종료
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
                if st.button("📥 더 불러오기", width="stretch", type="primary"):
                    self._load_more_reviews()
                    st.rerun()
        else:
            st.info("✅ 모든 리뷰를 불러왔습니다.")

    def _render_review_card(self, review: dict):
        """
        리뷰 카드 렌더링

        Args:
            review: 리뷰 정보 딕셔너리
        """
        with st.container(border=True):
            # 영화 정보
            if "movie" in review and review["movie"]:
                movie = review["movie"]
                release_date = movie.get("release_date") or "개봉일 미정"
                st.write(f"##### 🎬 {movie['title']} ({release_date})")
            else:
                st.write(f"##### 🎬 TMDB ID: {review['tmdb_id']}")

            cols2 = st.columns([1, 2, 6])
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

            cols3 = st.columns([6, 1])
            with cols3[0]:
                # 리뷰 내용
                st.write(review["content"])
            with cols3[1]:
                cols4 = st.columns([1, 1])
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
                    # 삭제 버튼
                    if st.button("삭제", key=f"delete_review_{review['id']}"):
                        self._delete_review(review["id"])

    def _get_movies(self) -> List[Dict]:
        """
        영화 목록 가져오기

        Returns:
            영화 목록
        """
        try:
            response = requests.get(f"{self.api_url}/movies/")
            if response.status_code == 200:
                return response.json()
        except requests.exceptions.RequestException:
            pass
        return []

    def _get_movie_rating(self, tmdb_id: int) -> Optional[Dict]:
        """
        영화 평점 조회

        Args:
            tmdb_id: TMDB 영화 ID

        Returns:
            평점 데이터 또는 None
        """
        try:
            response = requests.get(f"{self.api_url}/reviews/movie/{tmdb_id}/rating")
            if response.status_code == 200:
                return response.json()
        except requests.exceptions.RequestException:
            pass
        return None

    def _load_more_reviews(self):
        """
        다음 페이지의 리뷰를 로드하여 누적 목록에 추가
        """
        pagination_data = self._get_reviews_paginated(
            st.session_state["reviews_current_page"],
            st.session_state["reviews_page_size"],
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

    def _get_reviews_paginated(self, page: int, page_size: int) -> Optional[Dict]:
        """
        페이지네이션된 리뷰 목록 가져오기 (영화 정보 포함)

        Args:
            page: 페이지 번호
            page_size: 페이지당 항목 수

        Returns:
            페이지네이션 데이터 (리뷰 목록, 전체 개수 등)
        """
        try:
            response = requests.get(
                f"{self.api_url}/reviews/paginated",
                params={"page": page, "page_size": page_size},
            )
            if response.status_code == 200:
                return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch paginated reviews: {str(e)}")
        return None

    def _register_review(self, tmdb_id: int, author: str, content: str):
        """
        리뷰 등록

        Args:
            tmdb_id: TMDB 영화 ID
            author: 작성자
            content: 리뷰 내용
        """
        review_data = {"tmdb_id": tmdb_id, "author": author, "content": content}

        try:
            response = requests.post(f"{self.api_url}/reviews/", json=review_data)

            if response.status_code == 201:
                review = response.json()
                sentiment = "긍정" if review.get("is_positive") == 1 else "부정"

                # 캐시 무효화 (리뷰 목록 및 평점 갱신)
                if "loaded_reviews" in st.session_state:
                    st.session_state["loaded_reviews"] = []
                    st.session_state["reviews_current_page"] = 1
                    st.session_state["reviews_has_more"] = True
                # 평점 캐시 무효화
                cache_key = f"cached_rating_{tmdb_id}"
                if cache_key in st.session_state:
                    del st.session_state[cache_key]

                st.success(f"리뷰가 등록되었습니다! (AI 분석 결과: {sentiment})")
                st.balloons()
                st.rerun()
            else:
                error_detail = response.json().get("detail", "알 수 없는 오류")
                st.error(f"리뷰 등록 실패: {error_detail}")
        except requests.exceptions.RequestException as e:
            st.error(f"API 연결 오류: {str(e)}")

    def _delete_review(self, review_id: int):
        """
        리뷰 삭제

        Args:
            review_id: 리뷰 ID
        """
        try:
            response = requests.delete(f"{self.api_url}/reviews/{review_id}")

            if response.status_code == 204:
                # 캐시 무효화 (리뷰 목록 및 평점 갱신)
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


# 페이지 실행
if __name__ == "__main__":
    manager = ReviewManager()
    manager.render()
