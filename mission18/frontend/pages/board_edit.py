"""
리뷰 작성 모듈 (board_edit.py)
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

logger = get_auto_logger(log_level=logging.DEBUG)

# 백엔드 API URL
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
TEST_MODE = os.getenv("TEST_MODE", "False") == "True"


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


class ReviewEditManager:
    """
    리뷰 작성 관리 클래스
    """

    def __init__(self):
        """ReviewEditManager 초기화"""
        self.api_url = API_BASE_URL

    def render(self):
        """리뷰 작성 폼 렌더링"""
        st.write("##### 리뷰 작성")

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
        selected_movie = st.selectbox(
            "영화 선택 *",
            options=movie_names,
            key="selected_movie_for_review",
            help="검색된 영화 중에서 선택하세요",
        )

        # 선택된 영화 TMDB ID 가져오기
        selected_tmdb_id = movie_options[selected_movie]

        with st.form("review_form"):
            # 작성자 이름
            cols1 = st.columns([1, 9])
            with cols1[0]:
                st_label("작성자")

            with cols1[1]:
                author = st_text_input(
                    "작성자 이름 *",
                    placeholder="예: 홍길동",
                    value="작성자",
                    label_visibility="collapsed",
                )

            cols2 = st.columns([1, 9])
            with cols2[0]:
                st_label("리뷰<br/>내용")
            with cols2[1]:
                content = st_text_review_memo(
                    "리뷰 내용 *",
                    placeholder="영화에 대한 리뷰를 작성해주세요...",
                    height=70,
                    value="이 영화는 정말 ",
                    label_visibility="collapsed",
                )

            cols3 = st.columns([1, 9])
            with cols3[0]:
                st_label("작성일")
            with cols3[1]:
                cols4 = st.columns([4, 1])
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
                    submitted = st.form_submit_button("리뷰 등록", width="stretch")

            if submitted:
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
        st.divider()
        self._render_movie_rating_section(selected_tmdb_id, movies)

    def _render_movie_rating_section(self, movie_id: int, movies: List[Dict]):
        """영화 AI 평점 섹션 렌더링"""
        st.header("영화 AI 평점")

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
                st.metric("AI 평점", f"{rating_data['ai_rating']}/5.0")
        else:
            st.info("해당 영화에 대한 리뷰가 아직 없습니다.")

    def _render_gauge_chart(self, rating_data: Dict):
        """Gauge Chart 렌더링"""
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

    def _get_movie_rating(self, tmdb_id: int) -> Optional[Dict]:
        """영화 평점 조회"""
        try:
            response = requests.get(f"{self.api_url}/reviews/movie/{tmdb_id}/rating")
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
            response = requests.post(f"{self.api_url}/reviews/", json=review_data)

            if response.status_code == 201:
                review = response.json()
                sentiment = "긍정" if review.get("is_positive") == 1 else "부정"

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
                success_msg += f"- AI 분석 결과: **{sentiment}**\n"
                if created_at_str:
                    success_msg += f"- 작성일시: {created_at_str}"
                st.success(success_msg)
                st.balloons()
                st.rerun()
            else:
                error_detail = response.json().get("detail", "알 수 없는 오류")
                st.error(f"리뷰 등록 실패: {error_detail}")
        except requests.exceptions.RequestException as e:
            st.error(f"API 연결 오류: {str(e)}")
