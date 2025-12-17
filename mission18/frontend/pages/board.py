"""
리뷰 게시판 페이지
"""

import streamlit as st
import requests
import plotly.graph_objects as go
from typing import List, Dict, Optional
import os

# 백엔드 API URL
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


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
        st.title("💬 리뷰 게시판")
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
        st.header("리뷰 작성")

        # 영화 목록 불러오기
        movies = self._get_movies()

        if not movies:
            st.warning("⚠️ 등록된 영화가 없습니다. 먼저 영화를 등록해주세요.")
            return

        with st.form("review_form"):
            # 영화 선택
            movie_options = {
                f"{m['title']} (TMDB ID: {m['tmdb_id']})": m["id"] for m in movies
            }
            selected_movie = st.selectbox(
                "영화 선택 *", options=list(movie_options.keys())
            )

            # 작성자 이름
            author = st.text_input("작성자 이름 *", placeholder="예: 홍길동")

            # 리뷰 내용
            content = st.text_area(
                "리뷰 내용 *",
                placeholder="영화에 대한 리뷰를 작성해주세요...",
                height=200,
            )

            submitted = st.form_submit_button("리뷰 등록", use_container_width=True)

            if submitted:
                if not author or not content:
                    st.error("⚠️ 작성자 이름과 리뷰 내용은 필수 입력 항목입니다.")
                else:
                    movie_id = movie_options[selected_movie]
                    self._register_review(movie_id, author, content)

        # 선택된 영화의 AI 평점 표시
        if movies:
            st.divider()
            self._render_movie_rating_section(movies)

    def _render_movie_rating_section(self, movies: List[Dict]):
        """
        영화 AI 평점 섹션 렌더링

        Args:
            movies: 영화 목록
        """
        st.header("📊 영화 AI 평점")

        # 영화 선택
        movie_options = {f"{m['title']}": m["id"] for m in movies}
        selected_movie_name = st.selectbox(
            "평점을 확인할 영화 선택",
            options=list(movie_options.keys()),
            key="rating_movie_select",
        )

        if selected_movie_name:
            movie_id = movie_options[selected_movie_name]
            rating_data = self._get_movie_rating(movie_id)

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

        st.plotly_chart(fig, use_container_width=True)

    def _render_review_list(self):
        """
        리뷰 목록 렌더링
        """
        st.header("최근 리뷰 목록")

        # 리뷰 개수 선택
        limit = st.slider(
            "표시할 리뷰 개수", min_value=5, max_value=50, value=10, step=5
        )

        try:
            response = requests.get(f"{self.api_url}/reviews/", params={"limit": limit})

            if response.status_code == 200:
                reviews = response.json()

                if not reviews:
                    st.info("📭 등록된 리뷰가 없습니다.")
                else:
                    st.write(f"총 {len(reviews)}개의 리뷰")

                    for review in reviews:
                        self._render_review_card(review)
            else:
                st.error("❌ 리뷰 목록을 불러오는데 실패했습니다.")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ API 연결 오류: {str(e)}")

    def _render_review_card(self, review: dict):
        """
        리뷰 카드 렌더링

        Args:
            review: 리뷰 정보 딕셔너리
        """
        with st.container(border=True):
            col1, col2 = st.columns([4, 1])

            with col1:
                # 영화 정보
                if "movie" in review and review["movie"]:
                    st.subheader(f"🎬 {review['movie']['title']}")
                else:
                    st.subheader(f"🎬 영화 ID: {review['movie_id']}")

                # 작성자
                st.caption(f"✍️ {review['author']}")

                # 리뷰 내용
                st.write(review["content"])

            with col2:
                # 감성 분석 결과
                if review.get("is_positive") is not None:
                    if review["is_positive"] == 1:
                        st.success("😊 긍정")
                    else:
                        st.error("😞 부정")
                else:
                    st.info("❓ 분석중")

                # 삭제 버튼
                if st.button("🗑️", key=f"delete_review_{review['id']}"):
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

    def _get_movie_rating(self, movie_id: int) -> Optional[Dict]:
        """
        영화 평점 가져오기

        Args:
            movie_id: 영화 ID

        Returns:
            평점 데이터 또는 None
        """
        try:
            response = requests.get(f"{self.api_url}/reviews/movie/{movie_id}/rating")
            if response.status_code == 200:
                return response.json()
        except requests.exceptions.RequestException:
            pass
        return None

    def _register_review(self, movie_id: int, author: str, content: str):
        """
        리뷰 등록

        Args:
            movie_id: 영화 ID
            author: 작성자
            content: 리뷰 내용
        """
        review_data = {"movie_id": movie_id, "author": author, "content": content}

        try:
            response = requests.post(f"{self.api_url}/reviews/", json=review_data)

            if response.status_code == 201:
                review = response.json()
                sentiment = "긍정 😊" if review.get("is_positive") == 1 else "부정 😞"
                st.success(f"리뷰가 등록되었습니다! (AI 분석 결과: {sentiment})")
                st.balloons()
                st.rerun()
            else:
                error_detail = response.json().get("detail", "알 수 없는 오류")
                st.error(f"❌ 리뷰 등록 실패: {error_detail}")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ API 연결 오류: {str(e)}")

    def _delete_review(self, review_id: int):
        """
        리뷰 삭제

        Args:
            review_id: 리뷰 ID
        """
        try:
            response = requests.delete(f"{self.api_url}/reviews/{review_id}")

            if response.status_code == 204:
                st.success("리뷰가 삭제되었습니다.")
                st.rerun()
            else:
                st.error("❌ 리뷰 삭제 실패")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ API 연결 오류: {str(e)}")


# 페이지 실행
if __name__ == "__main__":
    manager = ReviewManager()
    manager.render()
