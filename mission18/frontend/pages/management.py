"""
영화 관리 페이지
"""

import streamlit as st
import requests
from typing import Optional
import os
import random
import logging
from helper_dev_utils import get_auto_logger

logger = get_auto_logger(log_level=logging.DEBUG)


# 백엔드 API URL
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
TEST_MODE = os.getenv("TEST_MODE", "False") == "True"

logger.debug(f"API_BASE_URL: {API_BASE_URL}")
logger.debug(f"TEST_MODE: {TEST_MODE}")


def random_text(text: str) -> str:
    """텍스트에 순차적인 한글 자음 추가"""
    # 한글 자음: 가, 나, 다, 라, 마, 바, 사, 아, 자, 차, 카, 타, 파, 하
    consonants = [
        "가",
        "나",
        "다",
        "라",
        "마",
        "바",
        "사",
        "아",
        "자",
        "차",
        "카",
        "타",
        "파",
        "하",
    ]

    # 세션 상태에 카운터가 없으면 초기화
    if "text_counter" not in st.session_state:
        st.session_state["text_counter"] = 0

    # 현재 카운터에 해당하는 자음 선택
    consonant = consonants[st.session_state["text_counter"] % len(consonants)]
    st.session_state["text_counter"] += 1

    result = text + consonant
    logger.debug(f"Generated sequential text: {result}")
    return result


def st_text_input(label, **kwargs):
    """테스트 모드에서 랜덤 한글 글자 추가된 텍스트 입력"""
    if TEST_MODE and "value" in kwargs:
        kwargs["value"] = random_text(kwargs["value"])
    if TEST_MODE and not "value" in kwargs:
        kwargs["value"] = random_text("테스트_")
    return st.text_input(label, **kwargs)


class MovieManager:
    """
    영화 관리 클래스 - 영화 정보 입력 및 포스터 업로드
    """

    def __init__(self):
        """
        MovieManager 초기화
        """
        self.api_url = API_BASE_URL

    def render(self):
        """
        영화 관리 페이지 렌더링
        """
        st.title("🎬 영화 관리")
        st.write("영화 정보를 등록하고 관리할 수 있습니다.")

        # 탭 구성
        tab1, tab2 = st.tabs(["영화 등록", "영화 목록"])

        with tab1:
            self._render_movie_registration()

        with tab2:
            self._render_movie_list()

    def _render_movie_registration(self):
        """
        영화 등록 폼 렌더링
        """
        st.header("영화 등록")

        with st.form("movie_registration_form"):
            col1, col2 = st.columns(2)

            with col1:
                test_counter = st.session_state.get("test_counter", 0)

                tmdb_id = st.number_input(
                    "TMDB ID *",
                    min_value=(test_counter + 1),
                    help="The Movie Database (TMDB)의 영화 ID",
                )
                st.session_state["test_counter"] = tmdb_id

                title = st_text_input("영화 제목 *", placeholder="예: 인터스텔라")
                release_date = st.text_input(
                    "개봉일",
                    placeholder="예: 2014-11-26 (YYYY-MM-DD)",
                    value="2025-12-17",
                )
                director = st_text_input("감독", placeholder="예: 크리스토퍼 놀란")

            with col2:
                genre = st_text_input("장르", placeholder="예: SF, 드라마")
                test_value = "https://media.themoviedb.org/t/p/w440_and_h660_face/klfSEbFOquMFjBQJ5uKAfp0rrsK.jpg"
                poster_url = st.text_input(
                    "포스터 URL",
                    placeholder="이미지 URL을 입력하세요",
                    value=test_value if TEST_MODE else "",
                )
                tmdb_rating = st.number_input(
                    "TMDB 평점",
                    min_value=0.0,
                    max_value=10.0,
                    step=0.1,
                    value=0.0,
                )

            submitted = st.form_submit_button("영화 등록", width="content")

            if submitted:
                if not tmdb_id or not title:
                    st.error("TMDB ID와 영화 제목은 필수 입력 항목입니다.")
                else:
                    self._register_movie(
                        tmdb_id=int(tmdb_id),
                        title=title,
                        release_date=release_date if release_date else None,
                        director=director if director else None,
                        genre=genre if genre else None,
                        poster_url=poster_url if poster_url else None,
                        tmdb_rating=float(tmdb_rating) if tmdb_rating > 0 else None,
                    )

    def _register_movie(
        self,
        tmdb_id: int,
        title: str,
        release_date: Optional[str] = None,
        director: Optional[str] = None,
        genre: Optional[str] = None,
        poster_url: Optional[str] = None,
        tmdb_rating: Optional[float] = None,
    ):
        """
        영화 등록 API 호출

        Args:
            tmdb_id: TMDB 영화 ID
            title: 영화 제목
            release_date: 개봉일
            director: 감독
            genre: 장르
            poster_url: 포스터 URL
            tmdb_rating: TMDB 평점
        """
        movie_data = {
            "tmdb_id": tmdb_id,
            "title": title,
            "release_date": release_date,
            "director": director,
            "genre": genre,
            "poster_url": poster_url,
            "tmdb_rating": tmdb_rating,
        }

        try:
            response = requests.post(f"{self.api_url}/movies/", json=movie_data)

            if response.status_code == 201:
                st.success(f"영화 '{title}'이(가) 성공적으로 등록되었습니다!")
                st.balloons()
                st.rerun()
            elif response.status_code == 400:
                # 중복 등록 등의 잘못된 요청
                error_detail = response.json().get("detail", "잘못된 요청입니다.")
                st.error(f"{error_detail}")
                logger.warning(
                    f"Movie registration failed - Bad Request: {error_detail}"
                )
            else:
                # 기타 서버 오류
                error_detail = response.json().get("detail", "알 수 없는 오류")
                st.error(f"영화 등록 실패: {error_detail}")
                logger.error(
                    f"Movie registration failed - Status {response.status_code}: {error_detail}"
                )
        except requests.exceptions.RequestException as e:
            st.error(f"🔌 API 연결 오류: {str(e)}")
            logger.error(f"API connection error: {str(e)}")

    def _render_movie_list(self):
        """
        영화 목록 렌더링
        """
        st.header("등록된 영화 목록")

        try:
            response = requests.get(f"{self.api_url}/movies/")

            if response.status_code == 200:
                movies = response.json()

                if not movies:
                    st.info("📭 등록된 영화가 없습니다.")
                else:
                    st.write(f"총 {len(movies)}개의 영화가 등록되어 있습니다.")

                    # 그리드 레이아웃으로 영화 카드 표시
                    cols = st.columns(3)
                    for idx, movie in enumerate(movies):
                        with cols[idx % 3]:
                            self._render_movie_card(movie)
            else:
                st.error("영화 목록을 불러오는데 실패했습니다.")
        except requests.exceptions.RequestException as e:
            st.error(f"API 연결 오류: {str(e)}")

    def _render_movie_card(self, movie: dict):
        """
        영화 카드 렌더링

        Args:
            movie: 영화 정보 딕셔너리
        """
        with st.container(border=True):
            # 포스터 이미지
            if movie.get("poster_local_path"):
                poster_url = f"{self.api_url}/static/{movie['poster_local_path'].replace('static/', '')}"
                st.image(poster_url, width="content")
            else:
                st.image(
                    "https://via.placeholder.com/300x450?text=No+Poster",
                    width="content",
                )

            # 영화 정보
            st.subheader(movie["title"])
            st.caption(f"TMDB ID: {movie['tmdb_id']}")

            if movie.get("director"):
                st.write(f"🎬 감독: {movie['director']}")
            if movie.get("genre"):
                st.write(f"🎭 장르: {movie['genre']}")
            if movie.get("release_date"):
                st.write(f"📅 개봉일: {movie['release_date']}")
            if movie.get("tmdb_rating"):
                st.write(f"⭐ 평점: {movie['tmdb_rating']}/10")

            # 삭제 버튼
            if st.button(
                "🗑️ 삭제",
                key=f"delete_{movie['id']}",
                width="content",
            ):
                self._delete_movie(movie["id"], movie["title"])

    def _delete_movie(self, movie_id: int, title: str):
        """
        영화 삭제

        Args:
            movie_id: 영화 ID
            title: 영화 제목
        """
        try:
            response = requests.delete(f"{self.api_url}/movies/{movie_id}")

            if response.status_code == 204:
                st.success(f"영화 '{title}'이(가) 삭제되었습니다.")
                st.rerun()
            else:
                st.error(f"영화 삭제 실패")
        except requests.exceptions.RequestException as e:
            st.error(f"API 연결 오류: {str(e)}")


# 페이지 실행
if __name__ == "__main__":
    manager = MovieManager()
    manager.render()
