"""
영화 관리 페이지
"""

import streamlit as st
import requests
from typing import Optional
import os
import random
import logging
import time
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

    def _get_max_tmdb_id(self) -> int:
        """
        서버에 저장된 최대 TMDB ID 조회 (효율적인 API 사용)

        Returns:
            int: 최대 TMDB ID (영화가 없으면 0)
        """
        try:
            response = requests.get(f"{self.api_url}/movies/max-tmdb-id", timeout=3)
            if response.status_code == 200:
                data = response.json()
                return data.get("max_tmdb_id", 0)
        except Exception as e:
            logger.warning(f"Failed to get max TMDB ID: {e}")
        return 0

    def render(self):
        """
        영화 관리 페이지 렌더링
        """
        logger.debug(f"영화 관리 페이지 렌더링 시작")

        st.title("🎬 영화 관리")

        # 영화 등록만 표시 (목록은 메인 페이지에 있음)
        self._render_movie_registration()

    def _render_movie_registration(self):
        """
        영화 등록 폼 렌더링
        """
        st.header("영화 등록")
        with st.form("movie_registration_form"):
            col1, col2 = st.columns(2)

            with col1:
                # 서버에서 최대 TMDB ID 가져오기
                max_tmdb_id = self._get_max_tmdb_id()
                next_tmdb_id = max_tmdb_id + 1

                tmdb_id = st.number_input(
                    "TMDB ID *",
                    min_value=1,
                    value=next_tmdb_id,
                    help=f"The Movie Database (TMDB)의 영화 ID (현재 최대값: {max_tmdb_id})",
                )

                title = st_text_input("영화 제목 *", placeholder="예: 인터스텔라")
                release_date = st.text_input(
                    "개봉일",
                    placeholder="예: 2014-11-26 (YYYY-MM-DD)",
                    value="2025-12-17",
                )
                director = st_text_input("감독", placeholder="예: 크리스토퍼 놀란")

            with col2:
                genre = st_text_input("장르", placeholder="예: SF, 드라마")
                test_value = "https://media.themoviedb.org/t/p/w440_and_h660_face/aEyqU9xvpT1ewVfutj6ctEX1sjq.jpg"
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

            # 버튼 비활성화는 무조건 3초후에 풀린다.
            if st.form_submit_button("영화 등록", width="content"):
                if not tmdb_id or not title:
                    st.error("TMDB ID와 영화 제목은 필수 입력 항목입니다.")
                else:

                    # 등록할 영화 데이터를 session_state에 저장
                    movie_data = {
                        "tmdb_id": int(tmdb_id),
                        "title": title,
                        "release_date": release_date if release_date else None,
                        "director": director if director else None,
                        "genre": genre if genre else None,
                        "poster_url": poster_url if poster_url else None,
                        "tmdb_rating": float(tmdb_rating) if tmdb_rating > 0 else None,
                    }
                    self._register_movie(**movie_data)
                    logger.debug("등록 완료")

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

        logger.debug(f"Sending POST request to {self.api_url}/movies/")
        request_start = time.time()

        try:
            response = requests.post(
                f"{self.api_url}/movies/", json=movie_data, timeout=3  # 3초 타임아웃
            )
            request_elapsed = time.time() - request_start
            logger.debug(
                f"API request completed in {request_elapsed:.2f} seconds (status: {response.status_code})"
            )

            if response.status_code == 201:
                # rerun 제거: 성공 메시지만 표시하고 다음 페이지 전환 시 자동 업데이트
                st.success(
                    f"✅ 영화 '{title}'이(가) 성공적으로 등록되었습니다! (등록 시간: {request_elapsed:.2f}초)"
                )
                # st.balloons()
                # st.rerun() 제거 - 페이지 전환 시 자동으로 목록이 업데이트됨
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
        except requests.exceptions.Timeout:
            st.error("⏱️ 요청 시간 초과: 서버 응답이 3초를 초과했습니다.")
            logger.error("API request timeout after 3 seconds")
        except requests.exceptions.RequestException as e:
            st.error(f"🔌 API 연결 오류: {str(e)}")
            logger.error(f"API connection error: {str(e)}")


# 페이지 실행
if __name__ == "__main__":
    manager = MovieManager()
    manager.render()
