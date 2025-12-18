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
import sys
from pathlib import Path
from helper_dev_utils import get_auto_logger

# pages 디렉토리를 sys.path에 추가
pages_dir = Path(__file__).parent
if str(pages_dir) not in sys.path:
    sys.path.insert(0, str(pages_dir))

from movie_edit import MovieEditManager

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

        st.write("##### 🎬 영화 관리")

        # 탭 생성
        tab1, tab2 = st.tabs(["📝 영화 등록", "✏️ 영화 수정/삭제"])

        with tab1:
            self._render_movie_registration()

        with tab2:
            # MovieEditManager를 사용하여 수정/삭제 UI 렌더링
            movie_edit_manager = MovieEditManager()
            movie_edit_manager.render()

    def _render_movie_registration(self):
        """
        영화 등록 폼 렌더링
        """
        # st.write("영화 등록")
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

                title = st_text_input(
                    "영화 제목 *", placeholder="예: 인터스텔라", value="제목"
                )
                release_date = st.text_input(
                    "개봉일",
                    placeholder="예: 2014-11-26 (YYYY-MM-DD)",
                    value="2025-12-17",
                )
                director = st_text_input(
                    "감독", placeholder="예: 크리스토퍼 놀란", value="감독"
                )

            with col2:
                genre = st_text_input(
                    "장르", placeholder="예: SF, 드라마", value="장르"
                )
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
                    value=5.0,
                )

            # 새 필드 추가 (TMDB API 필드)
            st.write("#### 추가 정보 (선택사항)")

            col3, col4 = st.columns(2)
            with col3:
                overview = st.text_area(
                    "줄거리",
                    placeholder="영화 줄거리를 입력하세요",
                    value="" if not TEST_MODE else "테스트 줄거리",
                    height=100,
                )
                original_title = st_text_input(
                    "원제 (Original Title)", placeholder="예: Interstellar", value=""
                )
                original_language = st.text_input(
                    "원어 (Original Language)",
                    placeholder="예: en (ISO 639-1 코드)",
                    value="",
                )
                adult = st.checkbox("성인 영화", value=False)

            with col4:
                popularity = st.number_input(
                    "인기도 (Popularity)",
                    min_value=0.0,
                    step=0.1,
                    value=0.0,
                    help="TMDB 인기도 지수",
                )
                vote_count = st.number_input(
                    "투표 수 (Vote Count)", min_value=0, value=0, help="TMDB 투표 수"
                )
                backdrop_path = st.text_input(
                    "배경 이미지 URL", placeholder="TMDB 배경 이미지 URL", value=""
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
                        "overview": overview if overview else None,
                        "popularity": float(popularity) if popularity > 0 else None,
                        "vote_count": int(vote_count) if vote_count > 0 else None,
                        "original_title": original_title if original_title else None,
                        "original_language": (
                            original_language if original_language else None
                        ),
                        "adult": adult,
                        "backdrop_path": backdrop_path if backdrop_path else None,
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
        overview: Optional[str] = None,
        popularity: Optional[float] = None,
        vote_count: Optional[int] = None,
        original_title: Optional[str] = None,
        original_language: Optional[str] = None,
        adult: Optional[bool] = None,
        backdrop_path: Optional[str] = None,
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
            overview: 영화 줄거리
            popularity: 인기도
            vote_count: 투표 수
            original_title: 원제
            original_language: 원어
            adult: 성인 영화 여부
            backdrop_path: 배경 이미지 URL
        """
        movie_data = {
            "tmdb_id": tmdb_id,
            "title": title,
            "release_date": release_date,
            "director": director,
            "genre": genre,
            "poster_url": poster_url,
            "tmdb_rating": tmdb_rating,
            "overview": overview,
            "popularity": popularity,
            "vote_count": vote_count,
            "original_title": original_title,
            "original_language": original_language,
            "adult": adult,
            "backdrop_path": backdrop_path,
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
