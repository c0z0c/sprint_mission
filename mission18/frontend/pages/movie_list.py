"""
영화 목록 페이지 - TMDB 스타일 대시보드
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

logger.debug(f"API_BASE_URL: {API_BASE_URL}")


# CSS 스타일링
st.markdown(
    """
    <style>
    .movie-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s;
        background-color: white;
    }
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .movie-title {
        font-size: 1.1rem;
        font-weight: bold;
        margin: 10px 0 5px 0;
        color: #1f1f1f;
    }
    .movie-date {
        font-size: 0.85rem;
        color: #666;
        margin-bottom: 8px;
    }
    .rating-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 5px;
    }
    .tmdb-rating {
        background-color: #01b4e4;
        color: white;
    }
    .ai-rating {
        background-color: #21d07a;
        color: white;
    }
    .positive-badge {
        background-color: #d4edda;
        color: #155724;
    }
    .negative-badge {
        background-color: #f8d7da;
        color: #721c24;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


class MovieListManager:
    """
    영화 목록 관리 클래스 - TMDB 스타일 대시보드
    """

    def __init__(self):
        """MovieListManager 초기화"""
        self.api_url = API_BASE_URL

    def render(self):
        """영화 목록 페이지 렌더링"""
        st.write("###### 🎬 영화 목록")

        # 세션 상태 초기화 (무한 스크롤 방식)
        if "loaded_movies" not in st.session_state:
            st.session_state["loaded_movies"] = []
            st.session_state["current_page"] = 1
            st.session_state["has_more"] = True
        if "page_size" not in st.session_state:
            st.session_state["page_size"] = 8
        if "search_mode" not in st.session_state:
            st.session_state["search_mode"] = False
        if "search_filters" not in st.session_state:
            st.session_state["search_filters"] = {}

        # 검색 UI 렌더링
        self._render_search_ui()

        st_div_divider()

        # 초기 로드: 첫 페이지 자동 로드 (검색 모드가 아닐 때만)
        if (
            not st.session_state["loaded_movies"]
            and st.session_state["has_more"]
            and not st.session_state["search_mode"]
        ):
            self._load_more_movies()

        # 로드된 영화가 없으면 안내 메시지
        if not st.session_state["loaded_movies"]:
            if st.session_state["search_mode"]:
                st.info("🔍 검색 결과가 없습니다.")
            else:
                st.info(
                    "📭 등록된 영화가 없습니다. 영화 관리 페이지에서 영화를 등록해주세요."
                )
            return

        # 로드된 영화 수 표시
        cols = st.columns([1.5, 1.5, 8])
        with cols[0]:
            if st.button("🔄 새로고침", type="secondary", width="stretch"):
                st.session_state["loaded_movies"] = []
                st.session_state["current_page"] = 1
                st.session_state["has_more"] = True
                st.session_state["search_mode"] = False
                st.session_state["search_filters"] = {}
                self._load_more_movies()
                st.rerun()

        with cols[1]:
            st_label(f"{len(st.session_state['loaded_movies'])}개의 영화")

        st_div_divider()

        # 그리드 레이아웃 (한 줄에 4개) - 누적된 모든 영화 표시
        movies = st.session_state["loaded_movies"]
        cols_per_row = 4
        for i in range(0, len(movies), cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if i + j < len(movies):
                    with cols[j]:
                        self._render_movie_card(movies[i + j])

        # "더 불러오기" 버튼
        if st.session_state["has_more"]:
            _, col2, _ = st.columns([1, 8, 1])
            with col2:
                if st.button("📥 더 불러오기", width="stretch", type="primary"):
                    self._load_more_movies()
                    st.rerun()
        else:
            st.info("✅ 모든 영화를 불러왔습니다.")

    def _render_movie_card(self, movie: Dict):
        """
        영화 카드 렌더링

        Args:
            movie: 영화 정보 딕셔너리
        """
        with st.container():
            # 포스터 이미지
            if movie.get("poster_local_path"):
                poster_url = f"{self.api_url}/{movie['poster_local_path']}"
                st.image(poster_url, width="content")
            else:
                st.image(
                    "https://via.placeholder.com/300x450?text=No+Poster",
                    width="content",
                )

            # 영화 제목
            st.markdown(
                f"<div class='movie-title'>{movie['title']}</div>",
                unsafe_allow_html=True,
            )

            # 원제 표시 (있는 경우)
            if (
                movie.get("original_title")
                and movie.get("original_title") != movie["title"]
            ):
                st.caption(f"🌐 원제: {movie['original_title']}")

            # 개봉일
            if movie.get("release_date"):
                st.markdown(
                    f"<div class='movie-date'>📅 {movie['release_date']}</div>",
                    unsafe_allow_html=True,
                )

            # 감독 및 장르
            info_items = []
            if movie.get("director"):
                info_items.append(f"🎬 {movie['director']}")
            if movie.get("genre"):
                info_items.append(f"🎭 {movie['genre']}")

            if info_items:
                st.caption(" | ".join(info_items))

            # 인기도 및 투표 수 표시
            if movie.get("popularity") or movie.get("vote_count"):
                metrics_items = []
                if movie.get("popularity"):
                    metrics_items.append(f"🔥 인기도: {movie['popularity']:.1f}")
                if movie.get("vote_count"):
                    metrics_items.append(f"👥 투표: {movie['vote_count']:,}명")
                if metrics_items:
                    st.caption(" | ".join(metrics_items))

            # 평점 정보
            rating_html = ""
            if movie.get("tmdb_rating"):
                rating_html += f"<span class='rating-badge tmdb-rating'>⭐ TMDB {movie['tmdb_rating']}/10</span>"

            # AI 평점 (페이지네이션 응답에 포함됨)
            if movie.get("total_reviews", 0) > 0:
                ai_rating = movie.get("ai_rating", 0.0)
                rating_html += (
                    f"<span class='rating-badge ai-rating'>🤖 AI {ai_rating}/5.0</span>"
                )

            if rating_html:
                st.markdown(rating_html, unsafe_allow_html=True)

            # 줄거리 표시 (있는 경우)
            if movie.get("overview"):
                with st.expander("📖 줄거리"):
                    st.write(movie["overview"])

            # st.markdown("<br>", unsafe_allow_html=True)

            # 리뷰 정보
            reviews = movie.get("reviews", [])
            if reviews:
                with st.expander(f"📝 리뷰 보기 ({len(reviews)}개)"):
                    for review in reviews:
                        self._render_review_item(review)
            else:
                st.caption("💬 아직 리뷰가 없습니다.")

    def _render_review_item(self, review: Dict):
        """
        리뷰 항목 렌더링

        Args:
            review: 리뷰 정보 딕셔너리
        """
        # 감성 분석 결과 배지
        sentiment_badge = ""
        if review.get("is_positive") is not None:
            if review["is_positive"] == 1:
                sentiment_badge = (
                    "<span class='rating-badge positive-badge'>😊 긍정</span>"
                )
            else:
                sentiment_badge = (
                    "<span class='rating-badge negative-badge'>😢 부정</span>"
                )

        st.markdown(
            f"""
            <div style='margin-bottom: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;'>
                <div style='margin-bottom: 5px;'>
                    <strong>✍️ {review['author']}</strong> {sentiment_badge}
                </div>
                <div style='color: #495057; font-size: 0.9rem;'>
                    {review['content']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def _render_search_ui(self):
        """영화 검색 UI 렌더링"""
        with st.expander("🔍 영화 검색", expanded=False):
            # 제목 및 감독 검색
            col1, col2 = st.columns(2)
            with col1:
                search_title = st.text_input(
                    "제목",
                    key="search_title_list",
                    placeholder="예: 인셉션",
                )
            with col2:
                search_director = st.text_input(
                    "감독",
                    key="search_director_list",
                    placeholder="예: 크리스토퍼 놀란",
                )

            # 장르 및 개봉일 검색
            col1, col2, col3 = st.columns(3)
            with col1:
                search_genre = st.text_input(
                    "장르",
                    key="search_genre_list",
                    placeholder="예: 액션",
                )
            with col2:
                search_date_from = st.text_input(
                    "개봉일 시작",
                    key="search_date_from_list",
                    placeholder="YYYY-MM-DD",
                )
            with col3:
                search_date_to = st.text_input(
                    "개봉일 종료",
                    key="search_date_to_list",
                    placeholder="YYYY-MM-DD",
                )

            # 평점 범위 검색
            col1, col2 = st.columns(2)
            with col1:
                st.write("TMDB 평점 범위")
                tmdb_rating_range = st.slider(
                    "TMDB 평점",
                    min_value=0.0,
                    max_value=10.0,
                    value=(0.0, 10.0),
                    step=0.5,
                    key="tmdb_rating_range_list",
                    label_visibility="collapsed",
                )
            with col2:
                st.write("AI 평점 범위")
                ai_rating_range = st.slider(
                    "AI 평점",
                    min_value=0.0,
                    max_value=5.0,
                    value=(0.0, 5.0),
                    step=0.1,
                    key="ai_rating_range_list",
                    label_visibility="collapsed",
                )

            # 정렬 옵션
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                sort_by = st.selectbox(
                    "정렬 기준",
                    options=["release_date", "tmdb_rating", "ai_rating", "title"],
                    format_func=lambda x: {
                        "release_date": "개봉일",
                        "tmdb_rating": "TMDB 평점",
                        "ai_rating": "AI 평점",
                        "title": "제목",
                    }[x],
                    key="sort_by_list",
                )
            with col2:
                sort_order = st.selectbox(
                    "정렬 방향",
                    options=["desc", "asc"],
                    format_func=lambda x: "내림차순" if x == "desc" else "오름차순",
                    key="sort_order_list",
                )

            # 검색 및 초기화 버튼
            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                    "🔍 검색",
                    type="primary",
                    key="btn_search_list",
                    use_container_width=True,
                ):
                    filters = {
                        "title": search_title if search_title.strip() else None,
                        "director": (
                            search_director if search_director.strip() else None
                        ),
                        "genre": search_genre if search_genre.strip() else None,
                        "release_date_from": (
                            search_date_from if search_date_from.strip() else None
                        ),
                        "release_date_to": (
                            search_date_to if search_date_to.strip() else None
                        ),
                        "tmdb_rating_min": (
                            tmdb_rating_range[0] if tmdb_rating_range[0] > 0 else None
                        ),
                        "tmdb_rating_max": (
                            tmdb_rating_range[1] if tmdb_rating_range[1] < 10 else None
                        ),
                        "ai_rating_min": (
                            ai_rating_range[0] if ai_rating_range[0] > 0 else None
                        ),
                        "ai_rating_max": (
                            ai_rating_range[1] if ai_rating_range[1] < 5 else None
                        ),
                        "sort_by": sort_by,
                        "sort_order": sort_order,
                    }

                    # None 값 제거
                    filters = {k: v for k, v in filters.items() if v is not None}

                    st.session_state["search_filters"] = filters
                    st.session_state["search_mode"] = True
                    st.session_state["loaded_movies"] = []
                    st.session_state["current_page"] = 1
                    st.session_state["has_more"] = True
                    self._load_more_movies()
                    st.rerun()

            with col2:
                if st.button(
                    "🔄 초기화",
                    type="secondary",
                    key="btn_reset_list",
                    use_container_width=True,
                ):
                    st.session_state["search_mode"] = False
                    st.session_state["search_filters"] = {}
                    st.session_state["loaded_movies"] = []
                    st.session_state["current_page"] = 1
                    st.session_state["has_more"] = True

                    # 검색 필드 초기화
                    for key in [
                        "search_title_list",
                        "search_director_list",
                        "search_genre_list",
                        "search_date_from_list",
                        "search_date_to_list",
                        "tmdb_rating_range_list",
                        "ai_rating_range_list",
                        "sort_by_list",
                        "sort_order_list",
                    ]:
                        if key in st.session_state:
                            del st.session_state[key]

                    st.rerun()

    def _load_more_movies(self):
        """
        다음 페이지의 영화를 로드하여 누적 목록에 추가
        """
        if st.session_state["search_mode"]:
            # 검색 모드: search API 사용
            pagination_data = self._search_movies_paginated(
                st.session_state["current_page"],
                st.session_state["page_size"],
                st.session_state["search_filters"],
            )
        else:
            # 일반 모드: 기본 paginated API 사용
            pagination_data = self._get_movies_paginated(
                st.session_state["current_page"], st.session_state["page_size"]
            )

        if pagination_data:
            movies = pagination_data.get("movies", [])
            total_pages = pagination_data.get("total_pages", 0)

            if movies:
                # 기존 목록에 새 영화 추가
                st.session_state["loaded_movies"].extend(movies)
                st.session_state["current_page"] += 1

                # 더 이상 로드할 페이지가 없는지 확인
                if st.session_state["current_page"] > total_pages:
                    st.session_state["has_more"] = False
            else:
                st.session_state["has_more"] = False
        else:
            st.session_state["has_more"] = False

    def _search_movies_paginated(
        self, page: int, page_size: int, filters: dict
    ) -> Optional[Dict]:
        """
        검색된 영화 목록 가져오기 (페이지네이션)

        Args:
            page: 페이지 번호
            page_size: 페이지당 항목 수
            filters: 검색 필터

        Returns:
            페이지네이션 데이터 (영화 목록, 전체 개수 등)
        """
        try:
            params = {
                "page": page,
                "page_size": page_size,
                **filters,
            }
            response = requests.get(
                f"{self.api_url}/movies/search",
                params=params,
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(
                    f"Failed to search movies: {response.status_code} - {response.text}"
                )
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to search movies: {str(e)}")
        return None

    def _get_movies_paginated(self, page: int, page_size: int) -> Optional[Dict]:
        """
        페이지네이션된 영화 목록 가져오기 (리뷰 포함)

        Args:
            page: 페이지 번호
            page_size: 페이지당 항목 수

        Returns:
            페이지네이션 데이터 (영화 목록, 전체 개수 등)
        """
        try:
            response = requests.get(
                f"{self.api_url}/movies/paginated",
                params={"page": page, "page_size": page_size},
            )
            if response.status_code == 200:
                return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch paginated movies: {str(e)}")
        return None

    def _get_movies_with_reviews(self) -> List[Dict]:
        """
        리뷰 정보를 포함한 영화 목록 가져오기 (deprecated - 페이지네이션 사용 권장)

        Returns:
            영화 목록 (리뷰 포함)
        """
        try:
            response = requests.get(f"{self.api_url}/movies/")
            if response.status_code == 200:
                movies = response.json()

                # 각 영화의 상세 정보 (리뷰 포함) 가져오기
                movies_with_reviews = []
                for movie in movies:
                    detail_response = requests.get(
                        f"{self.api_url}/movies/{movie['id']}"
                    )
                    if detail_response.status_code == 200:
                        movies_with_reviews.append(detail_response.json())
                    else:
                        movies_with_reviews.append(movie)

                return movies_with_reviews
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch movies: {str(e)}")
        return []

    def _get_movie_rating(self, tmdb_id: int) -> Optional[Dict]:
        """
        영화 평점 가져오기

        Args:
            tmdb_id: TMDB 영화 ID

        Returns:
            평점 데이터 또는 None
        """
        try:
            response = requests.get(f"{self.api_url}/reviews/movie/{tmdb_id}/rating")
            if response.status_code == 200:
                return response.json()
        except requests.exceptions.RequestException as e:
            logger.debug(f"No rating data for movie {tmdb_id}: {str(e)}")
        return None


# 페이지 실행
if __name__ == "__main__":
    manager = MovieListManager()
    manager.render()
