"""
검색 UI 공통 모듈
영화 검색 및 리뷰 검색 UI 컴포넌트
"""

import streamlit as st
from typing import Dict, Optional, Tuple
import logging
from helper_dev_utils import get_auto_logger

logger = get_auto_logger(log_level=logging.DEBUG)


class MovieSearchUI:
    """
    영화 검색 UI 컴포넌트
    """

    def __init__(self, show_ai_rating: bool = True, show_sort_options: bool = True):
        """
        MovieSearchUI 초기화

        Args:
            show_ai_rating: AI 평점 슬라이더 표시 여부
            show_sort_options: 정렬 옵션 표시 여부
        """
        self.show_ai_rating = show_ai_rating
        self.show_sort_options = show_sort_options
        self.update_data = False

    def _sort_current_movies(self):
        """현재 검색된 영화 목록을 클라이언트 사이드에서 정렬"""
        logger.debug("[_sort_current_movies] 호출됨")

        sort_by = st.session_state.get("sort_by", "release_date")
        sort_order = st.session_state.get("sort_order", "desc")

        logger.debug(
            f"[_sort_current_movies] sort_by={sort_by}, sort_order={sort_order}"
        )

        # 정렬 키 매핑
        sort_key_map = {
            "release_date": lambda m: m.get("release_date") or "",
            "tmdb_rating": lambda m: m.get("tmdb_rating") or 0,
            "ai_rating": lambda m: m.get("ai_rating") or 0,
            "title": lambda m: m.get("title") or "",
        }

        reverse = sort_order == "desc"

        # searched_movies 정렬 (review_edit, review_list 등에서 사용)
        if "searched_movies" in st.session_state:
            movies = st.session_state.get("searched_movies", [])
            if movies and sort_by in sort_key_map:
                sorted_movies = sorted(
                    movies, key=sort_key_map[sort_by], reverse=reverse
                )
                st.session_state["searched_movies"] = sorted_movies
                logger.debug(
                    f"[_sort_current_movies] searched_movies 정렬 완료: {len(sorted_movies)}개"
                )

        # loaded_movies 정렬 (movie_list에서 사용)
        if "loaded_movies" in st.session_state:
            movies = st.session_state.get("loaded_movies", [])
            if movies and sort_by in sort_key_map:
                sorted_movies = sorted(
                    movies, key=sort_key_map[sort_by], reverse=reverse
                )
                st.session_state["loaded_movies"] = sorted_movies
                logger.debug(
                    f"[_sort_current_movies] loaded_movies 정렬 완료: {len(sorted_movies)}개"
                )
                if sorted_movies:
                    logger.debug(
                        f"[_sort_current_movies] 첫 영화: {sorted_movies[0].get('title')} ({sort_by}={sorted_movies[0].get(sort_by)})"
                    )
                    logger.debug(
                        f"[_sort_current_movies] 마지막 영화: {sorted_movies[-1].get('title')} ({sort_by}={sorted_movies[-1].get(sort_by)})"
                    )

        # 세션 상태 업데이트
        st.session_state["search_params"]["sort_by"] = sort_by
        st.session_state["search_params"]["sort_order"] = sort_order

        # URL 쿼리 파라미터에도 저장 (화면 갱신 시에도 유지)
        query_params = dict(st.query_params)
        query_params["sort_by"] = sort_by
        query_params["sort_order"] = sort_order
        st.query_params.update(query_params)
        logger.debug(f"[_sort_current_movies] URL 쿼리 파라미터 업데이트 완료")

    def _update_rating_params(self):
        """평점 범위 변경 시 URL 쿼리 파라미터 업데이트"""
        logger.debug("[_update_rating_params] 호출됨")

        # 현재 평점 범위 값 가져오기
        tmdb_range = st.session_state.get("tmdb_rating_range", (0.0, 10.0))

        # 세션 상태 업데이트
        st.session_state["search_params"]["tmdb_min"] = str(tmdb_range[0])
        st.session_state["search_params"]["tmdb_max"] = str(tmdb_range[1])

        # URL 쿼리 파라미터 업데이트 (clear 후 재설정)
        st.query_params.clear()

        # TMDB 평점 처리
        if tmdb_range[0] > 0:
            st.query_params["tmdb_min"] = str(tmdb_range[0])
        if tmdb_range[1] < 10:
            st.query_params["tmdb_max"] = str(tmdb_range[1])

        # AI 평점 처리
        if "ai_rating_range" in st.session_state:
            ai_range = st.session_state.get("ai_rating_range", (0.0, 10.0))
            st.session_state["search_params"]["ai_min"] = str(ai_range[0])
            st.session_state["search_params"]["ai_max"] = str(ai_range[1])

            if ai_range[0] > 0:
                st.query_params["ai_min"] = str(ai_range[0])
            if ai_range[1] < 10:
                st.query_params["ai_max"] = str(ai_range[1])

        # 정렬 파라미터 복원
        if "sort_by" in st.session_state.get("search_params", {}):
            st.query_params["sort_by"] = st.session_state["search_params"]["sort_by"]
        if "sort_order" in st.session_state.get("search_params", {}):
            st.query_params["sort_order"] = st.session_state["search_params"][
                "sort_order"
            ]

        # 검색 필터 복원 (있는 경우)
        for key in ["title", "director", "genre", "date_from", "date_to"]:
            value = st.session_state.get("search_params", {}).get(key, "")
            if value:
                st.query_params[key] = value
        logger.debug(f"[_update_rating_params] URL 쿼리 파라미터 업데이트 완료")
        self.update_data = True

    def render(self) -> Tuple[bool, Optional[Dict]]:
        """
        영화 검색 UI 렌더링

        Returns:
            (search_triggered, filters): 검색 실행 여부와 필터 딕셔너리
            - search_triggered: True(검색 버튼 클릭), False(초기화 버튼 클릭), None(버튼 미클릭)
            - filters: 검색 필터 딕셔너리 (None 값 제거됨)
        """
        # URL 쿼리 파라미터에서 검색 조건 로드
        query_params = st.query_params

        # 세션 상태에 검색 조건 저장 (다른 페이지 갔다 와도 유지)
        if "search_params" not in st.session_state:
            st.session_state["search_params"] = {}

        # 초기 로드 플래그: URL 파라미터를 세션에 로드했는지 확인
        if "search_params_loaded" not in st.session_state:
            st.session_state["search_params_loaded"] = False

        # URL 파라미터 자동 검색 트리거 플래그
        if "auto_search_trigger" not in st.session_state:
            st.session_state["auto_search_trigger"] = False

        # URL 쿼리 파라미터가 있고 아직 로드하지 않았으면 세션에 저장 (최초 1회만)
        param_keys = [
            "title",
            "director",
            "genre",
            "date_from",
            "date_to",
            "tmdb_min",
            "tmdb_max",
        ]
        if self.show_ai_rating:
            param_keys.extend(["ai_min", "ai_max"])
        if self.show_sort_options:
            param_keys.extend(["sort_by", "sort_order"])

        if not st.session_state["search_params_loaded"] and any(
            key in query_params for key in param_keys
        ):
            st.session_state["search_params"] = {
                "title": query_params.get("title", ""),
                "director": query_params.get("director", ""),
                "genre": query_params.get("genre", ""),
                "date_from": query_params.get("date_from", ""),
                "date_to": query_params.get("date_to", ""),
                "tmdb_min": query_params.get("tmdb_min", "0.0"),
                "tmdb_max": query_params.get("tmdb_max", "10.0"),
            }
            if self.show_ai_rating:
                st.session_state["search_params"]["ai_min"] = query_params.get(
                    "ai_min", "0.0"
                )
                st.session_state["search_params"]["ai_max"] = query_params.get(
                    "ai_max", "10.0"
                )
            if self.show_sort_options:
                st.session_state["search_params"]["sort_by"] = query_params.get(
                    "sort_by", "release_date"
                )
                st.session_state["search_params"]["sort_order"] = query_params.get(
                    "sort_order", "desc"
                )
            st.session_state["search_params_loaded"] = True
            # URL 파라미터가 있으면 자동으로 검색 트리거
            st.session_state["auto_search_trigger"] = True

        with st.expander("🔍 영화 검색", expanded=False):

            # 제목 및 감독 검색
            col1, col2 = st.columns(2)
            with col1:
                search_title = st.text_input(
                    "제목",
                    value=st.session_state["search_params"].get("title", ""),
                    key="search_title",
                    placeholder="예: 인셉션",
                )
            with col2:
                search_director = st.text_input(
                    "감독",
                    value=st.session_state["search_params"].get("director", ""),
                    key="search_director",
                    placeholder="예: 크리스토퍼 놀란",
                )

            # 장르 및 개봉일 검색
            col1, col2, col3 = st.columns(3)
            with col1:
                search_genre = st.text_input(
                    "장르",
                    value=st.session_state["search_params"].get("genre", ""),
                    key="search_genre",
                    placeholder="예: 액션",
                )
            with col2:
                search_date_from = st.text_input(
                    "개봉일 시작",
                    value=st.session_state["search_params"].get("date_from", ""),
                    key="search_date_from",
                    placeholder="YYYY-MM-DD",
                )
            with col3:
                search_date_to = st.text_input(
                    "개봉일 종료",
                    value=st.session_state["search_params"].get("date_to", ""),
                    key="search_date_to",
                    placeholder="YYYY-MM-DD",
                )

            # 평점 범위 검색
            if self.show_ai_rating:
                col1, col2 = st.columns(2)
            else:
                col1 = st.container()
                col2 = None

            with col1:
                st.write("TMDB 평점 범위")
                tmdb_rating_range = st.slider(
                    "TMDB 평점",
                    min_value=0.0,
                    max_value=10.0,
                    value=(
                        float(st.session_state["search_params"].get("tmdb_min", "0.0")),
                        float(
                            st.session_state["search_params"].get("tmdb_max", "10.0")
                        ),
                    ),
                    step=0.5,
                    key="tmdb_rating_range",
                    label_visibility="collapsed",
                    on_change=self._update_rating_params,
                )
                logger.debug(f"[MovieSearchUI] TMDB 평점 범위: {tmdb_rating_range}")

            if self.show_ai_rating and col2:
                with col2:
                    st.write("AI 평점 범위")
                    ai_rating_range = st.slider(
                        "AI 평점",
                        min_value=0.0,
                        max_value=10.0,
                        value=(
                            float(
                                st.session_state["search_params"].get("ai_min", "0.0")
                            ),
                            float(
                                st.session_state["search_params"].get("ai_max", "10.0")
                            ),
                        ),
                        step=0.1,
                        key="ai_rating_range",
                        label_visibility="collapsed",
                        on_change=self._update_rating_params,
                    )
                    logger.debug(f"[MovieSearchUI] AI 평점 범위: {ai_rating_range}")

            # 정렬 옵션
            if self.show_sort_options:
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
                        index=[
                            "release_date",
                            "tmdb_rating",
                            "ai_rating",
                            "title",
                        ].index(
                            st.session_state["search_params"].get(
                                "sort_by", "release_date"
                            )
                        ),
                        key="sort_by",
                        on_change=self._sort_current_movies,
                    )
                    logger.debug(f"[MovieSearchUI] 정렬 기준: {sort_by}")
                with col2:
                    sort_order = st.selectbox(
                        "정렬 방향",
                        options=["desc", "asc"],
                        format_func=lambda x: "내림차순" if x == "desc" else "오름차순",
                        index=["desc", "asc"].index(
                            st.session_state["search_params"].get("sort_order", "desc")
                        ),
                        key="sort_order",
                        on_change=self._sort_current_movies,
                    )
                    logger.debug(f"[MovieSearchUI] 정렬 방향: {sort_order}")

            # 검색 및 초기화 버튼
            col1, col2 = st.columns(2)
            with col1:
                search_button = st.button(
                    "🔍 검색",
                    type="primary",
                    key="btn_search",
                    use_container_width=True,
                )
            with col2:
                reset_button = st.button(
                    "🔄 초기화",
                    type="secondary",
                    key="btn_reset",
                    use_container_width=True,
                )

        # 버튼 클릭 처리
        if reset_button:
            # 초기화
            # 위젯 키 삭제 (리런 시 초기화된 값으로 재생성되도록)
            widget_keys = [
                "search_title",
                "search_director",
                "search_genre",
                "search_date_from",
                "search_date_to",
                "tmdb_rating_range",
                "ai_rating_range",
                "sort_by",
                "sort_order",
            ]
            for key in widget_keys:
                if key in st.session_state:
                    del st.session_state[key]

            st.query_params.clear()
            st.session_state["search_params"] = {}
            st.session_state["search_params_loaded"] = False
            return False, None

        # 자동 검색 트리거 확인 (URL 파라미터로 진입한 경우)
        auto_trigger = st.session_state.get("auto_search_trigger", False)
        if auto_trigger:
            st.session_state["auto_search_trigger"] = False
            logger.debug("[MovieSearchUI] URL 파라미터 자동 검색 트리거 활성화")

        logger.debug(
            f"[MovieSearchUI] search_button: {search_button}, update_data: {self.update_data}, auto_trigger: {auto_trigger}"
        )
        if search_button or self.update_data or auto_trigger:
            # 검색 필터 구성
            filters = {
                "title": search_title if search_title.strip() else None,
                "director": search_director if search_director.strip() else None,
                "genre": search_genre if search_genre.strip() else None,
                "release_date_from": (
                    search_date_from if search_date_from.strip() else None
                ),
                "release_date_to": search_date_to if search_date_to.strip() else None,
                "tmdb_rating_min": (
                    tmdb_rating_range[0] if tmdb_rating_range[0] > 0 else None
                ),
                "tmdb_rating_max": (
                    tmdb_rating_range[1] if tmdb_rating_range[1] < 10 else None
                ),
            }

            if self.show_ai_rating:
                filters["ai_rating_min"] = (
                    ai_rating_range[0] if ai_rating_range[0] > 0 else None
                )
                filters["ai_rating_max"] = (
                    ai_rating_range[1] if ai_rating_range[1] < 10 else None
                )

            if self.show_sort_options:
                filters["sort_by"] = sort_by
                filters["sort_order"] = sort_order

            # None 값 제거
            filters = {k: v for k, v in filters.items() if v is not None}

            # 세션 상태 및 URL 쿼리 파라미터 업데이트
            new_query_params = {}
            st.session_state["search_params"] = {
                "title": search_title,
                "director": search_director,
                "genre": search_genre,
                "date_from": search_date_from,
                "date_to": search_date_to,
                "tmdb_min": str(tmdb_rating_range[0]),
                "tmdb_max": str(tmdb_rating_range[1]),
            }

            if search_title:
                new_query_params["title"] = search_title
            if search_director:
                new_query_params["director"] = search_director
            if search_genre:
                new_query_params["genre"] = search_genre
            if search_date_from:
                new_query_params["date_from"] = search_date_from
            if search_date_to:
                new_query_params["date_to"] = search_date_to
            if tmdb_rating_range[0] > 0:
                new_query_params["tmdb_min"] = str(tmdb_rating_range[0])
            if tmdb_rating_range[1] < 10:
                new_query_params["tmdb_max"] = str(tmdb_rating_range[1])

            if self.show_ai_rating:
                st.session_state["search_params"]["ai_min"] = str(ai_rating_range[0])
                st.session_state["search_params"]["ai_max"] = str(ai_rating_range[1])
                if ai_rating_range[0] > 0:
                    new_query_params["ai_min"] = str(ai_rating_range[0])
                if ai_rating_range[1] < 10:
                    new_query_params["ai_max"] = str(ai_rating_range[1])

            if self.show_sort_options:
                st.session_state["search_params"]["sort_by"] = sort_by
                st.session_state["search_params"]["sort_order"] = sort_order
                new_query_params["sort_by"] = sort_by
                new_query_params["sort_order"] = sort_order

            st.query_params.update(new_query_params)

            return True, filters

        return None, None
