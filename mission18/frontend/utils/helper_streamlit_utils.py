import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, Union
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
from helper_dev_utils import get_auto_logger

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
logger = get_auto_logger(log_level=logging.DEBUG)

__all__ = [
    "st_label",
    "st_style_page_margin_hidden",
    "st_style_toolbar_hidden",
    "st_div_divider",
    "st_style_page_margin",
    "st_settings_panel_show",
    "st_query_param_get",
    "st_query_param_set",
]


# Streamlit text_input과 동일한 기본 스타일
DEFAULT_LABEL_STYLE: Dict[str, str] = {
    "border": "0px solid #FFFFFF",
    "border-radius": "8px",
    "padding": "8px 12px",
    "background-color": "#FFFFFF",
    "color": "#31333F",
    "height": "38px",
    "line-height": "22px",
    "font-size": "14px",
    "display": "flex",
    "align-items": "center",
    "justify-content": "center",
    "text-align": "center",
}


def st_label(text: Union[str, int], **kwargs: str) -> int:
    """Streamlit text_input과 동일한 스타일의 커스텀 라벨을 렌더링합니다.

    기본 스타일은 Streamlit의 text_input 위젯과 동일하며,
    **kwargs를 통해 원하는 CSS 속성을 자유롭게 오버라이드할 수 있습니다.

    Args:
        text: 문자열이면 새 라벨 생성, 정수면 해당 ID의 라벨 반환 (업데이트용)
        **kwargs: CSS 속성을 snake_case로 전달 (자동으로 kebab-case로 변환됨)
                 업데이트 시 'value' 키워드로 새 텍스트 전달
                 예: text_align="center", background_color="#f0f0f0"

    Returns:
        int: 라벨의 고유 ID (타임스탬프 기반)

    Examples:
        >>> # 새 라벨 생성 (ID 반환)
        >>> label_id = st_label("Hello World")

        >>> # 해당 ID의 라벨 업데이트
        >>> st_label(label_id, value="Updated Text", color="red")

        >>> # 한 번에 생성 및 스타일링
        >>> my_label = st_label("Count: 0", color="blue")
        >>> # 나중에 업데이트
        >>> st_label(my_label, value="Count: 10")

        >>> # 배경색, 폰트 크기, 높이 변경
        >>> st_label(
        ...     "Custom Label",
        ...     background_color="#e3f2fd",
        ...     font_size="16px",
        ...     height="50px",
        ...     font_weight="bold"
        ... )

    Notes:
        - text가 str이면 새 라벨 생성하고 ID 반환
        - text가 int면 해당 ID의 라벨 참조 (value로 텍스트 업데이트)
        - snake_case 키는 자동으로 kebab-case CSS로 변환됩니다
          (예: text_align → text-align, background_color → background-color)
        - 기본 스타일: Streamlit 1.31.0 text_input 기반 (높이 38px, 폰트 14px 등)
        - 고유 ID는 타임스탬프(년월일시분초밀리초)로 생성됩니다
    """
    # 기본 스타일 복사
    merged_style = DEFAULT_LABEL_STYLE.copy()

    # kwargs에서 value 추출
    value = kwargs.pop("value", None)

    # kwargs를 kebab-case로 변환하여 병합
    for k, v in kwargs.items():
        css_key = k.replace("_", "-")
        merged_style[css_key] = v

    # CSS 문자열 생성
    style_string = "; ".join(f"{k}: {v}" for k, v in merged_style.items())

    if isinstance(text, int):
        # 정수면 업데이트 모드
        label_id = text
        if value is not None:
            components.html(
                f"""
                <script>
                    const targetDoc = window.parent.document;
                    const label = targetDoc.getElementById('st_label_{label_id}');
                    if (label) {{
                        label.innerHTML = '{value}';
                        label.setAttribute('style', '{style_string}');
                    }}
                </script>
                """,
                height=0,
            )
        return label_id
    else:
        # 문자열이면 새로 생성
        new_id = int(datetime.now().strftime("%Y%m%d%H%M%S%f"))
        st.html(f'<div id="st_label_{new_id}" style="{style_string}">{text}</div>')
        return new_id


def st_style_page_margin_hidden(
    top: int = 0, left: int = 10, right: int = 10, bottom: int = 0
) -> None:
    """상단 여백을 제거하면서 사이드바 토글 버튼은 유지합니다.

    Streamlit의 기본 헤더/툴바를 숨기고 페이지 상단 여백을 제거하여
    더 많은 화면 공간을 확보합니다. 단, 사이드바 토글 버튼은 유지합니다.

    적용되는 스타일:
        - Streamlit 헤더/툴바 숨김
        - 메인 컨테이너 상단 여백 제거
        - 텍스트 요소 여백 최소화
        - 사이드바 토글 버튼 유지
    """
    st.html(
        f"""<style>
[data-testid="stHeader"] {{ display: none !important; }}
[data-testid="stToolbar"] {{ display: none !important; }}        
.stMainBlockContainer {{
    padding-top: {top}px !important;
    padding-left: {left}px !important;
    padding-right: {right}px !important;
    padding-bottom: {bottom}px !important;
}}
.stVerticalBlock:first-of-type {{ margin: 0 !important; padding: 0 !important; }}
h1, h2, h3, h4, h5, h6, p, div {{ margin: 0 !important; padding: 0 !important; }}
hr.compact {{ display: none !important; margin: 0 !important; padding: 0 !important; }}
[data-testid="stSidebarNav"] {{ display: block !important; }}
</style>
"""
    )


def st_style_toolbar_hidden() -> None:
    """상단 여백을 제거하면서 사이드바 토글 버튼은 유지합니다.

    Streamlit의 기본 헤더/툴바를 숨기고 페이지 상단 여백을 제거하여
    더 많은 화면 공간을 확보합니다. 단, 사이드바 토글 버튼은 유지합니다.

    적용되는 스타일:
        - Streamlit 헤더/툴바 숨김
        - 메인 컨테이너 상단 여백 제거
        - 텍스트 요소 여백 최소화
        - 사이드바 토글 버튼 유지
    """
    st.html(
        """
<style>
    [data-testid="stHeader"] { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }
</style>
"""
    )


def st_div_divider(height: int = 1, color: str = "#ddd") -> None:
    """여백 최소화 가로선을 렌더링합니다.

    기본 st.markdown("---")의 과도한 여백을 제거하고
    얇은 회색 구분선만 표시합니다.

    사용 예:
        st_div_divider()  # 1px 회색 구분선 표시
    """
    st.html(
        f'<div style="height: {height}px; background-color: {color}; margin: 0; padding: 0;"></div>'
    )


def st_style_page_margin(
    top: int = 0, left: int = 10, right: int = 10, bottom: int = 0
) -> None:
    """상단 여백을 제거하면서 사이드바 토글 버튼은 유지합니다.

    Streamlit의 기본 헤더/툴바를 숨기고 페이지 상단 여백을 제거하여
    더 많은 화면 공간을 확보합니다. 단, 사이드바 토글 버튼은 유지합니다.

    적용되는 스타일:
        - Streamlit 헤더/툴바 숨김
        - 메인 컨테이너 상단 여백 제거
        - 텍스트 요소 여백 최소화
        - 사이드바 토글 버튼 유지
    """
    st.html(
        f"""<style>
.stMainBlockContainer {{
    padding-top: {top}px !important;
    padding-left: {left}px !important;
    padding-right: {right}px !important;
    padding-bottom: {bottom}px !important;
}}
</style>"""
    )


def st_settings_panel_show() -> None:
    """상단 설정 패널을 렌더링합니다."""
    # 매번 다른 코드로 인식되도록 고유 ID 추가
    unique_id = random.randint(1000, 9999)

    components.html(
        f"""
        <script>
        // 고유 ID: {unique_id}
        (function() {{
            setTimeout(function() {{
                const targetDoc = window.parent.document;
                const mainMenuButton = targetDoc.querySelector('[data-testid="stMainMenu"] button');
                
                if (mainMenuButton) {{
                    mainMenuButton.click();
                    console.log('MainMenu opened (ID: {unique_id})');
                }} else {{
                    console.error('MainMenu button not found');
                }}
            }}, 100);
        }})();
        </script>
        """,
        height=0,
    )


def st_query_param_get(key: str, default: int) -> int:
    """URL query parameter에서 정수 값을 가져옵니다.

    Args:
        key: query parameter 키
        default: 값이 없거나 변환 실패 시 반환할 기본값

    Returns:
        int: query parameter 값 또는 기본값

    Examples:
        >>> # URL에서 page_size 가져오기
        >>> page_size = st_query_param_get("page_size", 10)

        >>> # URL: ?page_size=25 -> 25 반환
        >>> # URL: ?other=value -> 10 반환 (기본값)

    Notes:
        - 브라우저 새로고침 후에도 값이 유지됩니다
        - URL 공유 시 설정도 함께 공유됩니다
    """
    try:
        value = st.query_params.get(key)
        if value is not None:
            return int(value)
        return default
    except (ValueError, TypeError) as e:
        logger.debug(f"Failed to parse query param '{key}': {e}")
        return default


def st_query_param_set(key: str, value: int) -> None:
    """URL query parameter에 정수 값을 저장합니다.

    Args:
        key: query parameter 키
        value: 저장할 정수 값

    Examples:
        >>> # URL에 page_size 저장
        >>> st_query_param_set("page_size", 25)
        >>> # URL이 ?page_size=25로 업데이트됨

        >>> # 슬라이더 값을 URL에 저장
        >>> page_size = st.slider("Page Size", 5, 50, 10)
        >>> st_query_param_set("page_size", page_size)

    Notes:
        - 브라우저 새로고침 후에도 값이 유지됩니다
        - 북마크 및 URL 공유 가능
    """
    try:
        st.query_params[key] = str(value)
    except Exception as e:
        logger.debug(f"Failed to set query param '{key}': {e}")
