"""
AI 이미지 생성기 메인 애플리케이션
"""

import streamlit as st
from datetime import datetime
from helper_streamlit_utils import (
    st_style_page_margin_hidden,
    st_div_divider,
    st_sidebar_show,
)

# 모듈 임포트
from ui_components import (
    render_prompt_selector,
    render_image_uploader,
    render_seed_control,
    render_auto_generation_control,
    render_image_card,
    render_bulk_download_section,
)
from generation_logic import handle_manual_generation, handle_auto_generation

# 페이지 설정
st.set_page_config(page_title="AI 이미지 생성기", page_icon="🎨", layout="wide")

# 세션 상태 초기화
if "generated_images" not in st.session_state:
    st.session_state.generated_images = []
if "auto_generating" not in st.session_state:
    st.session_state.auto_generating = False
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False
if "generation_message" not in st.session_state:
    st.session_state.generation_message = ""

st_style_page_margin_hidden()

# 메인 UI
if st.button("##### 🎨 AI 이미지 생성기"):
    st_sidebar_show()

st.markdown("Stable Diffusion XL을 사용한 이미지 생성 도구")

# ========== 사이드바 - 입력 컨트롤 ==========
with st.sidebar:
    st.markdown("⚙️ 설정")

    # 생성 모드 선택
    mode = st.radio(
        "생성 모드",
        ["텍스트 → 이미지", "이미지 유사 생성", "이미지 + 텍스트"],
        help="텍스트→이미지: 프롬프트만으로 생성\n이미지 유사: 업로드한 이미지와 유사한 이미지 생성\n이미지+텍스트: 업로드한 이미지에 효과 적용",
    )

    st_div_divider()

    # 프롬프트 조합 (텍스트 모드일 때)
    custom_prompt = ""
    if mode in ["텍스트 → 이미지", "이미지 + 텍스트"]:
        final_prompt, custom_prompt = render_prompt_selector()
    else:
        final_prompt = ""

    st_div_divider()

    # 이미지 업로드 (이미지 모드일 때)
    uploaded_file = None
    strength = 0.75
    if mode in ["이미지 유사 생성", "이미지 + 텍스트"]:
        uploaded_file, strength = render_image_uploader(mode)

    st_div_divider()

    # 시드 설정
    use_fixed_seed, fixed_seed = render_seed_control()

    st_div_divider()

    # 자동 생성 설정
    auto_mode, auto_delay, max_auto_images = render_auto_generation_control()

    st_div_divider()

    # 생성 버튼
    generate_btn = False
    if not auto_mode:
        generate_btn = st.button("🎨 이미지 생성", type="primary", width="content")
    else:
        col1, col2 = st.columns(2)
        with col1:
            start_auto = st.button("▶️ 시작", width="content")
        with col2:
            stop_auto = st.button("⏹️ 중지", width="content")

        if start_auto:
            st.session_state.auto_generating = True
        if stop_auto:
            st.session_state.auto_generating = False

    st_div_divider()

    # 일괄 다운로드
    render_bulk_download_section()

    # 전체 삭제
    if st.button("🗑️ 전체 삭제", width="content"):
        st.session_state.generated_images = []
        st.rerun()

# ========== 메인 영역 - 이미지 생성 및 표시 ==========
st.markdown("🖼️ 생성된 이미지")

# 생성 상태 메시지를 상단에 표시
if st.session_state.generation_message:
    if st.session_state.is_generating:
        st.info(f"⏳ {st.session_state.generation_message}")
    else:
        if (
            "완료" in st.session_state.generation_message
            or "✅" in st.session_state.generation_message
        ):
            st.success(st.session_state.generation_message)
        elif (
            "실패" in st.session_state.generation_message
            or "❌" in st.session_state.generation_message
        ):
            st.error(st.session_state.generation_message)
        else:
            st.info(st.session_state.generation_message)

# 이미지 그리드를 먼저 표시 (생성 중에도 기존 이미지가 보임)
if len(st.session_state.generated_images) > 0:
    # 삭제할 인덱스를 저장
    delete_idx = None

    # 최신 순으로 정렬
    for i in range(0, len(st.session_state.generated_images), 3):
        cols = st.columns(3)

        for j in range(3):
            idx = i + j
            if idx < len(st.session_state.generated_images):
                img_data = st.session_state.generated_images[-(idx + 1)]  # 최신부터

                with cols[j]:
                    result = render_image_card(img_data, idx)
                    if result is not None:
                        delete_idx = result

    # 삭제 처리
    if delete_idx is not None:
        del st.session_state.generated_images[delete_idx]
        st.success("✅ 이미지가 삭제되었습니다.")
        st.rerun()
else:
    st.info(
        "생성된 이미지가 없습니다. 왼쪽 사이드바에서 설정 후 이미지를 생성해보세요!"
    )

# ========== 이미지 생성 로직 (화면 표시 후 실행) ==========
# 단일 이미지 생성
if not auto_mode and generate_btn:
    handle_manual_generation(
        mode, final_prompt, uploaded_file, fixed_seed, strength, custom_prompt
    )

# 자동 생성 모드
if auto_mode and st.session_state.auto_generating:
    # 이전 생성이 완료되었다면 delay 적용
    if len(st.session_state.generated_images) > 0 and not st.session_state.is_generating:
        import time
        time.sleep(auto_delay)

    handle_auto_generation(max_auto_images, use_fixed_seed, fixed_seed, auto_delay)

# 푸터
st_div_divider()
st.caption("Powered by Stable Diffusion XL • Made with Streamlit")
