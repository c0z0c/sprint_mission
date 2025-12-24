"""
UI 컴포넌트들
"""

import io
import json
import streamlit as st
from datetime import datetime
from config import PROMPT_CATEGORIES


def render_prompt_selector():
    """프롬프트 선택 UI 렌더링"""
    st.subheader("📝 프롬프트 조합")

    selected_prompts = []
    for category, items in PROMPT_CATEGORIES.items():
        with st.expander(f"🔹 {category}", expanded=False):
            selected = st.multiselect(
                f"{category} 선택",
                options=list(items.keys()),
                key=f"select_{category}",
            )
            for item in selected:
                selected_prompts.append(items[item])

    # 직접 입력
    custom_prompt = st.text_area(
        "추가 프롬프트 (영문)",
        placeholder="예: beautiful landscape, trending on artstation",
        height=100,
    )

    # 최종 프롬프트 조합
    final_prompt = ", ".join(selected_prompts)
    if custom_prompt:
        final_prompt = (
            f"{final_prompt}, {custom_prompt}" if final_prompt else custom_prompt
        )

    st.info(f"**최종 프롬프트:**\n{final_prompt if final_prompt else '(비어있음)'}")

    return final_prompt, custom_prompt


def render_image_uploader(mode):
    """이미지 업로드 UI 렌더링"""
    st.subheader("🖼️ 이미지 업로드")
    uploaded_file = st.file_uploader(
        "이미지를 선택하세요",
        type=["png", "jpg", "jpeg"],
        help="PNG, JPG, JPEG 형식 지원",
    )

    if uploaded_file:
        st.image(uploaded_file, caption="업로드된 이미지", width="content")

    # Strength 조절
    strength = st.slider(
        "변형 강도",
        min_value=0.1,
        max_value=1.0,
        value=0.3 if mode == "이미지 유사 생성" else 0.75,
        step=0.05,
        help="낮을수록 원본과 유사, 높을수록 변형이 큼",
    )

    return uploaded_file, strength


def render_seed_control():
    """시드 제어 UI 렌더링"""
    st.subheader("🎲 시드 설정")
    use_fixed_seed = st.checkbox("시드 고정")
    fixed_seed = None
    if use_fixed_seed:
        fixed_seed = st.number_input(
            "시드 값", min_value=0, max_value=2**32 - 1, value=42, step=1
        )
    return use_fixed_seed, fixed_seed


def render_auto_generation_control():
    """자동 생성 제어 UI 렌더링"""
    st.subheader("🤖 자동 테스트")
    auto_mode = st.checkbox("자동 생성 모드", value=st.session_state.auto_generating)

    # 변수 초기화
    auto_delay = 5
    max_auto_images = 10

    if auto_mode:
        auto_delay = st.slider(
            "생성 간격 (초)",
            min_value=1,
            max_value=30,
            value=5,
            help="각 이미지 생성 사이의 대기 시간",
        )

        max_auto_images = st.number_input(
            "최대 생성 개수", min_value=1, max_value=10000, value=10, step=1
        )

    return auto_mode, auto_delay, max_auto_images


def render_image_card(img_data, idx):
    """이미지 카드 렌더링 (프롬프트, 메타데이터, 다운로드 버튼 포함)"""
    st.image(img_data["image"], width="content")

    # 프롬프트를 접힌 상태로 표시
    with st.expander("📝 프롬프트", expanded=False):
        if img_data.get("prompt"):
            st.text(img_data["prompt"])
        else:
            st.caption("(프롬프트 없음)")

    # 상세 정보
    with st.expander("ℹ️ 상세 정보", expanded=False):
        st.caption(f"**모드:** {img_data['mode']}")
        st.caption(f"**시드:** {img_data['seed']}")
        st.caption(f"**시간:** {img_data['timestamp']}")

        # Strength 정보 (이미지 모드일 때)
        if img_data.get("strength") is not None:
            st.caption(f"**변형 강도:** {img_data['strength']}")

        # 메타데이터 (재현 가능한 설정)
        if img_data.get("metadata"):
            st.caption("**선택된 옵션:**")
            for cat, items in img_data["metadata"].items():
                if isinstance(items, list):
                    st.caption(f"  • {cat}: {', '.join(items)}")
                else:
                    st.caption(f"  • {cat}: {items}")

        # 커스텀 프롬프트
        if img_data.get("custom_prompt"):
            st.caption(f"**추가 프롬프트:** {img_data['custom_prompt']}")

    # JSON 다운로드 버튼
    json_data = {
        "prompt": img_data.get("prompt", ""),
        "seed": img_data["seed"],
        "mode": img_data["mode"],
        "timestamp": img_data["timestamp"],
        "metadata": img_data.get("metadata", {}),
        "custom_prompt": img_data.get("custom_prompt", ""),
        "strength": img_data.get("strength"),
    }
    json_str = json.dumps(json_data, ensure_ascii=False, indent=2)

    col_img, col_json, col_delete = st.columns(3)

    with col_img:
        # 이미지 다운로드
        img_buffer = io.BytesIO()
        img_data["image"].save(img_buffer, format="PNG")
        img_buffer.seek(0)

        st.download_button(
            label="💾 이미지",
            data=img_buffer,
            file_name=f"image_seed{img_data['seed']}.png",
            mime="image/png",
            width="content",
            key=f"download_img_{idx}",
        )

    with col_json:
        # JSON 다운로드
        st.download_button(
            label="📄 JSON",
            data=json_str,
            file_name=f"metadata_seed{img_data['seed']}.json",
            mime="application/json",
            width="content",
            key=f"download_json_{idx}",
        )

    with col_delete:
        # 삭제 버튼
        if st.button("🗑️ 삭제", width="content", key=f"delete_{idx}"):
            # 실제 인덱스 계산 (최신순으로 표시되므로)
            actual_idx = len(st.session_state.generated_images) - idx - 1
            return actual_idx

    return None


def render_bulk_download_section():
    """일괄 다운로드 섹션 렌더링"""
    from utils import create_download_zip

    st.subheader("💾 일괄 다운로드")
    if len(st.session_state.generated_images) > 0:
        st.download_button(
            label=f"📥 ZIP 다운로드 ({len(st.session_state.generated_images)}개)",
            data=create_download_zip(),
            file_name=f"generated_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
            width="content",
        )
    else:
        st.info("생성된 이미지가 없습니다.")
