"""
이미지 생성 로직
"""

import random
import time
import streamlit as st
from datetime import datetime
from config import PROMPT_CATEGORIES
from api_client import generate_image
from utils import collect_metadata_from_session


def handle_manual_generation(mode, final_prompt, uploaded_file, fixed_seed, strength, custom_prompt):
    """수동 이미지 생성 처리"""
    if mode in ["이미지 유사 생성", "이미지 + 텍스트"] and uploaded_file is None:
        st.warning("이미지를 업로드해주세요.")
        return

    if mode in ["텍스트 → 이미지", "이미지 + 텍스트"] and not final_prompt:
        st.warning("프롬프트를 입력해주세요.")
        return

    # 생성 시작 플래그 설정
    st.session_state.is_generating = True
    st.session_state.generation_message = "🎨 이미지 생성 중..."

    image, seed = generate_image(
        mode=mode,
        prompt=final_prompt,
        uploaded_file=uploaded_file,
        seed=fixed_seed,
        strength=strength,
    )

    if image:
        # 메타데이터 수집 (재현 가능하도록)
        manual_metadata = collect_metadata_from_session()

        st.session_state.generated_images.append(
            {
                "image": image,
                "prompt": final_prompt,
                "seed": seed,
                "mode": mode,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "metadata": manual_metadata,
                "custom_prompt": custom_prompt,
                "strength": strength
                if mode in ["이미지 유사 생성", "이미지 + 텍스트"]
                else None,
            }
        )
        st.session_state.is_generating = False
        st.session_state.generation_message = f"✅ 이미지 생성 완료! (Seed: {seed})"
        st.rerun()
    else:
        st.session_state.is_generating = False
        st.session_state.generation_message = "❌ 이미지 생성 실패"


def handle_auto_generation(max_auto_images, use_fixed_seed, fixed_seed, auto_delay):
    """자동 이미지 생성 처리"""
    if len(st.session_state.generated_images) >= max_auto_images:
        st.session_state.auto_generating = False
        st.session_state.generation_message = f"✅ 자동 생성 완료! 총 {max_auto_images}개의 이미지가 생성되었습니다."
        return

    # 랜덤 프롬프트 생성
    auto_prompts = []
    auto_metadata = {}  # 재현 가능하도록 선택된 항목 저장

    for category, items in PROMPT_CATEGORIES.items():
        # 해당 카테고리의 선택된 항목 가져오기
        selected_key = f"select_{category}"
        selected_items = st.session_state.get(selected_key, [])

        if selected_items:
            # 선택된 항목이 있으면 그 중에서 랜덤 선택
            chosen_korean = random.choice(selected_items)
            chosen_english = items[chosen_korean]
            auto_prompts.append(chosen_english)
            auto_metadata[category] = chosen_korean
        else:
            # 선택된 항목이 없으면 전체에서 랜덤 선택
            chosen_korean = random.choice(list(items.keys()))
            chosen_english = items[chosen_korean]
            # 3-6개 카테고리만 랜덤으로 포함
            if len(auto_prompts) < random.randint(3, 6):
                auto_prompts.append(chosen_english)
                auto_metadata[category] = chosen_korean

    auto_prompt = ", ".join(auto_prompts)

    # 시드 생성 (재현 가능하도록 명시적으로 생성)
    if use_fixed_seed and fixed_seed is not None:
        auto_seed = fixed_seed
    else:
        auto_seed = random.randint(0, 2**32 - 1)

    # 생성 상태 메시지 업데이트
    st.session_state.is_generating = True
    st.session_state.generation_message = f"🎨 자동 생성 중... ({len(st.session_state.generated_images)+1}/{max_auto_images})"

    image, seed = generate_image(
        mode="텍스트 → 이미지",
        prompt=auto_prompt,
        seed=auto_seed,
    )

    if image:
        st.session_state.generated_images.append(
            {
                "image": image,
                "prompt": auto_prompt,
                "seed": seed,
                "mode": "자동 생성",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "metadata": auto_metadata,  # 재현 가능한 한글 선택 항목
            }
        )
        st.session_state.is_generating = False
        st.session_state.generation_message = f"✅ 이미지 생성 완료! (Seed: {seed})"

        # 화면을 갱신하여 새 이미지를 먼저 표시
        st.rerun()

        # delay는 rerun 후 다음 생성 전에 적용됨 (app.py에서 처리)
    else:
        st.session_state.is_generating = False
        st.session_state.generation_message = "❌ 이미지 생성 실패"
        st.rerun()
