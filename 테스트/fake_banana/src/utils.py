"""
유틸리티 함수들
"""

import io
import json
import zipfile
import streamlit as st
from datetime import datetime


def create_download_zip():
    """생성된 모든 이미지를 ZIP으로 압축 (JSON 메타데이터 포함)"""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for idx, img_data in enumerate(st.session_state.generated_images):
            # 이미지 저장
            img_buffer = io.BytesIO()
            img_data["image"].save(img_buffer, format="PNG")
            img_buffer.seek(0)

            # 파일명에 시드값 포함
            filename = f"image_{idx+1:03d}_seed{img_data['seed']}.png"
            zip_file.writestr(filename, img_buffer.getvalue())

            # JSON 메타데이터 저장 (재현 가능)
            json_metadata = {
                "prompt": img_data.get("prompt", ""),
                "seed": img_data["seed"],
                "mode": img_data["mode"],
                "timestamp": img_data["timestamp"],
                "metadata": img_data.get("metadata", {}),
                "custom_prompt": img_data.get("custom_prompt", ""),
                "strength": img_data.get("strength"),
            }
            json_str = json.dumps(json_metadata, ensure_ascii=False, indent=2)
            zip_file.writestr(f"image_{idx+1:03d}_metadata.json", json_str)

    zip_buffer.seek(0)
    return zip_buffer


def collect_metadata_from_session():
    """현재 세션에서 선택된 옵션을 메타데이터로 수집"""
    from config import PROMPT_CATEGORIES

    metadata = {}
    for category, items in PROMPT_CATEGORIES.items():
        selected_key = f"select_{category}"
        selected_items = st.session_state.get(selected_key, [])
        if selected_items:
            metadata[category] = selected_items
    return metadata


def format_metadata_display(metadata):
    """메타데이터를 읽기 쉬운 형식으로 포맷"""
    if not metadata:
        return None

    lines = []
    for cat, items in metadata.items():
        if isinstance(items, list):
            lines.append(f"• {cat}: {', '.join(items)}")
        else:
            lines.append(f"• {cat}: {items}")
    return "\n".join(lines)
