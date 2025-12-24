"""
API 클라이언트 - 원격 서버와 통신
"""

import io
import random
import base64
import requests
import streamlit as st
from PIL import Image
from config import API_URL


def generate_image(mode, prompt, uploaded_file=None, seed=None, strength=0.75):
    """
    원격 API를 사용한 이미지 생성 함수

    Args:
        mode: 생성 모드 ("텍스트 → 이미지", "이미지 유사 생성", "이미지 + 텍스트")
        prompt: 텍스트 프롬프트
        uploaded_file: 업로드된 이미지 파일
        seed: 시드값 (None이면 랜덤)
        strength: 이미지 변형 강도 (0.1 ~ 1.0)

    Returns:
        (image, seed): 생성된 PIL Image 객체와 사용된 시드값
    """
    # 시드 설정
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    # API 요청 페이로드 구성
    payload = {"strength": strength, "seed": seed}

    try:
        if mode == "텍스트 → 이미지":
            # Text-to-Image: 프롬프트만 전송
            if not prompt:
                st.error("프롬프트를 입력해주세요.")
                return None, None
            payload["prompt"] = prompt

        elif mode == "이미지 유사 생성":
            # Image-to-Image (유사): 이미지만 전송
            if uploaded_file is None:
                st.error("이미지를 업로드해주세요.")
                return None, None

            # 이미지를 base64로 인코딩
            image_bytes = uploaded_file.read()
            uploaded_file.seek(0)  # 파일 포인터 리셋
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            payload["image"] = image_b64
            # strength는 낮게 설정 (원본과 유사하게)
            payload["strength"] = min(strength, 0.5)

        elif mode == "이미지 + 텍스트":
            # Image-to-Image (효과): 이미지 + 프롬프트 전송
            if uploaded_file is None:
                st.error("이미지를 업로드해주세요.")
                return None, None
            if not prompt:
                st.error("프롬프트를 입력해주세요.")
                return None, None

            # 이미지를 base64로 인코딩
            image_bytes = uploaded_file.read()
            uploaded_file.seek(0)  # 파일 포인터 리셋
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            payload["image"] = image_b64
            payload["prompt"] = prompt

        # API 호출
        headers = {"Content-Type": "application/json"}
        response = requests.post(API_URL, json=payload, headers=headers, timeout=120)
        response.raise_for_status()

        # 응답 이미지 파싱
        image = Image.open(io.BytesIO(response.content))

        return image, seed

    except requests.exceptions.Timeout:
        st.error("요청 시간 초과: 서버 응답이 없습니다. 잠시 후 다시 시도해주세요.")
        return None, None
    except requests.exceptions.ConnectionError:
        st.error(f"서버 연결 실패: {API_URL}에 접속할 수 없습니다.")
        return None, None
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP 오류: {e.response.status_code} - {e.response.text}")
        return None, None
    except Exception as e:
        st.error(f"이미지 생성 중 오류가 발생했습니다: {e}")
        return None, None
