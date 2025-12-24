import streamlit as st
import io
import random
import time
import zipfile
import base64
import requests
from PIL import Image
from datetime import datetime

# 페이지 설정
st.set_page_config(page_title="AI 이미지 생성기", page_icon="🎨", layout="wide")

# API 설정
API_URL = "http://34.64.206.210:8080/predictions/sdxl"

# 세션 상태 초기화
if "generated_images" not in st.session_state:
    st.session_state.generated_images = []
if "auto_generating" not in st.session_state:
    st.session_state.auto_generating = False

# 한글-영문 프롬프트 매핑 딕셔너리
PROMPT_CATEGORIES = {
    "피사체": {
        "10대 후반 여성": "a beautiful teenage girl, late teens",
        "20대 초반 여성": "a beautiful young woman in her early 20s",
        "20대 후반 여성": "a woman in her late 20s",
        "30대 여성": "a woman in her 30s",
        "10대 후반 남성": "a teenage boy, late teens",
        "20대 초반 남성": "a handsome young man in his early 20s",
        "20대 후반 남성": "a man in his late 20s",
        "30대 남성": "a man in his 30s",
        "중년 남성": "a middle-aged man",
        "중년 여성": "a middle-aged woman",
        "어린이": "a child",
        "노인": "an elderly person",
        "비즈니스맨": "a businessman",
        "비즈니스우먼": "a businesswoman",
        "학생": "a student",
        "커플": "a couple",
        "가족": "a family",
        "고양이": "a cat",
        "강아지": "a dog",
        "말": "a horse",
        "용": "a dragon",
        "유니콘": "a unicorn",
        "새": "a bird",
        "물고기": "a fish",
        "꽃": "flowers",
        "나무": "a tree",
        "산": "mountains",
        "집": "a house",
        "성": "a castle",
        "우주선": "a spaceship",
        "자동차": "a car",
        "로봇": "a robot",
        "천사": "an angel",
    },
    "인종/민족": {
        "한국인": "Korean",
        "동양인 (일반)": "East Asian",
        "중국인": "Chinese",
        "일본인": "Japanese",
        "동남아시아인": "Southeast Asian",
        "백인 (유럽계)": "Caucasian, European",
        "백인 (미국)": "White American",
        "흑인 (아프리카계)": "African, Black",
        "흑인 (미국)": "African American",
        "라틴계": "Latino, Hispanic",
        "중동계": "Middle Eastern",
        "인도계": "Indian, South Asian",
        "혼혈": "mixed race, multiracial",
    },
    "인물 스타일": {
        "K-POP 스타": "K-pop idol, Korean pop star style",
        "K-드라마 배우": "Korean drama actor style",
        "할리우드 스타": "Hollywood celebrity style",
        "패션 모델": "fashion model",
        "슈퍼모델": "supermodel",
        "인플루언서": "social media influencer",
        "아이돌": "idol, pop star",
        "뮤지션": "musician, artist",
        "운동선수": "athlete, sporty",
        "댄서": "professional dancer",
        "배우": "actor, actress",
        "아티스트": "creative artist",
    },
    "스타일": {
        "사실적인": "photorealistic",
        "판타지": "fantasy art",
        "애니메이션": "anime style",
        "만화": "cartoon style",
        "수채화": "watercolor painting",
        "유화": "oil painting",
        "3D 렌더링": "3D render",
        "미니멀": "minimalist",
        "사이버펑크": "cyberpunk",
        "스팀펑크": "steampunk",
        "복고풍": "retro style",
        "미래적": "futuristic",
        "바로크": "baroque style",
        "인상주의": "impressionist",
        "추상화": "abstract art",
    },
    "배경/장소": {
        "자연": "natural scenery",
        "해변": "beach",
        "강가": "riverside, by the river",
        "호숫가": "lakeside",
        "숲": "forest",
        "사막": "desert",
        "산": "mountains",
        "눈 덮인 산": "snowy mountains",
        "공원": "park",
        "정원": "garden",
        "도시": "city background",
        "거리": "street scene, urban street",
        "도심": "downtown, city center",
        "골목길": "alleyway",
        "카페": "cafe, coffee shop",
        "레스토랑": "restaurant",
        "바": "bar",
        "사무실": "office",
        "회의실": "conference room",
        "세미나실": "seminar room",
        "강의실": "lecture hall, classroom",
        "도서관": "library",
        "병원": "hospital",
        "병원 로비": "hospital lobby",
        "진료실": "medical examination room",
        "호텔 로비": "hotel lobby",
        "호텔 객실": "hotel room",
        "쇼핑몰": "shopping mall",
        "백화점": "department store",
        "공항": "airport",
        "기차역": "train station",
        "체육관": "gym, fitness center",
        "스튜디오": "studio",
        "갤러리": "art gallery",
        "박물관": "museum",
        "극장": "theater",
        "콘서트홀": "concert hall",
        "학교": "school",
        "대학 캠퍼스": "university campus",
        "우주": "space background",
        "동굴": "cave",
        "수중": "underwater",
        "하늘": "sky",
        "구름": "clouds",
        "별이 빛나는 밤": "starry night",
        "일몰": "sunset",
        "일출": "sunrise",
        "밤": "night scene",
        "옥상": "rooftop",
    },
    "조명": {
        "자연광": "natural lighting",
        "극적인 조명": "dramatic lighting",
        "부드러운 조명": "soft lighting",
        "네온": "neon lighting",
        "역광": "backlighting",
        "황금 시간대": "golden hour",
        "파란 시간대": "blue hour",
        "스튜디오 조명": "studio lighting",
        "영화 같은 조명": "cinematic lighting",
        "빛나는": "glowing",
        "어두운": "dark moody lighting",
    },
    "품질/효과": {
        "초상세한": "highly detailed",
        "8K": "8k resolution",
        "4K": "4k resolution",
        "선명한": "sharp focus",
        "걸작": "masterpiece",
        "최고 품질": "best quality",
        "전문가 수준": "professional",
        "영화 같은": "cinematic",
        "생동감 있는": "vibrant colors",
        "대비 높은": "high contrast",
        "HDR": "HDR",
        "초현실적": "hyperrealistic",
    },
}


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


def create_download_zip():
    """생성된 모든 이미지를 ZIP으로 압축 (JSON 메타데이터 포함)"""
    import json

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


# 메인 UI
st.title("🎨 AI 이미지 생성기")
st.markdown("Stable Diffusion XL을 사용한 이미지 생성 도구")

# 사이드바 - 입력 컨트롤
with st.sidebar:
    st.header("⚙️ 설정")

    # 생성 모드 선택
    mode = st.radio(
        "생성 모드",
        ["텍스트 → 이미지", "이미지 유사 생성", "이미지 + 텍스트"],
        help="텍스트→이미지: 프롬프트만으로 생성\n이미지 유사: 업로드한 이미지와 유사한 이미지 생성\n이미지+텍스트: 업로드한 이미지에 효과 적용",
    )

    st.divider()

    # 프롬프트 조합 (텍스트 모드일 때)
    custom_prompt = ""  # 변수 초기화
    if mode in ["텍스트 → 이미지", "이미지 + 텍스트"]:
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
    else:
        final_prompt = ""

    st.divider()

    # 이미지 업로드 (이미지 모드일 때)
    uploaded_file = None
    strength = 0.75
    if mode in ["이미지 유사 생성", "이미지 + 텍스트"]:
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

    st.divider()

    # 시드 설정
    st.subheader("🎲 시드 설정")
    use_fixed_seed = st.checkbox("시드 고정")
    fixed_seed = None
    if use_fixed_seed:
        fixed_seed = st.number_input(
            "시드 값", min_value=0, max_value=2**32 - 1, value=42, step=1
        )

    st.divider()

    # 자동 생성 설정
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
            "최대 생성 개수", min_value=1, max_value=100, value=10, step=1
        )

    st.divider()

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

    st.divider()

    # 일괄 다운로드
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

    # 전체 삭제
    if st.button("🗑️ 전체 삭제", width="content"):
        st.session_state.generated_images = []
        st.rerun()

# 메인 영역 - 이미지 그리드
st.header("🖼️ 생성된 이미지")

# 단일 이미지 생성
if not auto_mode and generate_btn:
    if mode in ["이미지 유사 생성", "이미지 + 텍스트"] and uploaded_file is None:
        st.warning("이미지를 업로드해주세요.")
    elif mode in ["텍스트 → 이미지", "이미지 + 텍스트"] and not final_prompt:
        st.warning("프롬프트를 입력해주세요.")
    else:
        with st.spinner("이미지 생성 중..."):
            image, seed = generate_image(
                mode=mode,
                prompt=final_prompt,
                uploaded_file=uploaded_file,
                seed=fixed_seed,
                strength=strength,
            )

            if image:
                # 메타데이터 수집 (재현 가능하도록)
                manual_metadata = {}
                for category, items in PROMPT_CATEGORIES.items():
                    selected_key = f"select_{category}"
                    selected_items = st.session_state.get(selected_key, [])
                    if selected_items:
                        manual_metadata[category] = selected_items

                st.session_state.generated_images.append(
                    {
                        "image": image,
                        "prompt": final_prompt,
                        "seed": seed,
                        "mode": mode,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "metadata": manual_metadata,
                        "custom_prompt": custom_prompt,
                        "strength": (
                            strength
                            if mode in ["이미지 유사 생성", "이미지 + 텍스트"]
                            else None
                        ),
                    }
                )
                st.success(f"✅ 이미지 생성 완료! (Seed: {seed})")
                st.rerun()

# 자동 생성 모드
if auto_mode and st.session_state.auto_generating:
    if len(st.session_state.generated_images) < max_auto_images:
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

        with st.spinner(
            f"자동 생성 중... ({len(st.session_state.generated_images)+1}/{max_auto_images})"
        ):
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
                st.success(
                    f"✅ 자동 생성 완료! ({len(st.session_state.generated_images)}/{max_auto_images})"
                )

                # 대기 후 재실행
                time.sleep(auto_delay)
                st.rerun()
    else:
        st.session_state.auto_generating = False
        st.info(f"자동 생성 완료! 총 {max_auto_images}개의 이미지가 생성되었습니다.")

# 이미지 그리드 (3열)
if len(st.session_state.generated_images) > 0:
    # 최신 순으로 정렬
    for i in range(0, len(st.session_state.generated_images), 3):
        cols = st.columns(3)

        for j in range(3):
            idx = i + j
            if idx < len(st.session_state.generated_images):
                img_data = st.session_state.generated_images[-(idx + 1)]  # 최신부터

                with cols[j]:
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
                            st.caption(
                                f"**추가 프롬프트:** {img_data['custom_prompt']}"
                            )

                    # JSON 다운로드 버튼
                    import json

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

                    col_img, col_json = st.columns(2)

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
else:
    st.info(
        "생성된 이미지가 없습니다. 왼쪽 사이드바에서 설정 후 이미지를 생성해보세요!"
    )

# 푸터
st.divider()
st.caption("Powered by Stable Diffusion XL • Made with Streamlit")
