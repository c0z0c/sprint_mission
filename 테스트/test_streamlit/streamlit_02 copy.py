import streamlit as st
import torch
import requests
import io
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np

# 페이지 설정
st.set_page_config(
    page_title="AI 이미지 생성기",
    page_icon="🎨",
    layout="wide"
)

# Stable Diffusion 파이프라인 캐싱
@st.cache_resource
def load_pipeline():
    """Stable Diffusion 파이프라인을 로드하고 캐시합니다."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe = pipe.to(device)
    return pipe

# 이미지 생성 함수
def generate_image(prompt, seed=42, progress_callback=None):
    """주어진 프롬프트와 시드로 이미지를 생성합니다."""
    pipe = load_pipeline()
    
    # 시드 고정
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    
    # CPU에서 더 빠른 생성을 위해 단계 수 미리 결정
    steps = 15 if pipe.device.type == "cpu" else 20
    
    # 진행 상황 콜백 설정 (steps 변수를 사용하도록 수정)
    def progress_fn(step, timestep, latents):
        if progress_callback:
            progress = (step + 1) / steps
            progress_callback(progress, f"단계 {step + 1}/{steps} 진행 중...")
    
    with torch.autocast(pipe.device.type):
        image = pipe(
            prompt=prompt,
            generator=generator,
            num_inference_steps=steps,
            guidance_scale=7.5,
            callback=progress_fn,
            callback_steps=1
        ).images[0]
    
    return image

# 메인 애플리케이션
def main():
    st.title("🎨 AI 이미지 생성기")
    st.markdown("Stable Diffusion을 사용하여 다양한 이미지를 생성해보세요!")
    
    # 사이드바 설정
    with st.sidebar:
        st.header("🎮 이미지 설정")
        
        # 인물 옵션
        st.subheader("👤 인물")
        gender = st.selectbox(
            "성별",
            ["여성", "남성", "중성적"]
        )
        
        age_range = st.selectbox(
            "연령대",
            ["10대 후반", "20대 초반", "20대 중반"]
        )
        
        style = st.selectbox(
            "스타일",
            ["사실적", "애니메이션", "만화", "일러스트", "판타지"]
        )
        
        # 배경 옵션
        st.subheader("🌅 배경")
        background = st.selectbox(
            "배경 종류",
            ["자연 풍경", "도시", "실내", "추상적", "단색", "없음"]
        )
        
        background_detail = st.selectbox(
            "배경 세부사항",
            {
                "자연 풍경": ["산", "바다", "숲", "초원", "해변", "일몰"],
                "도시": ["거리", "빌딩", "카페", "공원", "야경"],
                "실내": ["방", "스튜디오", "도서관", "카페 내부"],
                "추상적": ["기하학적", "컬러풀", "미니멀"],
                "단색": ["흰색", "검은색", "파란색", "분홍색"],
                "없음": ["투명", "심플"]
            }.get(background, ["기본"])
        )
        
        # 추가 옵션
        st.subheader("⚙️ 고급 설정")
        seed = st.number_input("시드 값", min_value=1, max_value=999999, value=42)
        
        custom_prompt = st.text_area(
            "커스텀 프롬프트 (선택사항)",
            placeholder="추가하고 싶은 설명을 입력하세요..."
        )
    
    # 프롬프트 생성
    prompt_parts = []
    
    # 성별과 연령대 매핑
    gender_map = {
        "여성": "young woman",
        "남성": "young man", 
        "중성적": "young person"
    }
    
    age_map = {
        "10대 후반": "teenager, 18-19 years old",
        "20대 초반": "young adult, 20-23 years old",
        "20대 중반": "young adult, 24-26 years old"
    }
    
    style_map = {
        "사실적": "photorealistic, highly detailed",
        "애니메이션": "anime style, animated",
        "만화": "cartoon style, comic book art",
        "일러스트": "illustration, artistic",
        "판타지": "fantasy art, magical"
    }
    
    background_map = {
        "자연 풍경": f"{background_detail} landscape",
        "도시": f"urban {background_detail} setting",
        "실내": f"indoor {background_detail} environment",
        "추상적": f"{background_detail} background",
        "단색": f"{background_detail} background",
        "없음": "simple background"
    }
    
    # 기본 프롬프트 구성
    base_prompt = f"{gender_map[gender]}, {age_map[age_range]}, {style_map[style]}"
    if background != "없음":
        base_prompt += f", {background_map[background]}"
    
    # 품질 향상 키워드 추가
    quality_keywords = "high quality, detailed, masterpiece, best quality"
    
    final_prompt = f"{base_prompt}, {quality_keywords}"
    
    if custom_prompt:
        final_prompt += f", {custom_prompt}"
    
    # 메인 컨텐츠 영역
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 생성될 프롬프트")
        st.code(final_prompt, language="text")
        
        if st.button("🎨 이미지 생성", type="primary", use_container_width=True):
            # 진행 상황 표시를 위한 플레이스홀더
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            try:
                # 진행 상황 콜백 함수
                def update_progress(progress, message):
                    progress_placeholder.progress(progress)
                    status_placeholder.info(f"🎨 {message}")
                
                # 초기 상태 표시
                progress_placeholder.progress(0.0)
                status_placeholder.info("🚀 이미지 생성을 시작합니다...")
                
                # 이미지 생성
                generated_image = generate_image(final_prompt, seed, update_progress)
                
                # 완료 상태
                progress_placeholder.progress(1.0)
                status_placeholder.success("✅ 이미지가 성공적으로 생성되었습니다!")
                
                # 세션 상태에 이미지 저장
                st.session_state.generated_image = generated_image
                st.session_state.used_prompt = final_prompt
                
                # 잠시 후 진행 상황 표시 제거
                import time
                time.sleep(2)
                progress_placeholder.empty()
                status_placeholder.empty()
                
            except Exception as e:
                progress_placeholder.empty()
                status_placeholder.error(f"❌ 이미지 생성 중 오류가 발생했습니다: {str(e)}")
    
    with col2:
        st.subheader("🖼️ 생성된 이미지")
        
        if hasattr(st.session_state, 'generated_image') and st.session_state.generated_image:
            st.image(
                st.session_state.generated_image,
                caption=f"프롬프트: {st.session_state.used_prompt[:50]}...",
                use_column_width=True
            )
            
            # 이미지 다운로드 버튼
            img_buffer = io.BytesIO()
            st.session_state.generated_image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            st.download_button(
                label="📥 이미지 다운로드",
                data=img_buffer,
                file_name=f"generated_image_seed_{seed}.png",
                mime="image/png",
                use_container_width=True
            )
        else:
            st.info("이미지를 생성하려면 왼쪽의 '이미지 생성' 버튼을 클릭하세요.")
    
    # 팁 및 정보
    with st.expander("💡 사용 팁"):
        st.markdown("""
        - **시드 값**: 같은 시드 값과 프롬프트로는 동일한 이미지가 생성됩니다.
        - **커스텀 프롬프트**: 원하는 특징을 더 구체적으로 설명할 수 있습니다.
        - **스타일**: 각 스타일마다 다른 느낌의 이미지가 생성됩니다.
        - **배경**: 배경을 '없음'으로 설정하면 인물에 더 집중된 이미지가 생성됩니다.
        """)

if __name__ == "__main__":
    main()

