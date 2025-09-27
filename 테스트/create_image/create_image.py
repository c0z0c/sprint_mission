import streamlit as st
import torch
import requests
import io
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
import time
import threading
import pandas as pd

# CPU 모니터링을 위한 패키지
try:
    import psutil
except ImportError:
    psutil = None

# CPU 모니터링 클래스
class CPUMonitor:
    def __init__(self, update_interval=3):
        self.update_interval = update_interval
        self.is_running = False
        self.cpu_data = []
        self.timestamps = []
        self.thread = None
        
    def start_monitoring(self):
        """CPU 모니터링 시작"""
        if not psutil:
            return
        
        self.is_running = True
        self.cpu_data = []
        self.timestamps = []
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        
    def stop_monitoring(self):
        """CPU 모니터링 종료"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1)
            
    def get_data(self):
        """현재 CPU 데이터 반환"""
        if not self.cpu_data:
            return pd.DataFrame()
        
        return pd.DataFrame({
            'CPU 사용률 (%)': self.cpu_data,
            'Time': self.timestamps
        })
        
    def _monitor_loop(self):
        """CPU 모니터링 루프"""
        start_time = time.time()
        while self.is_running:
            if psutil:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                current_time = time.time() - start_time
                
                self.cpu_data.append(cpu_percent)
                self.timestamps.append(current_time)
                
                # 최대 100개 데이터 포인트만 유지
                if len(self.cpu_data) > 100:
                    self.cpu_data.pop(0)
                    self.timestamps.pop(0)
            
            time.sleep(self.update_interval)

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
def generate_image(prompt, negative_prompt="", seed=42, progress_callback=None):
    """주어진 프롬프트와 시드로 이미지를 생성합니다."""
    pipe = load_pipeline()
    
    # 시드 고정
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    
    # CPU에서 더 빠른 생성을 위해 단계 수 미리 결정
    steps = 20 if pipe.device.type == "cpu" else 20
    
    # 진행 상황 콜백 설정 (steps 변수를 사용하도록 수정)
    def progress_fn(step, timestep, latents):
        if progress_callback:
            progress = (step + 1) / steps
            progress_callback(progress, f"단계 {step + 1}/{steps} 진행 중...")
    
    #with torch.autocast(pipe.device.type):
    
    print("이미지 생성 시작")
    
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        generator=generator,
        num_inference_steps=steps,
        guidance_scale=7.5,
        callback=progress_fn,
        callback_steps=1,
        height=512,
        width=512
    ).images[0]
    
    print("이미지 생성 완료!")

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
        include_person = st.selectbox(
            "인물 포함",
            ["있음", "없음"]
        )
        
        if include_person == "있음":
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
        else:
            gender = None
            age_range = None  
            style = st.selectbox(
                "이미지 스타일",
                ["사실적", "애니메이션", "만화", "일러스트", "판타지", "풍경화", "추상화"]
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
        
        # CPU 모니터링 설정
        st.subheader("📊 모니터링 설정")
        cpu_update_interval = st.slider(
            "CPU 상태 갱신 주기 (초)",
            min_value=1,
            max_value=10,
            value=3,
            help="CPU 사용률을 갱신하는 주기를 설정합니다."
        )
        
        show_cpu_info = st.checkbox("CPU 정보 표시", value=True)
        
        custom_prompt = st.text_area(
            "커스텀 프롬프트 (선택사항)",
            placeholder="추가하고 싶은 설명을 입력하세요..."
        )
    
    # 프롬프트 생성
    prompt_parts = []
    
    # 성별과 연령대 매핑 - 더 구체적이고 사실적으로
    gender_map = {
        "여성": "young Asian woman, natural beauty, professional photography",
        "남성": "young Asian man, handsome, professional photography", 
        "중성적": "young Asian person, androgynous features, professional photography"
    }
    
    age_map = {
        "10대 후반": "18-19 years old, youthful appearance, clear skin",
        "20대 초반": "20-23 years old, young adult, vibrant look",
        "20대 중반": "24-26 years old, mature young adult, confident expression"
    }
    
    style_map = {
        "사실적": "photorealistic, ultra realistic, highly detailed, sharp focus, professional photography, DSLR quality",
        "애니메이션": "anime style, animated, detailed anime art",
        "만화": "cartoon style, comic book art, illustration",
        "일러스트": "digital illustration, artistic rendering, detailed artwork",
        "판타지": "fantasy art, magical, mystical atmosphere",
        "풍경화": "landscape painting, scenic view, natural beauty",
        "추상화": "abstract art, artistic interpretation"
    }
    
    background_map = {
        "자연 풍경": {
            "산": "majestic mountain landscape, snow-capped peaks, dramatic sky, natural lighting, photorealistic scenery, ultra detailed nature",
            "바다": "pristine ocean view, crystal clear water, gentle waves, beach sand, coastal scenery, natural sunlight, realistic seascape",
            "숲": "dense forest environment, tall trees, natural foliage, dappled sunlight through leaves, woodland atmosphere, realistic forest",
            "초원": "vast green meadow, rolling hills, wildflowers, clear blue sky, pastoral landscape, natural grassland",
            "해변": "pristine sandy beach, turquoise water, palm trees, tropical paradise, coastal setting, realistic beach scene",
            "일몰": "golden hour sunset, warm lighting, dramatic sky colors, silhouettes, romantic atmosphere, cinematic lighting"
        },
        "도시": {
            "거리": "urban street scene, modern city buildings, realistic architecture, street lighting, bustling cityscape, contemporary urban environment",
            "빌딩": "modern skyscrapers, glass buildings, urban architecture, city skyline, metropolitan setting, realistic cityscape",
            "카페": "cozy cafe exterior, urban coffee shop, modern storefront, street-side cafe, realistic commercial setting",
            "공원": "urban park setting, city green space, trees and pathways, modern park design, realistic urban nature",
            "야경": "city at night, illuminated buildings, street lights, neon signs, urban nightscape, realistic night photography"
        },
        "실내": {
            "방": "modern interior room, well-lit space, contemporary furniture, clean design, realistic indoor lighting",
            "스튜디오": "professional photography studio, studio lighting setup, clean background, photoshoot environment",
            "도서관": "modern library interior, bookshelves, reading area, soft lighting, scholarly atmosphere",
            "카페 내부": "cozy cafe interior, warm lighting, comfortable seating, coffee shop atmosphere, realistic indoor setting"
        },
        "추상적": {
            "기하학적": "geometric abstract background, clean lines, modern design, minimalist composition",
            "컬러풀": "vibrant colorful abstract background, artistic composition, dynamic colors",
            "미니멀": "minimal abstract background, simple composition, clean aesthetic"
        },
        "단색": {
            "흰색": "clean white background, studio lighting, professional photography backdrop",
            "검은색": "elegant black background, dramatic lighting, professional studio setup",
            "파란색": "soft blue background, gradient lighting, professional backdrop",
            "분홍색": "gentle pink background, soft lighting, warm atmosphere"
        },
        "없음": {
            "투명": "simple clean background, studio lighting, professional photography setup",
            "심플": "minimal background, clean aesthetic, professional lighting"
        }
    }.get(background, {}).get(background_detail, f"realistic {background} {background_detail} setting, natural lighting, photorealistic")
    
    # 기본 프롬프트 구성
    if include_person == "있음":
        # 인물이 있는 경우
        base_prompt = f"{gender_map[gender]}, {age_map[age_range]}, {style_map[style]}"
        if background != "없음":
            base_prompt += f", {background_map}"
    else:
        # 인물이 없는 경우 - 배경만 생성
        if background != "없음":
            base_prompt = f"{background_map}, {style_map[style]}"
        else:
            base_prompt = f"{style_map[style]} artwork"
    
    # 품질 향상 키워드 추가 - 배경 타입별로 세분화
    if include_person == "있음" and style == "사실적":
        quality_keywords = "masterpiece, best quality, ultra high resolution, 8k, detailed face, perfect anatomy, natural skin texture, realistic lighting, sharp focus"
    elif include_person == "없음" and background in ["자연 풍경", "도시"]:
        # 배경만 생성하는 경우 - 풍경/도시 전용 품질 키워드
        quality_keywords = "masterpiece, best quality, ultra high resolution, 8k, photorealistic landscape photography, professional photography, sharp focus, vivid colors, natural lighting, detailed textures"
    elif include_person == "없음" and background == "실내":
        # 실내 배경 전용 품질 키워드
        quality_keywords = "masterpiece, best quality, high resolution, realistic interior design, professional architectural photography, perfect lighting, detailed textures, clean composition"
    else:
        quality_keywords = "high quality, detailed, masterpiece, best quality, sharp focus"
    
    final_prompt = f"{base_prompt}, {quality_keywords}"
    
    # 네거티브 프롬프트 생성 - 배경 타입별로 세분화
    if include_person == "있음":
        negative_prompt = "blurry, low quality, bad anatomy, deformed, disfigured, mutation, mutated, extra limbs, ugly, disgusting, poorly drawn, bad proportions, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, cartoon, anime, nsfw"
    elif include_person == "없음" and background == "자연 풍경":
        # 자연 풍경 전용 네거티브 프롬프트
        negative_prompt = "blurry, low quality, bad composition, distorted, ugly, artificial looking, fake, plastic, oversaturated, unrealistic colors, poor lighting, muddy textures, pixelated, jpeg artifacts, watermark, signature, text, people, humans, buildings, urban elements, cartoon, anime, painting style, illustration"
    elif include_person == "없음" and background == "도시":
        # 도시 풍경 전용 네거티브 프롬프트  
        negative_prompt = "blurry, low quality, bad composition, distorted, ugly, poor architecture, unrealistic buildings, bad perspective, muddy textures, oversaturated, artificial colors, poor lighting, pixelated, jpeg artifacts, watermark, signature, text, people, humans, cartoon, anime, painting style, illustration, rural elements, nature"
    elif include_person == "없음" and background == "실내":
        # 실내 배경 전용 네거티브 프롬프트
        negative_prompt = "blurry, low quality, bad composition, distorted, ugly, poor interior design, unrealistic furniture, bad lighting, cluttered, messy, dirty, poor perspective, muddy textures, pixelated, jpeg artifacts, watermark, signature, text, people, humans, cartoon, anime, painting style, illustration"
    else:
        negative_prompt = "blurry, low quality, bad composition, distorted, ugly, disgusting, poorly drawn, bad proportions, deformed, cartoon, anime, watermark, signature, text"
    
    if custom_prompt:
        final_prompt += f", {custom_prompt}"
    
    # 메인 컨텐츠 영역
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 생성될 프롬프트")
        st.write("**포지티브 프롬프트:**")
        st.code(final_prompt, language="text")
        
        st.write("**네거티브 프롬프트:**")
        st.code(negative_prompt, language="text")
        
        if st.button("🎨 이미지 생성", type="primary", use_container_width=True):
            # 진행 상황 및 모니터링 표시를 위한 플레이스홀더
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            running_placeholder = st.empty()
            cpu_chart_placeholder = st.empty()
            cpu_info_placeholder = st.empty()
            
            # CPU 모니터 초기화
            cpu_monitor = CPUMonitor(update_interval=cpu_update_interval)
            
            try:
                # 초기 상태 표시
                progress_placeholder.progress(0.0)
                status_placeholder.info("🚀 이미지 생성을 시작합니다...")
                running_placeholder.markdown("### 🔄 **RUNNING**")
                
                # CPU 모니터링 시작
                cpu_monitor.start_monitoring()
                
                # 이미지 생성을 별도 스레드에서 실행
                generation_complete = threading.Event()
                generated_image = None
                generation_error = None
                
                def generate_in_background():
                    nonlocal generated_image, generation_error
                    try:
                        generated_image = generate_image(final_prompt, negative_prompt, seed, None)
                    except Exception as e:
                        generation_error = e
                    finally:
                        generation_complete.set()
                
                generation_thread = threading.Thread(target=generate_in_background, daemon=True)
                generation_thread.start()
                
                # 실시간 CPU 모니터링 및 차트 업데이트
                chart_update_counter = 0
                while not generation_complete.is_set():
                    # CPU 데이터 가져오기
                    cpu_data = cpu_monitor.get_data()
                    
                    if not cpu_data.empty:
                        # CPU 차트 업데이트 (3초마다)
                        if chart_update_counter % cpu_update_interval == 0:
                            cpu_chart_placeholder.line_chart(
                                cpu_data.set_index('Time')['CPU 사용률 (%)'],
                                height=200
                            )
                            
                            # CPU 정보 표시
                            if show_cpu_info and psutil:
                                current_cpu = cpu_data['CPU 사용률 (%)'].iloc[-1] if not cpu_data.empty else 0
                                cpu_info = f"""
                                **CPU 상태**
                                - 현재 사용률: {current_cpu:.1f}%
                                - 논리 코어 수: {psutil.cpu_count(logical=True)}
                                - 물리 코어 수: {psutil.cpu_count(logical=False)}
                                - 평균 사용률: {cpu_data['CPU 사용률 (%)'].mean():.1f}%
                                """
                                cpu_info_placeholder.markdown(cpu_info)
                    
                    chart_update_counter += 1
                    time.sleep(1)  # 1초마다 체크
                
                # 이미지 생성 완료 대기
                generation_thread.join()
                
                # CPU 모니터링 종료
                cpu_monitor.stop_monitoring()
                
                # 에러 체크
                if generation_error:
                    raise generation_error
                
                # 완료 상태
                progress_placeholder.progress(1.0)
                status_placeholder.success("✅ 이미지가 성공적으로 생성되었습니다!")
                running_placeholder.markdown("### ✅ **COMPLETED**")
                
                # 세션 상태에 이미지 저장
                st.session_state.generated_image = generated_image
                st.session_state.used_prompt = final_prompt
                
                # 잠시 후 진행 상황 표시 제거
                time.sleep(3)
                progress_placeholder.empty()
                status_placeholder.empty()
                running_placeholder.empty()
                cpu_chart_placeholder.empty()
                cpu_info_placeholder.empty()
                
            except Exception as e:
                # CPU 모니터링 종료
                cpu_monitor.stop_monitoring()
                
                progress_placeholder.empty()
                status_placeholder.error(f"❌ 이미지 생성 중 오류가 발생했습니다: {str(e)}")
                running_placeholder.markdown("### ❌ **ERROR**")
                cpu_chart_placeholder.empty()
                cpu_info_placeholder.empty()
    
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
        - **인물 설정**: '없음'을 선택하면 인물 없이 배경이나 풍경만 생성됩니다.
        - **사실적 인물 생성**: '사실적' 스타일을 선택하면 더욱 현실적인 아시아인 인물이 생성됩니다.
        - **배경 사실성**: 각 배경 타입별로 최적화된 프롬프트가 적용되어 더욱 사실적인 결과를 얻을 수 있습니다.
        - **자연 풍경**: 산, 바다, 숲 등 각 세부 옵션마다 전문적인 풍경 사진 품질로 생성됩니다.
        - **도시 배경**: 현대적인 도시 건축물과 거리 풍경이 사실적으로 표현됩니다.
        - **실내 배경**: 모던한 인테리어와 조명이 전문 건축 사진 수준으로 생성됩니다.
        - **프롬프트 최적화**: 포지티브 프롬프트는 원하는 특징을, 네거티브 프롬프트는 피하고 싶은 요소를 명시합니다.
        - **시드 값**: 같은 시드 값과 프롬프트로는 동일한 이미지가 생성됩니다.
        - **커스텀 프롬프트**: 원하는 특징을 더 구체적으로 설명할 수 있습니다.
        - **스타일**: '사실적' 선택 시 8K 화질의 전문적인 사진 품질로 생성됩니다.
        - **배경 품질**: 배경만 생성 시 전문 사진작가 수준의 풍경/건축 사진으로 생성됩니다.
        - **네거티브 최적화**: 각 배경 타입별로 특화된 네거티브 프롬프트가 자동 적용됩니다.
        """)

if __name__ == "__main__":
    main()
