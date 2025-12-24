# 🎨 AI 이미지 생성기

Google Cloud VM의 Stable Diffusion XL API 서버를 사용하는 Streamlit 기반 이미지 생성 웹 애플리케이션입니다.

**특징**: GPU 없이도 실행 가능! 원격 API 서버를 통해 이미지를 생성합니다.

## 주요 기능

### 1. 3가지 생성 모드
- **텍스트 → 이미지**: 프롬프트만으로 이미지 생성
- **이미지 유사 생성**: 업로드한 이미지와 유사한 이미지 생성
- **이미지 + 텍스트**: 업로드한 이미지에 텍스트 효과 적용

### 2. 프롬프트 조합 시스템
- 카테고리별 한글 옵션 선택 (피사체, 스타일, 배경, 조명, 품질/효과)
- 선택한 옵션이 자동으로 영문 프롬프트로 변환
- 실시간 최종 프롬프트 미리보기
- 추가 커스텀 프롬프트 입력 가능

### 3. 이미지 관리
- 3열 그리드 레이아웃으로 생성된 이미지 표시
- 무한 스크롤로 이미지 추가
- 각 이미지별 메타데이터 (시드, 프롬프트, 생성 시간) 표시
- 개별 다운로드 및 일괄 ZIP 다운로드 지원

### 4. 고급 기능
- **시드 제어**: 랜덤 시드 또는 고정 시드 선택
- **자동 테스트 모드**: 랜덤 프롬프트 조합으로 무한 자동 생성
- **변형 강도 조절**: 이미지 변형 정도 조절 (0.1 ~ 1.0)
- **일괄 다운로드**: ZIP 파일로 모든 이미지 + 메타데이터 다운로드

## 설치 방법

### 1. 필수 요구사항
- Python 3.8 이상
- **GPU 불필요!** (원격 API 서버 사용)
- 인터넷 연결 (API 서버 접근용)

### 2. 환경 설정

```bash
# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치 (매우 간단!)
pip install -r requirements.txt
```

필요한 패키지는 3개뿐입니다:
- `streamlit` - 웹 인터페이스
- `requests` - API 호출
- `Pillow` - 이미지 처리

### 3. API 서버 설정

[src/app.py](src/app.py)의 19번째 줄에서 API URL을 확인하세요:

```python
API_URL = "http://34.64.206.210:8080/predictions/sdxl"
```

다른 API 서버를 사용하려면 이 URL을 변경하세요.

### 4. 실행
```bash
# 앱 실행
streamlit run src/app.py

# 또는 포트 지정
streamlit run src/app.py --server.port 8501
```

브라우저에서 자동으로 열리거나, `http://localhost:8501`로 접속합니다.

## 사용 가이드

### 기본 사용법

#### 1. 텍스트로 이미지 생성
1. 왼쪽 사이드바에서 "텍스트 → 이미지" 선택
2. 프롬프트 조합 섹션에서 원하는 옵션 선택
   - 피사체: 사람, 동물, 건물 등
   - 스타일: 사실적, 판타지, 애니메이션 등
   - 배경: 자연, 도시, 우주 등
   - 조명: 자연광, 극적인 조명 등
   - 품질: 초상세한, 8K, 걸작 등
3. 추가 프롬프트를 영문으로 직접 입력 (선택사항)
4. "이미지 생성" 버튼 클릭
5. 생성된 이미지가 메인 화면에 3열 그리드로 표시됩니다

#### 2. 이미지 유사 생성
1. "이미지 유사 생성" 모드 선택
2. 이미지 파일 업로드 (PNG, JPG, JPEG)
3. 변형 강도 조절 (낮을수록 원본과 유사)
4. "이미지 생성" 버튼 클릭

#### 3. 이미지에 효과 적용
1. "이미지 + 텍스트" 모드 선택
2. 이미지 파일 업로드
3. 프롬프트 조합에서 적용할 효과 선택
   - 예: "cyberpunk, neon lighting, highly detailed"
4. 변형 강도 조절 (0.75 권장)
5. "이미지 생성" 버튼 클릭

### 고급 기능

#### 시드 고정
1. 사이드바에서 "시드 고정" 체크박스 활성화
2. 시드 값 입력 (0 ~ 4,294,967,295)
3. 동일한 프롬프트와 시드로 재현 가능한 이미지 생성

#### 자동 테스트 모드
1. "자동 생성 모드" 체크박스 활성화
2. 생성 간격 설정 (1~30초)
3. 최대 생성 개수 설정 (1~100개)
4. "시작" 버튼 클릭
5. 자동으로 랜덤 프롬프트 조합 생성
6. "중지" 버튼으로 언제든 중단 가능

#### 일괄 다운로드
1. 여러 이미지 생성 후
2. 사이드바 하단의 "ZIP 다운로드" 버튼 클릭
3. 모든 이미지와 메타데이터가 포함된 ZIP 파일 다운로드
   - 파일명 형식: `image_001_seed12345.png`
   - 메타데이터: `image_001_metadata.txt` (프롬프트, 시드, 시간)

## 프롬프트 조합 가이드

### 카테고리별 추천 조합

#### 사실적인 인물 사진
- 피사체: 사람 (남성/여성)
- 스타일: 사실적인
- 배경: 자연 / 도시
- 조명: 자연광 / 황금 시간대
- 품질: 초상세한, 8K, 선명한

#### 판타지 아트
- 피사체: 용 / 천사 / 유니콘
- 스타일: 판타지
- 배경: 성 / 하늘 / 별이 빛나는 밤
- 조명: 극적인 조명 / 빛나는
- 품질: 걸작, 영화 같은, 초상세한

#### 미래도시 풍경
- 피사체: 우주선 / 로봇
- 스타일: 사이버펑크 / 미래적
- 배경: 도시 / 밤
- 조명: 네온 / 영화 같은 조명
- 품질: 8K, 대비 높은, HDR

## 시스템 요구사항

### Streamlit 서버 (클라이언트)
- CPU: 일반 PC (듀얼코어 이상)
- RAM: 4GB
- GPU: **불필요**
- 저장공간: 1GB (앱 및 의존성)
- 인터넷: 필수 (API 서버 통신)

### API 서버 (Google Cloud VM)
- GPU: NVIDIA L4 이상 권장
- RAM: 16GB+
- 저장공간: 50GB (모델 포함)

## 아키텍처

```
[사용자 브라우저]
      ↓
[Streamlit 서버] ← (GPU 불필요!)
      ↓ HTTP API
[Google Cloud VM]
      ↓
[Stable Diffusion XL Model] ← (GPU 사용)
```

## 문제 해결

### API 서버 연결 실패
```
서버 연결 실패: http://34.64.206.210:8080에 접속할 수 없습니다.
```
**해결**:
1. API 서버가 실행 중인지 확인
2. 방화벽 설정 확인
3. API URL이 올바른지 확인 ([app.py:19](src/app.py#L19))

### 요청 시간 초과
```
요청 시간 초과: 서버 응답이 없습니다.
```
**해결**:
- 네트워크 연결 상태 확인
- API 서버 부하 확인 (여러 요청이 동시에 처리 중일 수 있음)
- 잠시 후 다시 시도

### HTTP 오류 (500, 503 등)
**해결**:
- API 서버 로그 확인
- TorchServe 서비스 재시작
- 모델이 정상적으로 로드되었는지 확인

## 프로젝트 구조

모듈화된 구조로 유지보수가 쉽습니다:

```
src/
├── app.py               # 메인 애플리케이션 (147줄)
├── config.py            # 설정 및 프롬프트 매핑
├── api_client.py        # API 통신 로직
├── ui_components.py     # 재사용 가능한 UI 컴포넌트
├── generation_logic.py  # 이미지 생성 로직
└── utils.py             # 유틸리티 함수
```

자세한 내용은 [PROJECT_STRUCTURE.md](src/PROJECT_STRUCTURE.md)를 참고하세요.

## 기술 스택

### 클라이언트 (Streamlit 앱)
- **Framework**: Streamlit
- **HTTP Client**: Requests
- **Image Processing**: Pillow
- **Architecture**: Modular (6개 모듈로 분리)

### 서버 (Google Cloud VM)
- **Model Server**: TorchServe
- **AI Model**: Stable Diffusion XL (stabilityai/stable-diffusion-xl-base-1.0)
- **Deep Learning**: PyTorch, Diffusers
- **GPU**: NVIDIA L4
- **Optimization**: xformers (memory efficient attention)

## 라이선스
이 프로젝트는 Stable Diffusion XL 라이선스를 따릅니다.

## 참고 자료
- [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [Streamlit Documentation](https://docs.streamlit.io)
