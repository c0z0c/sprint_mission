# 프로젝트 구조

## 파일 구성

```
src/
├── app.py                  # 메인 애플리케이션 (147줄)
├── config.py               # 설정 및 상수 (175줄)
├── api_client.py           # API 통신 (101줄)
├── ui_components.py        # UI 컴포넌트 (199줄)
├── generation_logic.py     # 생성 로직 (110줄)
├── utils.py                # 유틸리티 함수 (66줄)
└── app_backup.py           # 백업 (기존 단일 파일)
```

**총 라인 수**: ~800줄 → 모듈화 후 각 파일 100-200줄

## 모듈 설명

### 1. `app.py` (메인)
**역할**: 애플리케이션 진입점 및 전체 흐름 제어

**주요 기능**:
- Streamlit 페이지 설정
- 세션 상태 초기화
- UI 레이아웃 구성
- 각 모듈 함수 호출

**의존성**:
```python
from ui_components import *
from generation_logic import *
```

---

### 2. `config.py` (설정)
**역할**: 전역 설정 및 프롬프트 매핑 딕셔너리

**포함 내용**:
- `API_URL`: 원격 API 서버 주소
- `PROMPT_CATEGORIES`: 한글-영문 프롬프트 매핑
  - 피사체 (34개)
  - 인종/민족 (13개)
  - 인물 스타일 (12개)
  - 스타일 (15개)
  - 배경/장소 (47개)
  - 조명 (11개)
  - 품질/효과 (12개)

**사용 방법**:
```python
from config import API_URL, PROMPT_CATEGORIES
```

---

### 3. `api_client.py` (API 통신)
**역할**: 원격 서버와의 HTTP 통신

**주요 함수**:
- `generate_image(mode, prompt, uploaded_file, seed, strength)`
  - 3가지 모드 지원 (Text-to-Image, Image-to-Image, Text+Image)
  - base64 이미지 인코딩
  - 오류 처리 (Timeout, ConnectionError, HTTPError)

**반환값**:
```python
(image: PIL.Image, seed: int) or (None, None)
```

---

### 4. `ui_components.py` (UI 컴포넌트)
**역할**: 재사용 가능한 Streamlit UI 컴포넌트

**주요 함수**:

#### `render_prompt_selector()`
프롬프트 조합 UI (콤보박스 + 커스텀 입력)
```python
Returns: (final_prompt: str, custom_prompt: str)
```

#### `render_image_uploader(mode)`
이미지 업로드 및 strength 슬라이더
```python
Returns: (uploaded_file, strength: float)
```

#### `render_seed_control()`
시드 고정/랜덤 선택
```python
Returns: (use_fixed_seed: bool, fixed_seed: int or None)
```

#### `render_auto_generation_control()`
자동 생성 설정
```python
Returns: (auto_mode: bool, auto_delay: int, max_auto_images: int)
```

#### `render_image_card(img_data, idx)`
생성된 이미지 카드 (프롬프트, 메타데이터, 다운로드 버튼)

#### `render_bulk_download_section()`
일괄 ZIP 다운로드 섹션

---

### 5. `generation_logic.py` (생성 로직)
**역할**: 이미지 생성 비즈니스 로직

**주요 함수**:

#### `handle_manual_generation(...)`
수동 이미지 생성 처리
- 입력 검증
- 메타데이터 수집
- 세션 상태 업데이트

#### `handle_auto_generation(...)`
자동 이미지 생성 처리
- 랜덤 프롬프트 생성
- 선택된 옵션 우선 사용
- 재현 가능한 메타데이터 저장

---

### 6. `utils.py` (유틸리티)
**역할**: 공통 헬퍼 함수

**주요 함수**:

#### `create_download_zip()`
모든 이미지를 ZIP으로 압축 (JSON 메타데이터 포함)

#### `collect_metadata_from_session()`
현재 세션의 선택된 옵션을 메타데이터로 수집

#### `format_metadata_display(metadata)`
메타데이터를 읽기 쉬운 형식으로 포맷

---

## 데이터 흐름

### 1. 수동 생성
```
app.py
  → ui_components.render_prompt_selector()  # 프롬프트 입력
  → generation_logic.handle_manual_generation()
    → utils.collect_metadata_from_session()  # 메타데이터 수집
    → api_client.generate_image()  # API 호출
    → st.session_state.generated_images.append()  # 저장
```

### 2. 자동 생성
```
app.py
  → generation_logic.handle_auto_generation()
    → 랜덤 프롬프트 생성 (선택 항목 우선)
    → api_client.generate_image()
    → st.session_state.generated_images.append()
    → time.sleep() + st.rerun()
```

### 3. 이미지 표시
```
app.py
  → ui_components.render_image_card()
    → 프롬프트 expander
    → 상세 정보 expander (메타데이터)
    → 다운로드 버튼 (이미지 + JSON)
```

### 4. 일괄 다운로드
```
app.py
  → ui_components.render_bulk_download_section()
    → utils.create_download_zip()
      → 각 이미지 + JSON 메타데이터 포함
```

---

## 확장 가이드

### 새로운 프롬프트 카테고리 추가
**파일**: `config.py`

```python
PROMPT_CATEGORIES = {
    # ... 기존 카테고리 ...
    "새 카테고리": {
        "한글 옵션1": "english option 1",
        "한글 옵션2": "english option 2",
    },
}
```

### 새로운 생성 모드 추가
**파일**: `api_client.py`

```python
def generate_image(mode, ...):
    if mode == "새로운 모드":
        # 새 로직 구현
        payload["new_param"] = value
```

### 새로운 UI 컴포넌트 추가
**파일**: `ui_components.py`

```python
def render_new_component():
    st.subheader("새 컴포넌트")
    # UI 구현
    return result
```

**파일**: `app.py`

```python
from ui_components import render_new_component

# 사이드바에서 호출
result = render_new_component()
```

---

## 테스트

### 로컬 테스트
```bash
# 모듈 임포트 테스트
cd src
python -c "from config import PROMPT_CATEGORIES; print(len(PROMPT_CATEGORIES))"

# 앱 실행
streamlit run app.py
```

### 모듈 단위 테스트 (예시)
```python
# test_utils.py
from utils import collect_metadata_from_session

def test_collect_metadata():
    # 테스트 코드
    pass
```

---

## 마이그레이션 가이드

### 기존 app.py에서 새 구조로

**Before** (단일 파일, 527줄):
```python
# app.py (527줄)
import streamlit as st
import torch
import io
# ... 모든 코드 한 파일에 ...
PROMPT_CATEGORIES = { ... }
def generate_image(): ...
def create_download_zip(): ...
# ... UI 코드 ...
```

**After** (모듈화, 각 100-200줄):
```python
# app.py (147줄) - 메인 로직만
from config import PROMPT_CATEGORIES
from api_client import generate_image
from utils import create_download_zip
from ui_components import render_*
from generation_logic import handle_*
```

### 백업 복원
```bash
cd src
mv app.py app_new.py
mv app_backup.py app.py
```

---

## 장점

### 1. 가독성
- 각 파일이 100-200줄으로 관리 용이
- 명확한 책임 분리

### 2. 유지보수
- 버그 수정 시 해당 모듈만 수정
- 기능 추가 시 새 함수만 작성

### 3. 재사용성
- UI 컴포넌트를 다른 페이지에서도 사용 가능
- 유틸리티 함수 공통 사용

### 4. 테스트
- 모듈 단위 테스트 가능
- 의존성 주입 용이

### 5. 협업
- 여러 개발자가 동시에 다른 모듈 작업 가능
- Git conflict 최소화
