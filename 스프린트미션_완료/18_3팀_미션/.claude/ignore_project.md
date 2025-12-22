 1단계 MVP(Minimum Viable Product) 개발을 시작하기 위한 프롬프트입니다.
 
--

미션 설명:
- 영화 정보, 사용자 리뷰, 리뷰 감성 분석을 표시하는 웹 애플리케이션
프론트엔드 (Streamlit):
기능 :
    - 영화 목록 표시
    - 제목, 포스터 이미지, (옵션) 평균 평점 표시
영화 추가 :
    - 입력: 제목, 개봉일, 감독, 장르, 포스터 URL
리뷰 등록:
    - 저장된 영화 선택
    - 작성자 이름, 리뷰 내용 입력
리뷰 감성 분석:
    - 리뷰 작성 후 자동 실행
    - 감성 분석 결과 표시
리뷰 표시:
    - 최근 10개 리뷰 표시
    - 항목: 영화 ID, 등록일, 리뷰 내용, 감성 분석 결과

백엔드 (FastAPI):
기능:
영화 관리:
    - 등록: 제목, 개봉일, 감독, 장르, 포스터 URL (나무위키 참고)
    - 전체/특정 영화 조회
    - 특정 영화 삭제
리뷰 관리:
    - 등록, 전체/특정 영화 리뷰 조회, 삭제
평점 조회:
    - 리뷰 감성 분석 점수의 평균

---

### 📂 프로젝트 구조 (Claude 전달용)

이 구조를 먼저 Claude에게 인지시킨 후 코딩을 시작하세요.

```text
/project-root
├── /backend
│   ├── /app
│   │   ├── main.py
│   │   ├── database.py
│   │   ├── models.py      # SQLModel Class
│   │   ├── schemas.py     # Pydantic Class
│   │   ├── /routes        # APIRouter Class (movies, reviews)
│   │   ├── /services      # Logic Class (MovieService, ReviewService)
│   │   └── /ai            # AI Class (SentimentPredictor)
│   ├── /models            # AI Model Files (Future use)
│   ├── /data/posters       # 포스터 이미지
│   ├── /data/movie_review.db # DB
│   └── Dockerfile
├── /frontend
│   ├── app.py             # Main Entry (Multi-page setup)
│   ├── /pages             # Page Classes (management, board)
│   └── Dockerfile
└── docker-compose.yml

```

---

### 🤖 Claude 입력용 프롬프트 명령 (Copy & Paste)

#### **프롬프트 1: 백엔드 아키텍처 및 DB 설계**

> "FastAPI를 사용하여 영화 감정 분류 서비스의 백엔드 MVP를 개발해줘. 아래 지침을 엄격히 준수해줘.
> FastAPI docs 준수
> **1. 아키텍처 및 코딩 스타일:**
> * 모든 로직은 **클래스 기반(Class-based)**으로 작성하고, 책임에 따라 파일을 분리해줘.
> * **중요:** 디버깅을 위해 코드 내부에 `try-except` 블록을 사용하지 마. 에러가 발생하면 스택 트레이스가 그대로 노출되어야 해.
> * SQLModel를 사용하여 `SQLModel`, `MovieModel`, `ReviewModel` 클래스를 작성해줘.
> from sqlmodel import Field, Session, SQLModel, create_engine, select
> 
> 
> **2. DB 및 API 기능:**
> * `Movies` 테이블: id, tmdb_id(unique), title, release_date, director, genre, poster_local_path, tmdb_rating.
> * `Reviews` 테이블: id, movie_id(FK), author, content, is_positive(int/null). (movie_id, author, content) 조합에 Unique 제약 추가.
> * `MovieService`: 영화 등록(이미지 저장 포함), 목록 조회 기능을 담당하는 클래스.
> * `ReviewService`: 리뷰 등록 및 목록 조회 기능을 담당하는 클래스.
> 
> 
> **3. AI 더미 로직:**
> * `app/ai/predictor.py`에 `SentimentPredictor` 클래스를 만들고, `predict` 메서드가 텍스트를 입력받아 0(부정) 또는 1(긍정)을 랜덤하게 반환하도록 해줘.
> 
> 
> 먼저 `database.py`, `models.py`, `schemas.py` 코드를 작성해줘."

#### **프롬프트 2: API 엔드포인트 및 서비스 로직**

> "앞서 정의한 모델을 바탕으로 `routes/`와 `services/` 폴더에 들어갈 코드를 작성해줘.
> * 모든 엔드포인트는 `APIRouter`를 사용하는 클래스 스타일로 구성해줘.
> * 영화 등록 시 전달받은 포스터 이미지를 `data/posters/{tmdb_id}.jpg` 경로로 저장하는 로직을 포함해줘.
> * 리뷰 등록 시 `SentimentPredictor` 클래스를 호출해서 `is_positive` 값을 채워넣어야 해.
> * `main.py`에서 이 모든 라우터를 통합해줘."
> 
> 

#### **프롬프트 3: 프론트엔드 멀티 페이지 설계**

> "Streamlit을 사용하여 멀티 페이지 구조의 프론트엔드 MVP를 개발해줘.
> * `app.py`: `st.navigation` 또는 기본 멀티 페이지 기능을 사용하여 '영화 관리'와 '리뷰 게시판' 메뉴를 구성해줘.
> * **페이지 1 (영화 관리):** `MovieManager` 클래스를 작성하여 영화 정보 입력(직접 입력 모드) 및 포스터 파일 업로드 기능을 구현해줘.
> * **페이지 2 (리뷰 게시판):** `ReviewManager` 클래스를 작성하여 영화 선택, 리뷰 작성, 그리고 AI 평점(긍정 비율 기반 5점 만점)을 시각화(Gauge chart 또는 Bar)해줘.
> * API 통신은 `requests` 라이브러리를 사용하고, 백엔드 주소는 환경변수 또는 상수로 관리해줘."
> 
> 

#### **프롬프트 4: Docker 배포 설정**

> "백엔드(FastAPI)와 프론트엔드(Streamlit)를 각각 빌드할 수 있는 `Dockerfile`을 작성해줘.
> * 백엔드는 `python:3.11-slim` 기반으로 하고 `data/` 폴더를 유지해줘.
> * 프론트엔드는 8501 포트를 사용하도록 설정해줘.
> * Google Cloud Run 배포를 고려하여 각 컨테이너의 포트 설정을 유연하게 만들어줘.
> * 두 서비스를 연결하는 `docker-compose.yml` 파일도 작성해줘."

# 개발환경
> * conda activate mis18