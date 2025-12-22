# Movie Review Sentiment Analysis

영화 리뷰 감성 분석 서비스 - FastAPI + Streamlit + SQLModel + AI

**배포 URL**:
- Backend: https://mis18-backend-1088737588166.asia-northeast3.run.app
- Frontend: https://mis18-frontend-1088737588166.asia-northeast3.run.app
- 시연 영상: https://youtu.be/JfuecNxS8Fo

**프로젝트 상태**:
- ✅ 유닛 테스트 86개 전체 통과 (성공률 100%)
- ✅ Google Cloud Run 배포 완료
- ✅ Docker 이미지 최적화 완료

---

## 📂 프로젝트 구조

```
/mission18
├── /backend                # FastAPI 백엔드
│   ├── /app
│   │   ├── main.py         # 메인 애플리케이션
│   │   ├── /ai             # AI 감성 분석 모델
│   │   │   └── SentimentPredictor.py  # koelectra-small-v3-nsmc
│   │   ├── /config         # 설정 관리
│   │   ├── /database       # DB 연결 및 세션
│   │   ├── /models         # SQLModel 클래스
│   │   │   ├── MovieModel.py
│   │   │   ├── ReviewModel.py
│   │   │   └── VisitorModel.py
│   │   ├── /schemas        # Pydantic 스키마
│   │   ├── /routes         # API 라우터
│   │   │   ├── MovieRouter.py
│   │   │   ├── ReviewRouter.py
│   │   │   └── VisitorRouter.py
│   │   └── /services       # 비즈니스 로직
│   │       ├── MovieService.py
│   │       ├── ReviewService.py
│   │       ├── TMDBService.py
│   │       ├── SyncScheduler.py
│   │       └── VisitorService.py
│   ├── /config
│   │   └── sync_config.yaml
│   ├── /data
│   │   ├── movie_review.db  # SQLite DB
│   │   └── /posters         # 포스터 이미지
│   ├── /tests              # 유닛 테스트 (86개, 100% 통과)
│   ├── requirements.txt
│   └── Dockerfile
│
├── /frontend               # Streamlit 프론트엔드
│   ├── app.py              # 메인 엔트리
│   ├── /pages
│   │   ├── movie_list.py   # 영화 목록
│   │   ├── movie_edit.py   # 영화 편집
│   │   ├── board_list.py   # 리뷰 목록
│   │   ├── board_edit.py   # 리뷰 작성
│   │   └── management.py   # 관리 페이지
│   ├── /utils
│   │   ├── api_client.py
│   │   ├── MovieSearchUI.py
│   │   └── ReviewSearchUI.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── /script
│   ├── koelectra-small-v3-nsmc.ipynb  # AI 모델 검증
│   └── gettmdb.ipynb                  # TMDB API 테스트
│
├── docker-compose.yml
├── README.md
└── 미션18_보고서.md         # 상세 프로젝트 보고서
```

## 🚀 실행 방법

### 로컬 개발 환경

**필수 조건**: Python 3.11, Conda 환경 (mis18)

#### 1. Conda 환경 설정
```bash
conda activate mis18
```

#### 2. 백엔드 실행
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### 3. 프론트엔드 실행 (별도 터미널)
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

### Docker 실행

```bash
# 전체 빌드 및 실행
docker-compose up --build

# 백그라운드 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 중지 및 컨테이너 제거
docker-compose down

# 볼륨까지 삭제
docker-compose down -v
```

## 🌐 접속 주소

- **백엔드 API**: http://localhost:8000
- **API 문서**: http://localhost:8000/docs
- **프론트엔드**: http://localhost:8501

## 📋 주요 기능

### 백엔드 (FastAPI)

#### 영화 관리
- ✅ TMDB API 연동 영화 검색 및 자동 등록
- ✅ 영화 정보 CRUD (등록/조회/수정/삭제)
- ✅ 포스터 이미지 자동 다운로드 및 로컬 저장
- ✅ 영화 검색 (제목, 감독, 장르, 개봉일, 평점)
- ✅ 페이지네이션 지원

#### 리뷰 관리
- ✅ 리뷰 CRUD (등록/조회/수정/삭제)
- ✅ 영화별 리뷰 목록 조회
- ✅ 리뷰 검색 (작성자, 내용, 영화)
- ✅ 최근 리뷰 조회

#### AI 감성 분석
- ✅ 자동 감성 분석 (koelectra-small-v3-nsmc)
- ✅ 긍정/부정 분류 (binary classification)
- ✅ AI 평점 자동 계산 (0-10점)
- ✅ 긴 텍스트 처리 (512 토큰 이상, 20% 오버랩)
- ✅ 배치 처리 지원

#### 시스템
- ✅ TMDB 자동 동기화 (초기 100개 인기 영화)
- ✅ 방문자 통계 추적
- ✅ 헬스체크 엔드포인트
- ✅ API 문서 자동 생성 (Swagger UI)

### 프론트엔드 (Streamlit)

#### 영화 관리
- ✅ 영화 목록 (포스터 그리드 뷰)
- ✅ TMDB 검색을 통한 영화 추가
- ✅ 영화 상세 정보 표시
- ✅ 영화 편집/삭제

#### 리뷰 게시판
- ✅ 리뷰 작성 및 실시간 감성 분석
- ✅ 리뷰 목록 (감성 분석 결과 표시)
- ✅ 영화별 리뷰 필터링
- ✅ 리뷰 검색

#### 관리 대시보드
- ✅ TMDB 수동 동기화
- ✅ 통계 현황 (영화/리뷰/방문자)
- ✅ AI 평점 시각화 (Gauge Chart)
- ✅ 동기화 진행 상태 모니터링

## 🛠 기술 스택

### Backend
- **Web Framework**: FastAPI 0.115.0
- **ORM**: SQLModel 0.0.22
- **Validation**: Pydantic 2.9.2
- **Database**: SQLite
- **Server**: Uvicorn 0.32.0
- **Scheduler**: APScheduler 3.10.4
- **HTTP Client**: httpx, requests

### Frontend
- **UI Framework**: Streamlit 1.41.1
- **Visualization**: Plotly
- **HTTP Client**: requests
- **Utils**: helper-streamlit-utils

### AI/ML
- **Framework**: PyTorch 2.0.0+ (CPU 최적화)
- **Transformers**: Hugging Face Transformers 4.30.0+
- **Model**: daekeun-ml/koelectra-small-v3-nsmc
- **Tokenizer**: SentencePiece 0.1.99+

### Database
- **Engine**: SQLite (로컬/개발)
- **ORM**: SQLModel
- **Migration**: SQLModel Table Creation

### DevOps
- **Container**: Docker, Docker Compose
- **Cloud**: Google Cloud Run
- **Registry**: Docker Hub
- **Environment**: Conda (Python 3.11)

### Development
- **Logging**: helper-dev-utils
- **Testing**: pytest 7.4.0, pytest-asyncio 0.21.0
- **Environment**: python-dotenv 1.0.1

## 📝 API 엔드포인트

### Health & System
- `GET /health` - 서버 헬스체크 및 초기 동기화 상태
- `GET /` - 루트 엔드포인트 (API Docs 리다이렉트)

### 영화 (Movies)
- `GET /movies/` - 영화 목록 조회 (페이지네이션)
- `GET /movies/{tmdb_id}` - 특정 영화 상세 조회
- `POST /movies/` - 영화 등록 (TMDB ID 기반)
- `PUT /movies/{tmdb_id}` - 영화 전체 수정
- `PATCH /movies/{tmdb_id}` - 영화 부분 수정
- `DELETE /movies/{tmdb_id}` - 영화 삭제
- `GET /movies/search/tmdb` - TMDB 영화 검색
- `GET /movies/{tmdb_id}/rating` - 영화 평점 조회 (AI 기반)

### 리뷰 (Reviews)
- `GET /reviews/` - 전체 리뷰 목록 조회 (페이지네이션)
- `GET /reviews/movie/{tmdb_id}` - 특정 영화 리뷰 목록
- `GET /reviews/{review_id}` - 특정 리뷰 상세 조회
- `POST /reviews/` - 리뷰 등록 (자동 감성 분석)
- `PUT /reviews/{review_id}` - 리뷰 전체 수정
- `PATCH /reviews/{review_id}` - 리뷰 부분 수정
- `DELETE /reviews/{review_id}` - 리뷰 삭제

### 방문자 (Visitors)
- `GET /visitors/count` - 오늘 방문자 수 조회
- `POST /visitors/increment` - 방문자 수 증가

### 동기화 (Sync)
- `POST /sync/start` - TMDB 수동 동기화 시작
- `GET /sync/status` - 동기화 상태 조회

**상세 API 문서**: http://localhost:8000/docs (Swagger UI)

## 🔧 환경 변수

### Frontend
- `API_BASE_URL`: 백엔드 API 주소 (기본값: http://localhost:8000)
- `BROWSER_API_URL`: 브라우저용 API 주소 (기본값: http://localhost:8000)
- `PORT`: Streamlit 포트 (기본값: 8501)

### Backend
- `PORT`: 서버 포트 (기본값: 8000)
- `TMDB_API_KEY`: TMDB API 키 (선택, 동기화 사용 시)
- `DATABASE_URL`: 데이터베이스 URL (기본값: sqlite:///./data/movie_review.db)

### Docker Compose
```yaml
services:
  backend:
    environment:
      - PORT=8000
  frontend:
    environment:
      - PORT=8501
      - API_BASE_URL=http://backend:8000
      - BROWSER_API_URL=http://localhost:8000
```

## 📦 배포

### Docker 이미지 빌드

```bash
# 백엔드 이미지 빌드
docker build -t c0z0c/mis18_backend:v1.1 ./backend

# 프론트엔드 이미지 빌드
docker build -t c0z0c/mis18_frontend:v1.1 ./frontend

# Docker Hub 푸시
docker push c0z0c/mis18_backend:v1.1
docker push c0z0c/mis18_frontend:v1.1
```

**이미지 크기**:
- Backend: 12.6 GB (압축 전) → 4.52 GB (압축 후)
- Frontend: 959 MB (압축 전) → 211 MB (압축 후)

### Google Cloud Run 배포

**배포 정보**:
- 프로젝트: codeit04
- 리전: asia-northeast3 (서울)
- 계정: spai0433@codeit-sprint.kr

```bash
# gcloud CLI를 통한 배포
gcloud run deploy mis18-backend \
  --image c0z0c/mis18_backend:v1.1 \
  --platform managed \
  --region asia-northeast3 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --timeout 300

gcloud run deploy mis18-frontend \
  --image c0z0c/mis18_frontend:v1.1 \
  --platform managed \
  --region asia-northeast3 \
  --allow-unauthenticated \
  --memory 512Mi \
  --set-env-vars API_BASE_URL=https://mis18-backend-1088737588166.asia-northeast3.run.app,BROWSER_API_URL=https://mis18-backend-1088737588166.asia-northeast3.run.app
```

**배포된 서비스**:
- Backend: https://mis18-backend-1088737588166.asia-northeast3.run.app
- Frontend: https://mis18-frontend-1088737588166.asia-northeast3.run.app

## 🧪 테스트

### 유닛 테스트 실행

```bash
cd backend
conda activate mis18
pytest tests/ -v
```

**테스트 결과**:
- 총 테스트: 86개
- 성공: 86개 (100%)
- 실패: 0개

**테스트 커버리지**:
- Database 연결 및 세션 관리
- 영화 CRUD 및 검색 (41개 테스트)
- 리뷰 CRUD 및 검색 (37개 테스트)
- AI 감성 분석 (8개 테스트)
- API 엔드포인트 통합 테스트
- 날짜/시간 필드 자동 관리
- 헬스체크 및 동기화 상태

**상세 테스트 보고서**: `backend/tests/unit_test_report_20251221204003.md`

## 📊 데이터베이스 스키마

### Movies 테이블
```sql
CREATE TABLE movies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tmdb_id INTEGER UNIQUE NOT NULL,
    title VARCHAR(255) NOT NULL,
    release_date VARCHAR(50),
    director VARCHAR(100),
    genre VARCHAR(100),
    poster_local_path VARCHAR(500),
    tmdb_rating FLOAT,
    ai_rating FLOAT,
    overview TEXT,
    popularity FLOAT,
    vote_count INTEGER,
    original_title VARCHAR(255),
    original_language VARCHAR(10),
    adult BOOLEAN DEFAULT FALSE,
    backdrop_path VARCHAR(200)
);
```

### Reviews 테이블
```sql
CREATE TABLE reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tmdb_id INTEGER NOT NULL,
    author VARCHAR(100) NOT NULL,
    content VARCHAR(2000) NOT NULL,
    is_positive INTEGER,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    FOREIGN KEY (tmdb_id) REFERENCES movies(tmdb_id) ON DELETE CASCADE,
    UNIQUE (tmdb_id, author, content)
);
```

### Visitors 테이블
```sql
CREATE TABLE visitors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    visit_date DATE NOT NULL UNIQUE,
    count INTEGER DEFAULT 0
);
```

## 🎯 AI 모델 정보

**모델명**: daekeun-ml/koelectra-small-v3-nsmc

**특징**:
- 한국어 영화 리뷰 감성 분석 특화
- NSMC(네이버 영화 리뷰) 데이터셋 파인튜닝
- F1 스코어 90% 이상
- 경량 모델 (빠른 추론)

**성능 최적화**:
- CPU 전용 PyTorch 빌드 사용
- 배치 처리로 처리량 향상 (52-68% 개선)
- 긴 텍스트 자동 청킹 (512 토큰 이상)
- 싱글톤 패턴으로 메모리 최적화

**추론 시간**:
- 단일 리뷰 (< 512 토큰): ~0.12초
- 배치 10개: ~1.2초
- 배치 100개: ~8초

## 📚 관련 문서

- **미션 보고서**: [미션18_보고서.md](./미션18_보고서.md)
- **API 문서**: http://localhost:8000/docs
- **테스트 보고서**: [backend/tests/unit_test_report_20251221204003.md](./backend/tests/unit_test_report_20251221204003.md)
- **AI 모델 검증**: [script/koelectra-small-v3-nsmc.ipynb](./script/koelectra-small-v3-nsmc.ipynb)
- **TMDB API 테스트**: [script/gettmdb.ipynb](./script/gettmdb.ipynb)

## 🤝 기여 및 라이선스

**작성자**: 김명환  
**프로젝트**: Codeit Sprint AI 엔지니어 부트캠프 - Mission 18  
**날짜**: 2025년 12월 22일

---

**End of README**
