# Movie Review Sentiment Analysis

영화 리뷰 감성 분석 서비스 - FastAPI + Streamlit + SQLModel

## 📂 프로젝트 구조

```
/mission18
├── /backend                # FastAPI 백엔드
│   ├── /app
│   │   ├── main.py         # 메인 애플리케이션
│   │   ├── /database       # DB 연결 및 세션
│   │   ├── /models         # SQLModel 클래스
│   │   ├── /schemas        # Pydantic 스키마
│   │   ├── /routes         # API 라우터
│   │   ├── /services       # 비즈니스 로직
│   │   └── /ai             # 감성 분석 AI
│   ├── /data/posters       # 포스터 이미지
│   ├── /data/movie_review.db # DB
│   ├── requirements.txt
│   └── Dockerfile
├── /frontend               # Streamlit 프론트엔드
│   ├── app.py              # 메인 엔트리
│   ├── /pages
│   │   ├── management.py   # 영화 관리
│   │   └── board.py        # 리뷰 게시판
│   ├── requirements.txt
│   └── Dockerfile
└── docker-compose.yml
```

## 실행 방법

### 로컬 개발 환경

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

#### 3. 프론트엔드 실행
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

### Docker 실행

```bash
# 빌드 및 실행
docker-compose up --build

# 백그라운드 실행
docker-compose up -d

# 중지
docker-compose down
```

## 🌐 접속 주소

- **백엔드 API**: http://localhost:8000
- **API 문서**: http://localhost:8000/docs
- **프론트엔드**: http://localhost:8501

## 📋 주요 기능

### 백엔드 (FastAPI)
- 영화 등록/조회/삭제
- 포스터 이미지 자동 다운로드
- 리뷰 등록/조회/삭제
- AI 감성 분석 (자동)
- 영화 평점 계산

### 프론트엔드 (Streamlit)
- 영화 관리 페이지
- 리뷰 게시판
- AI 평점 시각화 (Gauge Chart)
- 반응형 UI

## 🛠 기술 스택

- **Backend**: FastAPI, SQLModel, SQLite, Uvicorn
- **Frontend**: Streamlit, Plotly, Requests
- **Database**: SQLite
- **Container**: Docker, Docker Compose

## 📝 API 엔드포인트

### 영화 (Movies)
- `POST /movies/` - 영화 등록
- `GET /movies/` - 전체 영화 목록
- `GET /movies/{movie_id}` - 특정 영화 조회
- `DELETE /movies/{movie_id}` - 영화 삭제

### 리뷰 (Reviews)
- `POST /reviews/` - 리뷰 등록
- `GET /reviews/` - 최근 리뷰 목록
- `GET /reviews/movie/{movie_id}` - 특정 영화 리뷰
- `GET /reviews/movie/{movie_id}/rating` - AI 평점
- `DELETE /reviews/{review_id}` - 리뷰 삭제

## 🔧 환경 변수

### Frontend
- `API_BASE_URL`: 백엔드 API 주소 (기본값: http://localhost:8000)

### Backend
- `PORT`: 서버 포트 (기본값: 8000)

## 📦 배포

### Google Cloud Run 배포
각 서비스를 개별적으로 빌드하고 배포할 수 있습니다.

```bash
# 백엔드 배포
cd backend
gcloud run deploy movie-review-backend --source .

# 프론트엔드 배포
cd frontend
gcloud run deploy movie-review-frontend --source .
```
