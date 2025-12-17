"""
FastAPI 유닛 테스트
"""

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool
from datetime import datetime
from typing import Generator

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from app.main import app
from app.database import get_db, db_connector


# 테스트용 인메모리 데이터베이스 설정
@pytest.fixture(name="session")
def session_fixture():
    """
    테스트용 데이터베이스 세션 생성
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture(name="client")
def client_fixture(session: Session):
    """
    테스트 클라이언트 생성

    Note: get_db를 override하지만, generator 형태를 유지하여
    실제 코드의 동작 방식을 더 잘 반영합니다.
    """

    def get_session_override():
        """실제 get_db()와 동일하게 generator로 yield"""
        yield session

    app.dependency_overrides[get_db] = get_session_override
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


@pytest.fixture(name="client_with_real_db")
def client_with_real_db_fixture():
    """
    실제 get_db() 함수를 테스트하기 위한 클라이언트
    override 없이 실제 데이터베이스 연결 사용
    """
    # 테스트용 임시 데이터베이스 파일
    test_db_path = Path(__file__).parent / "test_temp.db"

    # 기존 파일 삭제
    if test_db_path.exists():
        test_db_path.unlink()

    # 임시 데이터베이스로 테스트
    from app.database import db_connector

    original_engine = db_connector.engine

    # 테스트용 엔진으로 교체
    test_engine = create_engine(
        f"sqlite:///{test_db_path}",
        connect_args={"check_same_thread": False},
    )
    db_connector.engine = test_engine
    SQLModel.metadata.create_all(test_engine)

    client = TestClient(app)
    yield client

    # 엔진 정리 및 원래 엔진으로 복구
    test_engine.dispose()  # 모든 연결 닫기
    db_connector.engine = original_engine

    # 테스트 DB 파일 삭제 (시도)
    import time

    for _ in range(3):  # 3번 재시도
        try:
            if test_db_path.exists():
                test_db_path.unlink()
            break
        except PermissionError:
            time.sleep(0.1)  # 잠시 대기 후 재시도


# ==================== Health Check Tests ====================


def test_health_check(client: TestClient):
    """헬스체크 엔드포인트 테스트"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "message" in data


def test_health_endpoint(client: TestClient):
    """헬스 엔드포인트 테스트"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ==================== Movie Tests ====================


def test_create_movie_success(client: TestClient):
    """영화 등록 성공 테스트"""
    movie_data = {
        "tmdb_id": 12345,
        "title": "테스트 영화",
        "release_date": "2024-01-01",
        "director": "홍길동",
        "genre": "액션",
        "poster_url": None,
        "tmdb_rating": 8.5,
    }
    response = client.post("/movies/", json=movie_data)
    assert response.status_code == 201
    data = response.json()
    assert data["title"] == "테스트 영화"
    assert data["tmdb_id"] == 12345
    assert data["director"] == "홍길동"


def test_create_movie_duplicate_tmdb_id(client: TestClient):
    """중복된 TMDB ID로 영화 등록 시도 테스트"""
    movie_data = {
        "tmdb_id": 99999,
        "title": "첫 번째 영화",
        "release_date": "2024-01-01",
        "director": "감독A",
        "genre": "드라마",
        "poster_url": None,
        "tmdb_rating": 7.0,
    }
    # 첫 번째 등록
    response1 = client.post("/movies/", json=movie_data)
    assert response1.status_code == 201
    assert response1.json()["title"] == "첫 번째 영화"

    # 중복 등록 시도
    response2 = client.post("/movies/", json=movie_data)
    assert response2.status_code == 400
    error_detail = response2.json()["detail"]
    assert "이미 등록된 영화" in error_detail
    assert "99999" in error_detail
    assert "첫 번째 영화" in error_detail


def test_get_all_movies_empty(client: TestClient):
    """영화 목록 조회 - 빈 목록 테스트"""
    response = client.get("/movies/")
    assert response.status_code == 200
    assert response.json() == []


def test_get_all_movies_with_data(client: TestClient):
    """영화 목록 조회 - 데이터 있음 테스트"""
    # 영화 2개 등록
    movies = [
        {
            "tmdb_id": 1001,
            "title": "영화1",
            "release_date": "2024-01-01",
            "director": "감독1",
            "genre": "액션",
            "poster_url": None,
            "tmdb_rating": 8.0,
        },
        {
            "tmdb_id": 1002,
            "title": "영화2",
            "release_date": "2024-02-01",
            "director": "감독2",
            "genre": "코미디",
            "poster_url": None,
            "tmdb_rating": 7.5,
        },
    ]
    for movie in movies:
        client.post("/movies/", json=movie)

    response = client.get("/movies/")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2


def test_get_movie_by_id_success(client: TestClient):
    """특정 영화 조회 성공 테스트"""
    # 영화 등록
    movie_data = {
        "tmdb_id": 2001,
        "title": "조회 테스트 영화",
        "release_date": "2024-03-01",
        "director": "테스트 감독",
        "genre": "스릴러",
        "poster_url": None,
        "tmdb_rating": 8.8,
    }
    create_response = client.post("/movies/", json=movie_data)
    movie_id = create_response.json()["id"]

    # 영화 조회
    response = client.get(f"/movies/{movie_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "조회 테스트 영화"
    assert "reviews" in data


def test_get_movie_by_id_not_found(client: TestClient):
    """존재하지 않는 영화 조회 테스트"""
    response = client.get("/movies/99999")
    assert response.status_code == 404
    assert "찾을 수 없습니다" in response.json()["detail"]


def test_delete_movie_success(client: TestClient):
    """영화 삭제 성공 테스트"""
    # 영화 등록
    movie_data = {
        "tmdb_id": 3001,
        "title": "삭제 테스트 영화",
        "release_date": "2024-04-01",
        "director": "삭제 감독",
        "genre": "호러",
        "poster_url": None,
        "tmdb_rating": 6.5,
    }
    create_response = client.post("/movies/", json=movie_data)
    movie_id = create_response.json()["id"]

    # 영화 삭제
    response = client.delete(f"/movies/{movie_id}")
    assert response.status_code == 204

    # 삭제 확인
    get_response = client.get(f"/movies/{movie_id}")
    assert get_response.status_code == 404


def test_delete_movie_not_found(client: TestClient):
    """존재하지 않는 영화 삭제 시도 테스트"""
    response = client.delete("/movies/99999")
    assert response.status_code == 404


# ==================== Review Tests ====================


def test_create_review_success(client: TestClient):
    """리뷰 등록 성공 테스트"""
    # 영화 먼저 등록
    movie_data = {
        "tmdb_id": 4001,
        "title": "리뷰 테스트 영화",
        "release_date": "2024-05-01",
        "director": "리뷰 감독",
        "genre": "SF",
        "poster_url": None,
        "tmdb_rating": 9.0,
    }
    movie_response = client.post("/movies/", json=movie_data)
    movie_id = movie_response.json()["id"]

    # 리뷰 등록
    review_data = {
        "movie_id": movie_id,
        "author": "테스터",
        "content": "정말 재미있는 영화였습니다!",
    }
    response = client.post("/reviews/", json=review_data)
    assert response.status_code == 201
    data = response.json()
    assert data["author"] == "테스터"
    assert data["content"] == "정말 재미있는 영화였습니다!"
    assert data["is_positive"] in [0, 1]  # AI 감성 분석 결과


def test_create_review_movie_not_found(client: TestClient):
    """존재하지 않는 영화에 리뷰 등록 시도 테스트"""
    review_data = {
        "movie_id": 99999,
        "author": "테스터",
        "content": "테스트 리뷰",
    }
    response = client.post("/reviews/", json=review_data)
    assert response.status_code == 404
    assert "찾을 수 없습니다" in response.json()["detail"]


def test_get_recent_reviews_empty(client: TestClient):
    """리뷰 목록 조회 - 빈 목록 테스트"""
    response = client.get("/reviews/")
    assert response.status_code == 200
    assert response.json() == []


def test_get_recent_reviews_with_limit(client: TestClient):
    """리뷰 목록 조회 - limit 파라미터 테스트"""
    # 영화 등록
    movie_data = {
        "tmdb_id": 5001,
        "title": "리뷰 다수 영화",
        "release_date": "2024-06-01",
        "director": "다수 감독",
        "genre": "드라마",
        "poster_url": None,
        "tmdb_rating": 7.0,
    }
    movie_response = client.post("/movies/", json=movie_data)
    movie_id = movie_response.json()["id"]

    # 리뷰 5개 등록
    for i in range(5):
        review_data = {
            "movie_id": movie_id,
            "author": f"테스터{i}",
            "content": f"테스트 리뷰 {i}",
        }
        client.post("/reviews/", json=review_data)

    # 3개만 조회
    response = client.get("/reviews/?limit=3")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3


def test_get_reviews_by_movie_id(client: TestClient):
    """특정 영화의 리뷰 목록 조회 테스트"""
    # 영화 등록
    movie_data = {
        "tmdb_id": 6001,
        "title": "특정 영화",
        "release_date": "2024-07-01",
        "director": "특정 감독",
        "genre": "판타지",
        "poster_url": None,
        "tmdb_rating": 8.2,
    }
    movie_response = client.post("/movies/", json=movie_data)
    movie_id = movie_response.json()["id"]

    # 리뷰 3개 등록
    for i in range(3):
        review_data = {
            "movie_id": movie_id,
            "author": f"작성자{i}",
            "content": f"리뷰 내용 {i}",
        }
        client.post("/reviews/", json=review_data)

    # 영화별 리뷰 조회
    response = client.get(f"/reviews/movie/{movie_id}")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    assert all(review["movie_id"] == movie_id for review in data)


def test_get_reviews_by_movie_id_not_found(client: TestClient):
    """존재하지 않는 영화의 리뷰 조회 테스트"""
    response = client.get("/reviews/movie/99999")
    assert response.status_code == 404


def test_get_movie_rating(client: TestClient):
    """영화 평점 조회 테스트"""
    # 영화 등록
    movie_data = {
        "tmdb_id": 7001,
        "title": "평점 테스트 영화",
        "release_date": "2024-08-01",
        "director": "평점 감독",
        "genre": "로맨스",
        "poster_url": None,
        "tmdb_rating": 7.8,
    }
    movie_response = client.post("/movies/", json=movie_data)
    movie_id = movie_response.json()["id"]

    # 리뷰 등록
    review_data = {
        "movie_id": movie_id,
        "author": "평가자",
        "content": "훌륭한 영화!",
    }
    client.post("/reviews/", json=review_data)

    # 평점 조회
    response = client.get(f"/reviews/movie/{movie_id}/rating")
    assert response.status_code == 200
    data = response.json()
    assert data["movie_id"] == movie_id
    assert data["title"] == "평점 테스트 영화"
    assert "total_reviews" in data
    assert "positive_reviews" in data
    assert "negative_reviews" in data
    assert "positive_ratio" in data
    assert "ai_rating" in data
    assert 0 <= data["ai_rating"] <= 5.0


def test_get_movie_rating_no_reviews(client: TestClient):
    """리뷰가 없는 영화의 평점 조회 테스트"""
    # 영화 등록
    movie_data = {
        "tmdb_id": 8001,
        "title": "리뷰 없는 영화",
        "release_date": "2024-09-01",
        "director": "무명 감독",
        "genre": "다큐멘터리",
        "poster_url": None,
        "tmdb_rating": 6.0,
    }
    movie_response = client.post("/movies/", json=movie_data)
    movie_id = movie_response.json()["id"]

    # 평점 조회
    response = client.get(f"/reviews/movie/{movie_id}/rating")
    assert response.status_code == 200
    data = response.json()
    assert data["total_reviews"] == 0
    assert data["ai_rating"] == 0.0


def test_get_review_by_id(client: TestClient):
    """특정 리뷰 조회 테스트"""
    # 영화 등록
    movie_data = {
        "tmdb_id": 9001,
        "title": "개별 리뷰 영화",
        "release_date": "2024-10-01",
        "director": "개별 감독",
        "genre": "애니메이션",
        "poster_url": None,
        "tmdb_rating": 8.5,
    }
    movie_response = client.post("/movies/", json=movie_data)
    movie_id = movie_response.json()["id"]

    # 리뷰 등록
    review_data = {
        "movie_id": movie_id,
        "author": "개별 작성자",
        "content": "개별 리뷰 내용",
    }
    review_response = client.post("/reviews/", json=review_data)
    review_id = review_response.json()["id"]

    # 리뷰 조회
    response = client.get(f"/reviews/{review_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["author"] == "개별 작성자"
    assert "movie" in data


def test_get_review_by_id_not_found(client: TestClient):
    """존재하지 않는 리뷰 조회 테스트"""
    response = client.get("/reviews/99999")
    assert response.status_code == 404


def test_delete_review_success(client: TestClient):
    """리뷰 삭제 성공 테스트"""
    # 영화 등록
    movie_data = {
        "tmdb_id": 10001,
        "title": "삭제 리뷰 영화",
        "release_date": "2024-11-01",
        "director": "삭제 리뷰 감독",
        "genre": "뮤지컬",
        "poster_url": None,
        "tmdb_rating": 7.2,
    }
    movie_response = client.post("/movies/", json=movie_data)
    movie_id = movie_response.json()["id"]

    # 리뷰 등록
    review_data = {
        "movie_id": movie_id,
        "author": "삭제 작성자",
        "content": "삭제될 리뷰",
    }
    review_response = client.post("/reviews/", json=review_data)
    review_id = review_response.json()["id"]

    # 리뷰 삭제
    response = client.delete(f"/reviews/{review_id}")
    assert response.status_code == 204

    # 삭제 확인
    get_response = client.get(f"/reviews/{review_id}")
    assert get_response.status_code == 404


def test_delete_review_not_found(client: TestClient):
    """존재하지 않는 리뷰 삭제 시도 테스트"""
    response = client.delete("/reviews/99999")
    assert response.status_code == 404


# ==================== Integration Tests ====================


def test_full_workflow(client: TestClient):
    """전체 워크플로우 통합 테스트"""
    # 1. 영화 등록
    movie_data = {
        "tmdb_id": 11001,
        "title": "통합 테스트 영화",
        "release_date": "2024-12-01",
        "director": "통합 감독",
        "genre": "액션",
        "poster_url": None,
        "tmdb_rating": 8.9,
    }
    movie_response = client.post("/movies/", json=movie_data)
    assert movie_response.status_code == 201
    movie_id = movie_response.json()["id"]

    # 2. 리뷰 3개 등록
    for i in range(3):
        review_data = {
            "movie_id": movie_id,
            "author": f"통합작성자{i}",
            "content": f"통합 리뷰 {i}",
        }
        review_response = client.post("/reviews/", json=review_data)
        assert review_response.status_code == 201

    # 3. 영화 조회 (리뷰 포함)
    movie_get_response = client.get(f"/movies/{movie_id}")
    assert movie_get_response.status_code == 200
    movie_data = movie_get_response.json()
    assert len(movie_data["reviews"]) == 3

    # 4. 평점 조회
    rating_response = client.get(f"/reviews/movie/{movie_id}/rating")
    assert rating_response.status_code == 200
    rating_data = rating_response.json()
    assert rating_data["total_reviews"] == 3

    # 5. 영화 삭제 (리뷰도 함께 삭제됨)
    delete_response = client.delete(f"/movies/{movie_id}")
    assert delete_response.status_code == 204

    # 6. 삭제 확인
    movie_check = client.get(f"/movies/{movie_id}")
    assert movie_check.status_code == 404


# ==================== Database Connection Tests ====================


def test_get_db_returns_generator():
    """get_db() 함수가 제대로 generator를 반환하는지 테스트"""
    result = get_db()
    assert isinstance(result, Generator), "get_db()는 Generator를 반환해야 합니다"

    # generator에서 세션을 가져올 수 있는지 확인
    session = next(result)
    assert isinstance(
        session, Session
    ), "get_db()가 yield한 객체는 Session이어야 합니다"


def test_database_connector_session():
    """DatabaseConnector의 세션 생성이 제대로 작동하는지 테스트"""
    session_gen = db_connector.get_session()
    assert isinstance(
        session_gen, Generator
    ), "get_session()은 Generator를 반환해야 합니다"

    session = next(session_gen)
    assert isinstance(
        session, Session
    ), "get_session()이 yield한 객체는 Session이어야 합니다"


def test_real_db_workflow(client_with_real_db: TestClient):
    """
    실제 get_db() 함수를 사용한 전체 워크플로우 테스트
    이 테스트는 override 없이 실제 데이터베이스 연결을 사용합니다.
    """
    # 1. 영화 등록
    movie_data = {
        "tmdb_id": 99001,
        "title": "실제 DB 테스트 영화",
        "release_date": "2024-12-01",
        "director": "실제 DB 감독",
        "genre": "액션",
        "poster_url": None,
        "tmdb_rating": 8.0,
    }
    movie_response = client_with_real_db.post("/movies/", json=movie_data)
    assert movie_response.status_code == 201
    movie_id = movie_response.json()["id"]

    # 2. 리뷰 등록
    review_data = {
        "movie_id": movie_id,
        "author": "실제 DB 테스터",
        "content": "실제 데이터베이스 연결 테스트입니다.",
    }
    review_response = client_with_real_db.post("/reviews/", json=review_data)
    assert review_response.status_code == 201

    # 3. 조회
    get_response = client_with_real_db.get(f"/movies/{movie_id}")
    assert get_response.status_code == 200

    # 4. 정리
    delete_response = client_with_real_db.delete(f"/movies/{movie_id}")
    assert delete_response.status_code == 204


def test_poster_path_format(client: TestClient):
    """
    포스터 경로 형식 테스트
    poster_local_path가 올바른 형식(posters/파일명)으로 저장되는지 확인
    """
    movie_data = {
        "tmdb_id": 12345,
        "title": "포스터 경로 테스트",
        "release_date": "2024-01-01",
        "director": "테스트 감독",
        "genre": "액션",
        "poster_url": None,  # URL 없이 테스트
        "tmdb_rating": 8.0,
    }

    response = client.post("/movies/", json=movie_data)
    assert response.status_code == 201
    data = response.json()

    # poster_url이 None이면 poster_local_path도 None이어야 함
    assert data["poster_local_path"] is None

    # 영화 조회 시에도 동일한 형식이어야 함
    movie_id = data["id"]
    get_response = client.get(f"/movies/{movie_id}")
    assert get_response.status_code == 200
    get_data = get_response.json()
    assert get_data["poster_local_path"] is None


def test_poster_path_no_leading_slash(client: TestClient):
    """
    포스터 경로에 /data이 포함되지 않는지 테스트
    프론트엔드에서 /data을 추가하므로 백엔드는 posters/파일명만 반환
    """
    movie_data = {
        "tmdb_id": 54321,
        "title": "경로 슬래시 테스트",
        "release_date": "2024-01-01",
        "director": "테스트 감독",
        "genre": "드라마",
        "poster_url": None,
        "tmdb_rating": 7.5,
    }

    response = client.post("/movies/", json=movie_data)
    assert response.status_code == 201
    data = response.json()

    # poster_local_path가 있다면 data/posters/ 형식이어야 함
    if data["poster_local_path"]:
        assert not data["poster_local_path"].startswith("/data")
        assert data["poster_local_path"].startswith("data/posters/")


def test_get_movies_paginated_default(client: TestClient):
    """
    영화 목록 페이지네이션 기본 테스트
    """
    # 영화 5개 등록
    for i in range(5):
        movie_data = {
            "tmdb_id": 10000 + i,
            "title": f"페이지네이션 테스트 영화 {i+1}",
            "release_date": "2024-01-01",
            "director": "테스트 감독",
            "genre": "액션",
            "poster_url": None,
            "tmdb_rating": 8.0,
        }
        client.post("/movies/", json=movie_data)

    # 페이지네이션 조회 (기본값: page=1, page_size=10)
    response = client.get("/movies/paginated")
    assert response.status_code == 200
    data = response.json()

    assert data["total"] == 5
    assert data["page"] == 1
    assert data["page_size"] == 10
    assert data["total_pages"] == 1
    assert len(data["movies"]) == 5


def test_get_movies_paginated_with_params(client: TestClient):
    """
    페이지네이션 파라미터 테스트
    """
    # 영화 15개 등록
    for i in range(15):
        movie_data = {
            "tmdb_id": 20000 + i,
            "title": f"페이지 테스트 영화 {i+1}",
            "release_date": "2024-01-01",
            "director": "테스트 감독",
            "genre": "드라마",
            "poster_url": None,
            "tmdb_rating": 7.5,
        }
        client.post("/movies/", json=movie_data)

    # 첫 번째 페이지 (5개씩)
    response1 = client.get("/movies/paginated?page=1&page_size=5")
    assert response1.status_code == 200
    data1 = response1.json()
    assert data1["total"] == 15
    assert data1["page"] == 1
    assert data1["page_size"] == 5
    assert data1["total_pages"] == 3
    assert len(data1["movies"]) == 5

    # 두 번째 페이지
    response2 = client.get("/movies/paginated?page=2&page_size=5")
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["page"] == 2
    assert len(data2["movies"]) == 5

    # 세 번째 페이지 (마지막 페이지)
    response3 = client.get("/movies/paginated?page=3&page_size=5")
    assert response3.status_code == 200
    data3 = response3.json()
    assert data3["page"] == 3
    assert len(data3["movies"]) == 5


def test_get_movies_paginated_with_reviews(client: TestClient):
    """
    페이지네이션에 리뷰 포함 테스트
    """
    # 영화 등록
    movie_data = {
        "tmdb_id": 30000,
        "title": "리뷰 포함 페이지네이션 테스트",
        "release_date": "2024-01-01",
        "director": "테스트 감독",
        "genre": "코미디",
        "poster_url": None,
        "tmdb_rating": 8.5,
    }
    movie_response = client.post("/movies/", json=movie_data)
    movie_id = movie_response.json()["id"]

    # 리뷰 2개 등록
    for i in range(2):
        review_data = {
            "movie_id": movie_id,
            "author": f"리뷰어 {i+1}",
            "content": f"테스트 리뷰 {i+1}",
        }
        client.post("/reviews/", json=review_data)

    # 페이지네이션 조회
    response = client.get("/movies/paginated")
    assert response.status_code == 200
    data = response.json()

    # 영화에 리뷰가 포함되어 있는지 확인
    movie = data["movies"][0]
    assert "reviews" in movie
    assert len(movie["reviews"]) == 2

    # AI 평점 정보가 포함되어 있는지 확인
    assert "total_reviews" in movie
    assert "positive_reviews" in movie
    assert "negative_reviews" in movie
    assert "positive_ratio" in movie
    assert "ai_rating" in movie
    assert movie["total_reviews"] == 2


def test_movies_paginated_ai_rating_calculation(client: TestClient):
    """
    페이지네이션 API의 AI 평점 계산 테스트
    """
    # 영화 등록
    movie_data = {
        "tmdb_id": 40000,
        "title": "AI 평점 계산 테스트",
        "release_date": "2024-01-01",
        "director": "테스트 감독",
        "genre": "드라마",
        "poster_url": None,
        "tmdb_rating": 8.0,
    }
    movie_response = client.post("/movies/", json=movie_data)
    movie_id = movie_response.json()["id"]

    # 리뷰 4개 등록 (AI가 실제로 분석)
    reviews = [
        {
            "movie_id": movie_id,
            "author": "리뷰어 1",
            "content": "정말 재미있었습니다! 강력 추천합니다.",
        },
        {
            "movie_id": movie_id,
            "author": "리뷰어 2",
            "content": "최고의 영화입니다. 감동적이에요.",
        },
        {
            "movie_id": movie_id,
            "author": "리뷰어 3",
            "content": "훌륭한 작품입니다.",
        },
        {
            "movie_id": movie_id,
            "author": "리뷰어 4",
            "content": "별로였습니다. 실망스러웠어요.",
        },
    ]

    for review_data in reviews:
        client.post("/reviews/", json=review_data)

    # 페이지네이션 조회
    response = client.get("/movies/paginated")
    assert response.status_code == 200
    data = response.json()

    movie = data["movies"][0]

    # 기본 필드 존재 확인
    assert "total_reviews" in movie
    assert "positive_reviews" in movie
    assert "negative_reviews" in movie
    assert "positive_ratio" in movie
    assert "ai_rating" in movie

    # 전체 리뷰 수 확인
    assert movie["total_reviews"] == 4

    # 긍정/부정 리뷰 합이 전체 리뷰 수와 일치하는지 확인
    assert movie["positive_reviews"] + movie["negative_reviews"] == 4

    # AI 평점이 0~5 사이인지 확인
    assert 0.0 <= movie["ai_rating"] <= 5.0

    # 긍정 비율이 0~1 사이인지 확인
    assert 0.0 <= movie["positive_ratio"] <= 1.0
