"""
데이터베이스 연결 및 세션 테스트
"""

from fastapi.testclient import TestClient
from typing import Generator
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from app.database import get_db, db_connector
from sqlalchemy.orm import Session


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
    # get_session()은 contextmanager를 사용하므로 with 문으로 사용
    with db_connector.get_session() as session:
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
        "tmdb_id": 99001,
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
