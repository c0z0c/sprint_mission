"""
Pytest 설정 및 Hook
"""

from datetime import datetime
from pathlib import Path
import pytest
import sys
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool
import time

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from app.main import app
from app.database import get_db, db_connector


# 테스트 결과를 저장할 딕셔너리
test_results = {}


# ==================== Pytest Fixtures ====================


@pytest.fixture(name="session")
def session_fixture():
    """
    테스트용 인메모리 데이터베이스 세션 생성
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
    for _ in range(3):  # 3번 재시도
        try:
            if test_db_path.exists():
                test_db_path.unlink()
            break
        except PermissionError:
            time.sleep(0.1)  # 잠시 대기 후 재시도


# ==================== Pytest Hooks ====================


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """각 테스트 실행 후 결과 수집"""
    outcome = yield
    report = outcome.get_result()

    # 테스트 실행(call) 단계에서만 결과 수집
    if report.when == "call":
        test_results[item.nodeid] = {
            "name": item.name,
            "outcome": report.outcome,  # passed, failed, skipped
            "duration": report.duration,
        }


def pytest_sessionfinish(session, exitstatus):
    """
    pytest 세션 종료 후 테스트 리포트 생성
    """
    # 현재 시간으로 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    report_path = Path(__file__).parent / f"unit_test_report_{timestamp}.md"

    # 테스트 결과 수집
    passed = session.testscollected - session.testsfailed
    failed = session.testsfailed
    total = session.testscollected

    # 테스트 설명 매핑
    test_descriptions = {
        "test_health_check": ("Health Check", "헬스체크 엔드포인트 테스트"),
        "test_health_endpoint": ("Health Check", "헬스 엔드포인트 테스트"),
        "test_create_movie_success": ("Movie", "영화 등록 성공 테스트"),
        "test_create_movie_duplicate_tmdb_id": (
            "Movie",
            "중복 TMDB ID 등록 시도 테스트",
        ),
        "test_get_all_movies_empty": ("Movie", "영화 목록 조회 - 빈 목록"),
        "test_get_all_movies_with_data": ("Movie", "영화 목록 조회 - 데이터 있음"),
        "test_get_movie_by_id_success": ("Movie", "특정 영화 조회 성공"),
        "test_get_movie_by_id_not_found": ("Movie", "존재하지 않는 영화 조회"),
        "test_delete_movie_success": ("Movie", "영화 삭제 성공"),
        "test_delete_movie_not_found": ("Movie", "존재하지 않는 영화 삭제 시도"),
        "test_create_review_success": ("Review", "리뷰 등록 성공"),
        "test_create_review_movie_not_found": (
            "Review",
            "존재하지 않는 영화에 리뷰 등록 시도",
        ),
        "test_get_recent_reviews_empty": ("Review", "리뷰 목록 조회 - 빈 목록"),
        "test_get_recent_reviews_with_limit": (
            "Review",
            "리뷰 목록 조회 - limit 파라미터",
        ),
        "test_get_reviews_by_movie_id": ("Review", "특정 영화의 리뷰 목록 조회"),
        "test_get_reviews_by_movie_id_not_found": (
            "Review",
            "존재하지 않는 영화의 리뷰 조회",
        ),
        "test_get_movie_rating": ("Review", "영화 평점 조회"),
        "test_get_movie_rating_no_reviews": ("Review", "리뷰 없는 영화의 평점 조회"),
        "test_get_review_by_id": ("Review", "특정 리뷰 조회"),
        "test_get_review_by_id_not_found": ("Review", "존재하지 않는 리뷰 조회"),
        "test_delete_review_success": ("Review", "리뷰 삭제 성공"),
        "test_delete_review_not_found": ("Review", "존재하지 않는 리뷰 삭제 시도"),
        "test_full_workflow": ("Integration", "전체 워크플로우 통합 테스트"),
        "test_get_db_returns_generator": ("Database", "get_db() generator 반환 테스트"),
        "test_database_connector_session": (
            "Database",
            "DatabaseConnector 세션 생성 테스트",
        ),
        "test_real_db_workflow": ("Database", "실제 DB 연결 워크플로우 테스트"),
    }

    # 마크다운 리포트 작성
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# 유닛 테스트 리포트\n\n")
        f.write(f"**생성 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"## 테스트 결과 요약\n\n")
        f.write(f"| 항목 | 값 |\n")
        f.write(f"|------|------|\n")
        f.write(f"| 총 테스트 수 | {total} |\n")
        f.write(f"| 성공 | {passed} |\n")
        f.write(f"| 실패 | {failed} |\n")

        if total > 0:
            f.write(f"| 성공률 | {(passed/total*100):.2f}% |\n\n")
        else:
            f.write(f"| 성공률 | N/A |\n\n")

        if exitstatus == 0:
            f.write(
                f"**전체 테스트 통과**: 모든 테스트가 성공적으로 완료되었습니다.\n\n"
            )
        else:
            f.write(f"**테스트 실패**: 일부 테스트가 실패했습니다.\n\n")

        # 테스트 상세 목록 표
        f.write(f"## 테스트 상세 목록\n\n")
        f.write(f"| 번호 | 카테고리 | 테스트명 | 설명 | 결과 |\n")
        f.write(f"|------|----------|----------|------|------|\n")

        idx = 1
        for nodeid, result_data in test_results.items():
            test_name = result_data["name"]
            outcome = result_data["outcome"]

            # 테스트 설명 가져오기
            if test_name in test_descriptions:
                category, description = test_descriptions[test_name]
            else:
                category = "Unknown"
                description = test_name

            # 결과 표시
            if outcome == "passed":
                result = "PASS"
            elif outcome == "failed":
                result = "FAIL"
            elif outcome == "skipped":
                result = "SKIP"
            else:
                result = outcome.upper()

            f.write(
                f"| {idx} | {category} | `{test_name}` | {description} | {result} |\n"
            )
            idx += 1

        f.write(f"\n---\n")

    print(f"\n테스트 리포트 생성 완료: {report_path}")
