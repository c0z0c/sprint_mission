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


def get_test_category(nodeid: str) -> str:
    """
    테스트 파일명으로 카테고리 자동 추론

    Args:
        nodeid: pytest의 테스트 노드 ID (예: "tests/test_movies.py::test_create_movie_success")

    Returns:
        카테고리 문자열
    """
    if "test_database" in nodeid:
        return "Database"
    elif "test_health" in nodeid:
        return "Health Check"
    elif "test_movie" in nodeid:
        return "Movie"
    elif "test_review" in nodeid or "test_sentiment" in nodeid:
        return "Review"
    elif "test_datetime" in nodeid:
        return "DateTime"
    elif "test_integration" in nodeid:
        return "Integration"
    return "Unknown"


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

    # 테스트 설명 매핑 (한글 설명문만 관리, 카테고리는 자동 추론)
    test_descriptions = {
        # Database
        "test_get_db_returns_generator": "get_db() generator 반환 테스트",
        "test_database_connector_session": "DatabaseConnector 세션 생성 테스트",
        "test_real_db_workflow": "실제 DB 연결 워크플로우 테스트",
        # DateTime
        "test_create_review_without_datetime": "리뷰 생성 - 날짜 자동 설정",
        "test_create_review_with_custom_datetime": "리뷰 생성 - 날짜 수동 설정",
        "test_update_review_without_datetime": "리뷰 수정 - 날짜 자동 업데이트",
        "test_update_review_with_custom_datetime": "리뷰 수정 - 날짜 수동 설정",
        "test_patch_review_without_datetime": "리뷰 부분 수정 - 날짜 자동 업데이트",
        "test_patch_review_with_custom_datetime": "리뷰 부분 수정 - 날짜 수동 설정",
        # Health Check
        "test_health_check": "헬스체크 엔드포인트 테스트",
        "test_health_endpoint": "헬스 엔드포인트 테스트",
        "test_health_endpoint_initial_sync_status": "헬스 엔드포인트 - 초기 동기화 상태",
        # Integration
        "test_full_workflow": "전체 워크플로우 통합 테스트",
        # Movie
        "test_create_movie_success": "영화 등록 성공",
        "test_create_movie_duplicate_tmdb_id": "중복 TMDB ID 등록 시도",
        "test_get_all_movies_empty": "영화 목록 조회 - 빈 목록",
        "test_get_all_movies_with_data": "영화 목록 조회 - 데이터 있음",
        "test_get_movie_by_id_success": "특정 영화 조회 성공",
        "test_get_movie_by_id_not_found": "존재하지 않는 영화 조회",
        "test_delete_movie_success": "영화 삭제 성공",
        "test_delete_movie_not_found": "존재하지 않는 영화 삭제 시도",
        "test_poster_path_format": "포스터 경로 형식 검증",
        "test_poster_path_no_leading_slash": "포스터 경로 슬래시 제거 검증",
        "test_get_movies_paginated_default": "영화 목록 페이징 - 기본값",
        "test_get_movies_paginated_with_params": "영화 목록 페이징 - 파라미터 지정",
        "test_get_movies_paginated_with_reviews": "영화 목록 페이징 - 리뷰 포함",
        "test_movies_paginated_ai_rating_calculation": "영화 목록 - AI 평점 계산",
        "test_search_movies_by_title": "영화 검색 - 제목",
        "test_search_movies_by_director": "영화 검색 - 감독",
        "test_search_movies_by_genre": "영화 검색 - 장르",
        "test_search_movies_by_release_date_range": "영화 검색 - 개봉일 범위",
        "test_search_movies_by_tmdb_rating_range": "영화 검색 - TMDB 평점 범위",
        "test_search_movies_multiple_filters": "영화 검색 - 복합 필터",
        "test_search_movies_with_sorting": "영화 검색 - 정렬",
        "test_search_movies_empty_result": "영화 검색 - 결과 없음",
        "test_search_movies_pagination": "영화 검색 - 페이징",
        "test_update_movie_put_success": "영화 전체 수정 성공",
        "test_update_movie_patch_success": "영화 부분 수정 성공",
        "test_update_movie_not_found": "존재하지 않는 영화 수정 시도",
        "test_update_movie_poster_change": "영화 수정 - 포스터 변경",
        # Review (리뷰 + AI 감성 분석 통합)
        "test_create_review_success": "리뷰 등록 성공",
        "test_create_review_movie_not_found": "존재하지 않는 영화에 리뷰 등록 시도",
        "test_get_recent_reviews_empty": "리뷰 목록 조회 - 빈 목록",
        "test_get_recent_reviews_with_limit": "리뷰 목록 조회 - limit 파라미터",
        "test_get_reviews_by_movie_id": "특정 영화의 리뷰 목록 조회",
        "test_get_reviews_by_movie_id_not_found": "존재하지 않는 영화의 리뷰 조회",
        "test_get_movie_rating": "영화 평점 조회",
        "test_get_movie_rating_no_reviews": "리뷰 없는 영화의 평점 조회",
        "test_get_review_by_id": "특정 리뷰 조회",
        "test_get_review_by_id_not_found": "존재하지 않는 리뷰 조회",
        "test_delete_review_success": "리뷰 삭제 성공",
        "test_delete_review_not_found": "존재하지 않는 리뷰 삭제 시도",
        "test_get_reviews_paginated_success": "리뷰 목록 페이징 - 정상",
        "test_get_reviews_paginated_empty": "리뷰 목록 페이징 - 빈 목록",
        "test_get_reviews_paginated_out_of_range": "리뷰 목록 페이징 - 범위 초과",
        "test_get_reviews_paginated_metadata": "리뷰 목록 페이징 - 메타데이터 검증",
        "test_search_reviews_by_author": "리뷰 검색 - 작성자",
        "test_search_reviews_by_content": "리뷰 검색 - 내용",
        "test_search_reviews_by_movie_title": "리뷰 검색 - 영화 제목",
        "test_search_reviews_by_tmdb_id": "리뷰 검색 - TMDB ID",
        "test_search_reviews_multiple_filters": "리뷰 검색 - 복합 필터",
        "test_search_reviews_with_sorting": "리뷰 검색 - 정렬",
        "test_search_reviews_pagination": "리뷰 검색 - 페이징",
        "test_search_reviews_empty_result": "리뷰 검색 - 결과 없음",
        "test_update_review_put_success": "리뷰 전체 수정 성공",
        "test_update_review_patch_success": "리뷰 부분 수정 성공",
        "test_update_review_content_triggers_ai_reanalysis": "리뷰 수정 시 AI 재분석 트리거",
        "test_update_review_not_found": "존재하지 않는 리뷰 수정 시도",
        "test_update_review_unique_constraint_violation": "리뷰 수정 - 유니크 제약 위반",
        "test_create_review_with_ai_positive_sentiment": "리뷰 생성 - AI 긍정 감성",
        "test_create_review_with_ai_negative_sentiment": "리뷰 생성 - AI 부정 감성",
        "test_update_review_content_ai_reanalysis": "리뷰 내용 수정 시 AI 재분석",
        "test_movie_ai_rating_calculation": "영화 AI 평점 계산",
        "test_ai_sentiment_not_random": "AI 감성 분석 - 재현성 테스트",
        "test_model_initialization": "AI 모델 초기화",
        "test_predict_positive_samples": "AI 예측 - 긍정 샘플",
        "test_predict_negative_samples": "AI 예측 - 부정 샘플",
        "test_predict_batch": "AI 예측 - 배치 처리",
        "test_long_text_chunking": "AI 예측 - 긴 텍스트 청킹",
        "test_predict_texts_batch_with_long_texts": "AI 예측 - 긴 텍스트 배치",
        "test_edge_case_empty_string": "AI 예측 - 빈 문자열 엣지 케이스",
        "test_edge_case_none": "AI 예측 - None 엣지 케이스",
        "test_edge_case_special_characters": "AI 예측 - 특수문자 엣지 케이스",
        "test_mixed_sentiment_majority_vote": "AI 예측 - 혼합 감성 다수결",
        "test_prediction_consistency": "AI 예측 - 일관성 검증",
        "test_batch_vs_single_prediction": "AI 예측 - 배치 vs 단일 비교",
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

            # 카테고리 자동 추론
            category = get_test_category(nodeid)

            # 테스트 설명 가져오기 (없으면 테스트명을 설명으로 사용)
            description = test_descriptions.get(test_name, test_name)

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
