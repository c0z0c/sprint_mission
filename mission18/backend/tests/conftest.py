"""
Pytest 설정 및 Hook
"""

from datetime import datetime
from pathlib import Path
import pytest


# 테스트 결과를 저장할 딕셔너리
test_results = {}


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
