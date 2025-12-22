"""
Health Check 테스트
"""

from fastapi.testclient import TestClient


def test_health_check(client: TestClient):
    """헬스체크 엔드포인트 테스트"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "message" in data


def test_health_endpoint(client: TestClient):
    """헬스 엔드포인트 테스트 (기본 상태)"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "ready" in data
    assert "initial_sync" in data


def test_health_endpoint_initial_sync_status(client: TestClient):
    """헬스 엔드포인트 초기 동기화 상태 테스트"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    # 초기 동기화 정보 확인
    initial_sync = data.get("initial_sync", {})
    assert "in_progress" in initial_sync
    assert "current" in initial_sync
    assert "total" in initial_sync
    assert "sync_type" in initial_sync

    # 데이터 타입 확인
    assert isinstance(initial_sync["in_progress"], bool)
    assert isinstance(initial_sync["current"], int)
    assert isinstance(initial_sync["total"], int)

    # ready 상태는 in_progress의 반대
    assert data["ready"] == (not initial_sync["in_progress"])
