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
    """헬스 엔드포인트 테스트"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
