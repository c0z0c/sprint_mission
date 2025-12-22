"""
통합 테스트 (Full Workflow)
"""

from fastapi.testclient import TestClient


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
            "tmdb_id": 11001,
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
    rating_response = client.get(f"/reviews/movie/11001/rating")
    assert rating_response.status_code == 200
    rating_data = rating_response.json()
    assert rating_data["total_reviews"] == 3

    # 5. 영화 삭제 (리뷰도 함께 삭제됨)
    delete_response = client.delete(f"/movies/{movie_id}")
    assert delete_response.status_code == 204

    # 6. 삭제 확인
    movie_check = client.get(f"/movies/{movie_id}")
    assert movie_check.status_code == 404
