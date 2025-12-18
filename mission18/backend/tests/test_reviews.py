"""
리뷰(Review) API 테스트
"""

from fastapi.testclient import TestClient


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
        "tmdb_id": 4001,
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
        "tmdb_id": 99999,
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
            "tmdb_id": 5001,
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
            "tmdb_id": 6001,
            "author": f"작성자{i}",
            "content": f"리뷰 내용 {i}",
        }
        client.post("/reviews/", json=review_data)

    # 영화별 리뷰 조회
    response = client.get(f"/reviews/movie/6001")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    assert all(review["tmdb_id"] == 6001 for review in data)


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
        "tmdb_id": 7001,
        "author": "평가자",
        "content": "훌륭한 영화!",
    }
    client.post("/reviews/", json=review_data)

    # 평점 조회
    response = client.get(f"/reviews/movie/7001/rating")
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
    response = client.get(f"/reviews/movie/8001/rating")
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
        "tmdb_id": 9001,
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
        "tmdb_id": 10001,
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


def test_get_reviews_paginated_success(client: TestClient):
    """리뷰 페이지네이션 정상 조회 테스트"""
    # 영화 등록
    movie_data = {
        "tmdb_id": 6001,
        "title": "페이지네이션 테스트 영화",
        "release_date": "2024-01-01",
        "director": "테스트 감독",
        "genre": "드라마",
        "poster_url": None,
        "tmdb_rating": 8.0,
    }
    client.post("/movies/", json=movie_data)

    # 리뷰 10개 등록
    for i in range(10):
        review_data = {
            "tmdb_id": 6001,
            "author": f"테스터{i}",
            "content": f"테스트 리뷰 {i}",
        }
        client.post("/reviews/", json=review_data)

    # 첫 번째 페이지 조회 (3개씩)
    response = client.get("/reviews/paginated?page=1&page_size=3")
    assert response.status_code == 200
    data = response.json()

    assert data["total"] == 10
    assert data["page"] == 1
    assert data["page_size"] == 3
    assert data["total_pages"] == 4
    assert len(data["reviews"]) == 3

    # 각 리뷰에 영화 정보가 포함되어 있는지 확인
    for review in data["reviews"]:
        assert "movie" in review
        assert review["movie"]["title"] == "페이지네이션 테스트 영화"

    # 두 번째 페이지 조회
    response2 = client.get("/reviews/paginated?page=2&page_size=3")
    assert response2.status_code == 200
    data2 = response2.json()
    assert len(data2["reviews"]) == 3

    # 마지막 페이지 조회
    response_last = client.get("/reviews/paginated?page=4&page_size=3")
    assert response_last.status_code == 200
    data_last = response_last.json()
    assert len(data_last["reviews"]) == 1  # 10 % 3 = 1


def test_get_reviews_paginated_empty(client: TestClient):
    """리뷰가 없을 때 페이지네이션 조회 테스트"""
    response = client.get("/reviews/paginated?page=1&page_size=10")
    assert response.status_code == 200
    data = response.json()

    assert data["total"] == 0
    assert data["page"] == 1
    assert data["page_size"] == 10
    assert data["total_pages"] == 0
    assert len(data["reviews"]) == 0


def test_get_reviews_paginated_out_of_range(client: TestClient):
    """범위 초과 페이지 조회 테스트"""
    # 영화 및 리뷰 등록
    movie_data = {
        "tmdb_id": 6002,
        "title": "범위 테스트 영화",
        "release_date": "2024-01-01",
    }
    client.post("/movies/", json=movie_data)

    review_data = {
        "tmdb_id": 6002,
        "author": "테스터",
        "content": "테스트 리뷰",
    }
    client.post("/reviews/", json=review_data)

    # 범위를 초과한 페이지 조회
    response = client.get("/reviews/paginated?page=10&page_size=10")
    assert response.status_code == 200
    data = response.json()

    assert data["total"] == 1
    assert data["page"] == 10
    assert data["total_pages"] == 1
    assert len(data["reviews"]) == 0  # 범위 초과 시 빈 리스트


def test_get_reviews_paginated_metadata(client: TestClient):
    """페이지네이션 메타데이터 검증 테스트"""
    # 영화 등록
    movie_data = {
        "tmdb_id": 6003,
        "title": "메타데이터 테스트 영화",
        "release_date": "2024-01-01",
    }
    client.post("/movies/", json=movie_data)

    # 리뷰 7개 등록
    for i in range(7):
        review_data = {
            "tmdb_id": 6003,
            "author": f"테스터{i}",
            "content": f"테스트 리뷰 {i}",
        }
        client.post("/reviews/", json=review_data)

    # 페이지당 5개씩 조회
    response = client.get("/reviews/paginated?page=1&page_size=5")
    assert response.status_code == 200
    data = response.json()

    assert data["total"] == 7
    assert data["page"] == 1
    assert data["page_size"] == 5
    assert data["total_pages"] == 2  # ceil(7/5) = 2
    assert len(data["reviews"]) == 5

    # 두 번째 페이지
    response2 = client.get("/reviews/paginated?page=2&page_size=5")
    data2 = response2.json()
    assert len(data2["reviews"]) == 2  # 7 - 5 = 2


# ==================== Review Search Tests ====================


def test_search_reviews_by_author(client: TestClient):
    """작성자 이름으로 리뷰 검색 테스트"""
    # 영화 등록
    client.post("/movies/", json={"tmdb_id": 60001, "title": "Test Movie"})

    # 리뷰 등록
    reviews = [
        {"tmdb_id": 60001, "author": "John Doe", "content": "Great movie!"},
        {"tmdb_id": 60001, "author": "Jane Smith", "content": "Not bad"},
        {"tmdb_id": 60001, "author": "John Kim", "content": "Amazing"},
    ]
    for review in reviews:
        client.post("/reviews/", json=review)

    # "john" 검색 (대소문자 무시)
    response = client.get("/reviews/search", params={"author": "john"})
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    authors = [r["author"] for r in data["reviews"]]
    assert "John Doe" in authors
    assert "John Kim" in authors


def test_search_reviews_by_content(client: TestClient):
    """리뷰 내용으로 검색 테스트"""
    client.post("/movies/", json={"tmdb_id": 60101, "title": "Movie A"})

    reviews = [
        {"tmdb_id": 60101, "author": "User1", "content": "This movie is amazing!"},
        {"tmdb_id": 60101, "author": "User2", "content": "Terrible experience"},
        {"tmdb_id": 60101, "author": "User3", "content": "The most amazing film"},
    ]
    for review in reviews:
        client.post("/reviews/", json=review)

    response = client.get("/reviews/search", params={"content": "amazing"})
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2


def test_search_reviews_by_movie_title(client: TestClient):
    """영화 제목으로 리뷰 검색 테스트"""
    movies = [
        {"tmdb_id": 60201, "title": "The Dark Knight"},
        {"tmdb_id": 60202, "title": "The Matrix"},
    ]
    for movie in movies:
        client.post("/movies/", json=movie)

    client.post(
        "/reviews/", json={"tmdb_id": 60201, "author": "User1", "content": "Review 1"}
    )
    client.post(
        "/reviews/", json={"tmdb_id": 60201, "author": "User2", "content": "Review 2"}
    )
    client.post(
        "/reviews/", json={"tmdb_id": 60202, "author": "User3", "content": "Review 3"}
    )

    response = client.get("/reviews/search", params={"movie_title": "dark knight"})
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2


def test_search_reviews_by_tmdb_id(client: TestClient):
    """TMDB ID로 리뷰 검색 테스트"""
    client.post("/movies/", json={"tmdb_id": 60301, "title": "Movie X"})
    client.post("/movies/", json={"tmdb_id": 60302, "title": "Movie Y"})

    client.post(
        "/reviews/", json={"tmdb_id": 60301, "author": "User1", "content": "Review 1"}
    )
    client.post(
        "/reviews/", json={"tmdb_id": 60301, "author": "User2", "content": "Review 2"}
    )
    client.post(
        "/reviews/", json={"tmdb_id": 60302, "author": "User3", "content": "Review 3"}
    )

    response = client.get("/reviews/search", params={"tmdb_id": 60301})
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2


def test_search_reviews_multiple_filters(client: TestClient):
    """복합 필터로 리뷰 검색 테스트 (AND 조합)"""
    client.post("/movies/", json={"tmdb_id": 60401, "title": "Test Film"})

    reviews = [
        {"tmdb_id": 60401, "author": "Alice", "content": "Great movie!"},
        {"tmdb_id": 60401, "author": "Bob", "content": "Not so great"},
        {"tmdb_id": 60401, "author": "Alice", "content": "Terrible"},
    ]
    for review in reviews:
        client.post("/reviews/", json=review)

    response = client.get(
        "/reviews/search", params={"author": "alice", "content": "great"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["reviews"][0]["author"] == "Alice"
    assert "Great movie" in data["reviews"][0]["content"]


def test_search_reviews_with_sorting(client: TestClient):
    """정렬 옵션으로 리뷰 검색 테스트"""
    import time

    client.post("/movies/", json={"tmdb_id": 60501, "title": "Movie"})

    # 시간차를 두고 등록
    client.post(
        "/reviews/", json={"tmdb_id": 60501, "author": "Charlie", "content": "First"}
    )
    time.sleep(0.1)
    client.post(
        "/reviews/", json={"tmdb_id": 60501, "author": "Alice", "content": "Second"}
    )
    time.sleep(0.1)
    client.post(
        "/reviews/", json={"tmdb_id": 60501, "author": "Bob", "content": "Third"}
    )

    # 작성자 이름 오름차순
    response = client.get(
        "/reviews/search", params={"sort_by": "author", "sort_order": "asc"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["reviews"][0]["author"] == "Alice"
    assert data["reviews"][2]["author"] == "Charlie"

    # 생성일 내림차순 (최신순)
    response = client.get(
        "/reviews/search", params={"sort_by": "created_at", "sort_order": "desc"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["reviews"][0]["content"] == "Third"  # 가장 최근


def test_search_reviews_pagination(client: TestClient):
    """리뷰 검색 페이지네이션 테스트"""
    client.post("/movies/", json={"tmdb_id": 60601, "title": "Popular Movie"})

    for i in range(12):
        client.post(
            "/reviews/",
            json={"tmdb_id": 60601, "author": f"User{i}", "content": f"Review {i}"},
        )

    response = client.get("/reviews/search", params={"page": 1, "page_size": 10})
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 12
    assert len(data["reviews"]) == 10
    assert data["total_pages"] == 2

    response = client.get("/reviews/search", params={"page": 2, "page_size": 10})
    assert response.status_code == 200
    data = response.json()
    assert len(data["reviews"]) == 2


def test_search_reviews_empty_result(client: TestClient):
    """리뷰 검색 결과가 없는 경우 테스트"""
    client.post("/movies/", json={"tmdb_id": 60701, "title": "Movie"})
    client.post(
        "/reviews/", json={"tmdb_id": 60701, "author": "User", "content": "Review"}
    )

    response = client.get("/reviews/search", params={"author": "NonExistent"})
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert len(data["reviews"]) == 0


# ==================== Review Update Tests ====================


def test_update_review_put_success(client: TestClient):
    """리뷰 전체 업데이트 (PUT) 성공 테스트"""
    # 영화 및 리뷰 등록
    client.post("/movies/", json={"tmdb_id": 70001, "title": "Test Movie"})
    review_data = {
        "tmdb_id": 70001,
        "author": "Original Author",
        "content": "Original content",
    }
    create_response = client.post("/reviews/", json=review_data)
    assert create_response.status_code == 201
    review_id = create_response.json()["id"]
    original_is_positive = create_response.json()["is_positive"]

    # 전체 업데이트 (content 변경 -> AI 재분석)
    update_data = {
        "author": "Updated Author",
        "content": "This movie was absolutely amazing and fantastic!",
    }
    response = client.put(f"/reviews/{review_id}", json=update_data)
    assert response.status_code == 200
    data = response.json()
    assert data["author"] == "Updated Author"
    assert data["content"] == "This movie was absolutely amazing and fantastic!"
    assert data["is_positive"] in [0, 1]  # AI 재분석 결과
    # content가 변경되었으므로 updated_at이 갱신됨
    assert "updated_at" in data


def test_update_review_patch_success(client: TestClient):
    """리뷰 부분 업데이트 (PATCH) 성공 테스트"""
    # 영화 및 리뷰 등록
    client.post("/movies/", json={"tmdb_id": 70002, "title": "Test Movie 2"})
    review_data = {
        "tmdb_id": 70002,
        "author": "Original Author",
        "content": "Original content here",
    }
    create_response = client.post("/reviews/", json=review_data)
    assert create_response.status_code == 201
    review_id = create_response.json()["id"]

    # 부분 업데이트 (작성자만 변경)
    update_data = {
        "author": "Partially Updated Author",
    }
    response = client.patch(f"/reviews/{review_id}", json=update_data)
    assert response.status_code == 200
    data = response.json()
    assert data["author"] == "Partially Updated Author"
    assert data["content"] == "Original content here"  # 변경되지 않음


def test_update_review_content_triggers_ai_reanalysis(client: TestClient):
    """리뷰 content 변경 시 AI 재분석 및 영화 AI 평점 업데이트 테스트"""
    # 영화 및 리뷰 등록
    client.post("/movies/", json={"tmdb_id": 70003, "title": "Test Movie 3"})

    # 첫 번째 리뷰
    review1_data = {"tmdb_id": 70003, "author": "User1", "content": "Great movie!"}
    client.post("/reviews/", json=review1_data)

    # 두 번째 리뷰
    review2_data = {"tmdb_id": 70003, "author": "User2", "content": "Not bad"}
    create_response = client.post("/reviews/", json=review2_data)
    review2_id = create_response.json()["id"]

    # 영화의 초기 AI 평점 확인
    movie_rating_before = client.get("/reviews/movie/70003/rating").json()

    # 두 번째 리뷰의 content 변경 (AI 재분석 트리거)
    update_data = {"content": "This is the worst movie I have ever seen!"}
    response = client.patch(f"/reviews/{review2_id}", json=update_data)
    assert response.status_code == 200

    # 영화의 AI 평점이 업데이트되었는지 확인
    movie_rating_after = client.get("/reviews/movie/70003/rating").json()
    # AI 평점이 변경되었을 수 있음 (랜덤 분석이지만 content 변경 시 재계산됨)
    assert "ai_rating" in movie_rating_after


def test_update_review_not_found(client: TestClient):
    """존재하지 않는 리뷰 업데이트 시도 테스트"""
    update_data = {
        "author": "Non-existent Review",
        "content": "This should fail",
    }
    response = client.put("/reviews/99999", json=update_data)
    assert response.status_code == 404
    assert "찾을 수 없습니다" in response.json()["detail"]


def test_update_review_unique_constraint_violation(client: TestClient):
    """리뷰 수정 시 UniqueConstraint 위반 테스트"""
    # 영화 및 첫 번째 리뷰 등록
    client.post("/movies/", json={"tmdb_id": 70004, "title": "Test Movie 4"})
    review1_data = {
        "tmdb_id": 70004,
        "author": "SameAuthor",
        "content": "First review content",
    }
    client.post("/reviews/", json=review1_data)

    # 두 번째 리뷰 등록
    review2_data = {
        "tmdb_id": 70004,
        "author": "SameAuthor",
        "content": "Second review content",
    }
    create_response = client.post("/reviews/", json=review2_data)
    review2_id = create_response.json()["id"]

    # 두 번째 리뷰를 첫 번째 리뷰와 동일하게 수정 시도 (UniqueConstraint 위반)
    update_data = {
        "author": "SameAuthor",
        "content": "First review content",
    }
    response = client.put(f"/reviews/{review2_id}", json=update_data)
    assert response.status_code == 400
    assert "동일한" in response.json()["detail"]
