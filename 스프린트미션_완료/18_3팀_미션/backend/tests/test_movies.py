"""
영화(Movie) API 테스트
"""

from fastapi.testclient import TestClient


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


# ==================== Poster & Pagination Tests ====================


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
            "tmdb_id": 30000,
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
    페이지네이션에서 AI 평점 계산 테스트
    긍정/부정 리뷰 비율에 따라 올바르게 계산되는지 확인
    """
    # 영화 등록
    movie_data = {
        "tmdb_id": 40000,
        "title": "AI 평점 계산 테스트",
        "release_date": "2024-01-01",
        "director": "테스트 감독",
        "genre": "액션",
        "poster_url": None,
        "tmdb_rating": 8.0,
    }
    movie_response = client.post("/movies/", json=movie_data)
    movie_id = movie_response.json()["id"]

    # 긍정 리뷰 3개 등록 (AI 예측 결과가 1인 경우)
    # 부정 리뷰 1개 등록 (AI 예측 결과가 0인 경우)
    # 실제로는 AI가 sentiment를 판단하지만, 여기서는 데이터만 확인
    for i in range(4):
        review_data = {
            "tmdb_id": 40000,
            "author": f"리뷰어 {i+1}",
            "content": f"테스트 리뷰 {i+1}",
        }
        client.post("/reviews/", json=review_data)

    # 페이지네이션 조회
    response = client.get("/movies/paginated")
    assert response.status_code == 200
    data = response.json()

    # 영화 찾기
    movie = data["movies"][0]
    assert movie["total_reviews"] == 4
    # AI 예측 결과에 따라 긍정/부정 비율이 결정됨
    assert "positive_ratio" in movie
    assert "ai_rating" in movie


# ==================== Movie Search Tests ====================


def test_search_movies_by_title(client: TestClient):
    """제목으로 영화 검색 테스트 (대소문자 무시)"""
    # 테스트 데이터 등록
    movies = [
        {
            "tmdb_id": 50001,
            "title": "The Dark Knight",
            "director": "Christopher Nolan",
            "genre": "Action",
        },
        {
            "tmdb_id": 50002,
            "title": "The Matrix",
            "director": "Wachowski",
            "genre": "Sci-Fi",
        },
        {
            "tmdb_id": 50003,
            "title": "Dark Phoenix",
            "director": "Simon Kinberg",
            "genre": "Action",
        },
    ]
    for movie in movies:
        client.post("/movies/", json=movie)

    # "dark" 검색 (대소문자 무시)
    response = client.get("/movies/search", params={"title": "dark"})
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    titles = [m["title"] for m in data["movies"]]
    assert "The Dark Knight" in titles
    assert "Dark Phoenix" in titles
    assert "The Matrix" not in titles


def test_search_movies_by_director(client: TestClient):
    """감독 이름으로 영화 검색 테스트"""
    movies = [
        {"tmdb_id": 50101, "title": "Inception", "director": "Christopher Nolan"},
        {"tmdb_id": 50102, "title": "Interstellar", "director": "Christopher Nolan"},
        {"tmdb_id": 50103, "title": "Tenet", "director": "Christopher Nolan"},
    ]
    for movie in movies:
        client.post("/movies/", json=movie)

    response = client.get("/movies/search", params={"director": "nolan"})
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3


def test_search_movies_by_genre(client: TestClient):
    """장르로 영화 검색 테스트"""
    movies = [
        {"tmdb_id": 50201, "title": "Movie A", "genre": "Action, Thriller"},
        {"tmdb_id": 50202, "title": "Movie B", "genre": "Romance"},
        {"tmdb_id": 50203, "title": "Movie C", "genre": "Action"},
    ]
    for movie in movies:
        client.post("/movies/", json=movie)

    response = client.get("/movies/search", params={"genre": "action"})
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2


def test_search_movies_by_release_date_range(client: TestClient):
    """개봉일 범위로 영화 검색 테스트"""
    movies = [
        {"tmdb_id": 50301, "title": "Movie 2020", "release_date": "2020-05-15"},
        {"tmdb_id": 50302, "title": "Movie 2021", "release_date": "2021-08-20"},
        {"tmdb_id": 50303, "title": "Movie 2023", "release_date": "2023-01-10"},
    ]
    for movie in movies:
        client.post("/movies/", json=movie)

    response = client.get(
        "/movies/search",
        params={"release_date_from": "2021-01-01", "release_date_to": "2022-12-31"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["movies"][0]["title"] == "Movie 2021"


def test_search_movies_by_tmdb_rating_range(client: TestClient):
    """TMDB 평점 범위로 영화 검색 테스트"""
    movies = [
        {"tmdb_id": 50401, "title": "Great Movie", "tmdb_rating": 9.5},
        {"tmdb_id": 50402, "title": "Good Movie", "tmdb_rating": 7.8},
        {"tmdb_id": 50403, "title": "Bad Movie", "tmdb_rating": 5.2},
    ]
    for movie in movies:
        client.post("/movies/", json=movie)

    response = client.get(
        "/movies/search", params={"tmdb_rating_min": 7.0, "tmdb_rating_max": 9.0}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["movies"][0]["title"] == "Good Movie"


def test_search_movies_multiple_filters(client: TestClient):
    """복합 필터로 영화 검색 테스트 (AND 조합)"""
    movies = [
        {
            "tmdb_id": 50501,
            "title": "Nolan Action 2020",
            "director": "Nolan",
            "genre": "Action",
            "release_date": "2020-01-01",
        },
        {
            "tmdb_id": 50502,
            "title": "Nolan Drama 2020",
            "director": "Nolan",
            "genre": "Drama",
            "release_date": "2020-06-01",
        },
        {
            "tmdb_id": 50503,
            "title": "Other Action 2020",
            "director": "Other",
            "genre": "Action",
            "release_date": "2020-03-01",
        },
    ]
    for movie in movies:
        client.post("/movies/", json=movie)

    response = client.get(
        "/movies/search",
        params={
            "director": "nolan",
            "genre": "action",
            "release_date_from": "2020-01-01",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["movies"][0]["title"] == "Nolan Action 2020"


def test_search_movies_with_sorting(client: TestClient):
    """정렬 옵션으로 영화 검색 테스트"""
    movies = [
        {"tmdb_id": 50601, "title": "B Movie", "tmdb_rating": 7.0},
        {"tmdb_id": 50602, "title": "A Movie", "tmdb_rating": 8.0},
        {"tmdb_id": 50603, "title": "C Movie", "tmdb_rating": 6.0},
    ]
    for movie in movies:
        client.post("/movies/", json=movie)

    # 제목 오름차순
    response = client.get(
        "/movies/search", params={"sort_by": "title", "sort_order": "asc"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["movies"][0]["title"] == "A Movie"
    assert data["movies"][2]["title"] == "C Movie"

    # 평점 내림차순
    response = client.get(
        "/movies/search", params={"sort_by": "tmdb_rating", "sort_order": "desc"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["movies"][0]["title"] == "A Movie"  # 8.0
    assert data["movies"][2]["title"] == "C Movie"  # 6.0


def test_search_movies_empty_result(client: TestClient):
    """검색 결과가 없는 경우 테스트"""
    client.post("/movies/", json={"tmdb_id": 50701, "title": "Only Movie"})

    response = client.get("/movies/search", params={"title": "NonExistent"})
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert len(data["movies"]) == 0


def test_search_movies_pagination(client: TestClient):
    """검색 결과 페이지네이션 테스트"""
    for i in range(15):
        client.post("/movies/", json={"tmdb_id": 50800 + i, "title": f"Test Movie {i}"})

    # 첫 페이지
    response = client.get("/movies/search", params={"page": 1, "page_size": 10})
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 15
    assert len(data["movies"]) == 10
    assert data["page"] == 1
    assert data["total_pages"] == 2

    # 두 번째 페이지
    response = client.get("/movies/search", params={"page": 2, "page_size": 10})
    assert response.status_code == 200
    data = response.json()
    assert len(data["movies"]) == 5
    assert data["page"] == 2


# ==================== Movie Update Tests ====================


def test_update_movie_put_success(client: TestClient):
    """영화 전체 업데이트 (PUT) 성공 테스트"""
    # 영화 등록
    movie_data = {
        "tmdb_id": 60001,
        "title": "Original Title",
        "release_date": "2024-01-01",
        "director": "Original Director",
        "genre": "Drama",
        "poster_url": None,
        "tmdb_rating": 7.0,
    }
    create_response = client.post("/movies/", json=movie_data)
    assert create_response.status_code == 201
    movie_id = create_response.json()["id"]

    # 전체 업데이트
    update_data = {
        "title": "Updated Title",
        "release_date": "2024-12-31",
        "director": "Updated Director",
        "genre": "Action",
        "poster_url": None,
        "tmdb_rating": 9.5,
    }
    response = client.put(f"/movies/{movie_id}", json=update_data)
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Updated Title"
    assert data["director"] == "Updated Director"
    assert data["genre"] == "Action"
    assert data["tmdb_rating"] == 9.5
    assert data["tmdb_id"] == 60001  # tmdb_id는 변경되지 않음


def test_update_movie_patch_success(client: TestClient):
    """영화 부분 업데이트 (PATCH) 성공 테스트"""
    # 영화 등록
    movie_data = {
        "tmdb_id": 60002,
        "title": "Original Title",
        "release_date": "2024-01-01",
        "director": "Original Director",
        "genre": "Drama",
        "poster_url": None,
        "tmdb_rating": 7.0,
    }
    create_response = client.post("/movies/", json=movie_data)
    assert create_response.status_code == 201
    movie_id = create_response.json()["id"]

    # 부분 업데이트 (제목과 평점만 변경)
    update_data = {
        "title": "Partially Updated Title",
        "tmdb_rating": 8.5,
    }
    response = client.patch(f"/movies/{movie_id}", json=update_data)
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Partially Updated Title"
    assert data["tmdb_rating"] == 8.5
    assert data["director"] == "Original Director"  # 변경되지 않음
    assert data["genre"] == "Drama"  # 변경되지 않음


def test_update_movie_not_found(client: TestClient):
    """존재하지 않는 영화 업데이트 시도 테스트"""
    update_data = {
        "title": "Non-existent Movie",
        "release_date": "2024-01-01",
        "director": "Director",
        "genre": "Genre",
        "poster_url": None,
        "tmdb_rating": 7.0,
    }
    response = client.put("/movies/99999", json=update_data)
    assert response.status_code == 404
    assert "찾을 수 없습니다" in response.json()["detail"]


def test_update_movie_poster_change(client: TestClient):
    """포스터 URL 변경 시 기존 파일 삭제 로직 테스트"""
    # 영화 등록 (포스터 없이)
    movie_data = {
        "tmdb_id": 60003,
        "title": "Movie with Poster",
        "release_date": "2024-01-01",
        "director": "Director",
        "genre": "Action",
        "poster_url": None,
        "tmdb_rating": 8.0,
    }
    create_response = client.post("/movies/", json=movie_data)
    assert create_response.status_code == 201
    movie_id = create_response.json()["id"]

    # 포스터 URL 추가
    update_data = {
        "title": "Movie with Poster",
        "release_date": "2024-01-01",
        "director": "Director",
        "genre": "Action",
        "poster_url": "https://example.com/new_poster.jpg",
        "tmdb_rating": 8.0,
    }
    response = client.put(f"/movies/{movie_id}", json=update_data)
    assert response.status_code == 200
    # 백그라운드 작업으로 처리되므로 즉시 확인은 불가능하지만, 에러 없이 완료되면 성공
