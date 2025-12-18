"""
날짜 필드 (created_at, updated_at) 동작 테스트
- 생성 시 입력값 우선, 없으면 자동
- 수정 시 입력값 우선, 없으면 자동
"""

import pytest
from datetime import datetime, timedelta
from app.schemas.review import ReviewCreate, ReviewUpdate, ReviewPatch


class TestDateTimeFields:
    """날짜 필드 동작 테스트"""

    @pytest.fixture
    def sample_movie_data(self):
        """테스트용 영화 데이터"""
        return {
            "tmdb_id": 99999,
            "title": "날짜 테스트 영화",
            "release_date": "2025-12-18",
        }

    def test_create_review_without_datetime(self, client, sample_movie_data):
        """리뷰 생성 - 날짜 필드 미제공 시 자동 생성"""
        # 1. 영화 등록
        movie_response = client.post("/movies/", json=sample_movie_data)
        assert movie_response.status_code == 201

        # 2. 리뷰 등록 (날짜 필드 미제공)
        review_data = {
            "tmdb_id": sample_movie_data["tmdb_id"],
            "author": "자동날짜테스터",
            "content": "날짜가 자동으로 생성되어야 합니다.",
        }

        before_time = datetime.now()
        response = client.post("/reviews/", json=review_data)
        after_time = datetime.now()

        assert response.status_code == 201
        result = response.json()

        # 3. 날짜가 자동 생성되었는지 확인
        created_at = datetime.fromisoformat(result["created_at"].replace("Z", "+00:00"))
        updated_at = datetime.fromisoformat(result["updated_at"].replace("Z", "+00:00"))

        # 현재 시간 범위 내에 있어야 함
        assert before_time <= created_at.replace(tzinfo=None) <= after_time
        assert before_time <= updated_at.replace(tzinfo=None) <= after_time
        assert created_at == updated_at  # 생성 시 둘이 같아야 함

    def test_create_review_with_custom_datetime(self, client, sample_movie_data):
        """리뷰 생성 - 날짜 필드 제공 시 입력값 사용"""
        # 1. 영화 등록
        movie_response = client.post("/movies/", json=sample_movie_data)
        assert movie_response.status_code == 201

        # 2. 리뷰 등록 (커스텀 날짜 제공)
        custom_created_at = datetime(2024, 1, 1, 12, 0, 0)
        custom_updated_at = datetime(2024, 1, 2, 13, 0, 0)

        review_data = {
            "tmdb_id": sample_movie_data["tmdb_id"],
            "author": "커스텀날짜테스터",
            "content": "날짜를 직접 지정합니다.",
            "created_at": custom_created_at.isoformat(),
            "updated_at": custom_updated_at.isoformat(),
        }

        response = client.post("/reviews/", json=review_data)
        assert response.status_code == 201
        result = response.json()

        # 3. 제공한 날짜가 사용되었는지 확인
        created_at = datetime.fromisoformat(result["created_at"].replace("Z", "+00:00"))
        updated_at = datetime.fromisoformat(result["updated_at"].replace("Z", "+00:00"))

        assert created_at.replace(tzinfo=None) == custom_created_at
        assert updated_at.replace(tzinfo=None) == custom_updated_at

    def test_update_review_without_datetime(self, client, sample_movie_data):
        """리뷰 수정 (PUT) - updated_at 미제공 시 자동 갱신"""
        # 1. 영화 등록
        movie_response = client.post("/movies/", json=sample_movie_data)
        assert movie_response.status_code == 201

        # 2. 리뷰 등록
        review_data = {
            "tmdb_id": sample_movie_data["tmdb_id"],
            "author": "수정테스터",
            "content": "원본 내용입니다.",
        }
        create_response = client.post("/reviews/", json=review_data)
        assert create_response.status_code == 201
        review_id = create_response.json()["id"]
        original_updated_at = datetime.fromisoformat(
            create_response.json()["updated_at"].replace("Z", "+00:00")
        )

        # 3. 리뷰 수정 (updated_at 미제공)
        import time

        time.sleep(0.1)  # 시간 차이를 만들기 위해 대기

        update_data = {
            "author": "수정테스터",
            "content": "수정된 내용입니다.",
        }

        before_time = datetime.now()
        update_response = client.put(f"/reviews/{review_id}", json=update_data)
        after_time = datetime.now()

        assert update_response.status_code == 200
        result = update_response.json()

        # 4. updated_at이 자동 갱신되었는지 확인
        new_updated_at = datetime.fromisoformat(
            result["updated_at"].replace("Z", "+00:00")
        )

        # 원본보다 나중 시간이어야 함
        assert new_updated_at > original_updated_at
        # 현재 시간 범위 내에 있어야 함
        assert before_time <= new_updated_at.replace(tzinfo=None) <= after_time

    def test_update_review_with_custom_datetime(self, client, sample_movie_data):
        """리뷰 수정 (PUT) - updated_at 제공 시 입력값 사용"""
        # 1. 영화 등록
        movie_response = client.post("/movies/", json=sample_movie_data)
        assert movie_response.status_code == 201

        # 2. 리뷰 등록
        review_data = {
            "tmdb_id": sample_movie_data["tmdb_id"],
            "author": "커스텀수정테스터",
            "content": "원본 내용입니다.",
        }
        create_response = client.post("/reviews/", json=review_data)
        assert create_response.status_code == 201
        review_id = create_response.json()["id"]

        # 3. 리뷰 수정 (커스텀 updated_at 제공)
        custom_updated_at = datetime(2025, 6, 15, 14, 30, 0)

        update_data = {
            "author": "커스텀수정테스터",
            "content": "수정된 내용입니다.",
            "updated_at": custom_updated_at.isoformat(),
        }

        update_response = client.put(f"/reviews/{review_id}", json=update_data)
        assert update_response.status_code == 200
        result = update_response.json()

        # 4. 제공한 updated_at이 사용되었는지 확인
        new_updated_at = datetime.fromisoformat(
            result["updated_at"].replace("Z", "+00:00")
        )

        assert new_updated_at.replace(tzinfo=None) == custom_updated_at

    def test_patch_review_without_datetime(self, client, sample_movie_data):
        """리뷰 수정 (PATCH) - updated_at 미제공 시 자동 갱신"""
        # 1. 영화 등록
        movie_response = client.post("/movies/", json=sample_movie_data)
        assert movie_response.status_code == 201

        # 2. 리뷰 등록
        review_data = {
            "tmdb_id": sample_movie_data["tmdb_id"],
            "author": "패치테스터",
            "content": "원본 내용입니다.",
        }
        create_response = client.post("/reviews/", json=review_data)
        assert create_response.status_code == 201
        review_id = create_response.json()["id"]
        original_updated_at = datetime.fromisoformat(
            create_response.json()["updated_at"].replace("Z", "+00:00")
        )

        # 3. 리뷰 부분 수정 (content만 수정, updated_at 미제공)
        import time

        time.sleep(0.1)

        patch_data = {"content": "PATCH로 수정된 내용입니다."}

        before_time = datetime.now()
        patch_response = client.patch(f"/reviews/{review_id}", json=patch_data)
        after_time = datetime.now()

        assert patch_response.status_code == 200
        result = patch_response.json()

        # 4. updated_at이 자동 갱신되었는지 확인
        new_updated_at = datetime.fromisoformat(
            result["updated_at"].replace("Z", "+00:00")
        )

        assert new_updated_at > original_updated_at
        assert before_time <= new_updated_at.replace(tzinfo=None) <= after_time

    def test_patch_review_with_custom_datetime(self, client, sample_movie_data):
        """리뷰 수정 (PATCH) - updated_at 제공 시 입력값 사용"""
        # 1. 영화 등록
        movie_response = client.post("/movies/", json=sample_movie_data)
        assert movie_response.status_code == 201

        # 2. 리뷰 등록
        review_data = {
            "tmdb_id": sample_movie_data["tmdb_id"],
            "author": "커스텀패치테스터",
            "content": "원본 내용입니다.",
        }
        create_response = client.post("/reviews/", json=review_data)
        assert create_response.status_code == 201
        review_id = create_response.json()["id"]

        # 3. 리뷰 부분 수정 (커스텀 updated_at 제공)
        custom_updated_at = datetime(2025, 7, 20, 16, 45, 0)

        patch_data = {
            "content": "PATCH로 수정된 내용입니다.",
            "updated_at": custom_updated_at.isoformat(),
        }

        patch_response = client.patch(f"/reviews/{review_id}", json=patch_data)
        assert patch_response.status_code == 200
        result = patch_response.json()

        # 4. 제공한 updated_at이 사용되었는지 확인
        new_updated_at = datetime.fromisoformat(
            result["updated_at"].replace("Z", "+00:00")
        )

        assert new_updated_at.replace(tzinfo=None) == custom_updated_at
