"""인덱스 생성 스크립트"""

from app.database.DatabaseConnector import DatabaseConnector
from sqlalchemy import text

db = DatabaseConnector()
with db.engine.connect() as conn:
    # 인덱스 존재 확인
    result = conn.execute(
        text(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name='idx_reviews_tmdb_id'"
        )
    )
    row = result.fetchone()

    if row:
        print("✅ 인덱스가 이미 존재합니다:")
        print(row[0])
    else:
        print("📝 인덱스를 생성합니다...")
        conn.execute(text("CREATE INDEX idx_reviews_tmdb_id ON reviews(tmdb_id)"))
        conn.commit()
        print("✅ 인덱스 생성 완료: idx_reviews_tmdb_id")

        # 확인
        result = conn.execute(
            text(
                "SELECT sql FROM sqlite_master WHERE type='index' AND name='idx_reviews_tmdb_id'"
            )
        )
        row = result.fetchone()
        if row:
            print(f"   {row[0]}")
