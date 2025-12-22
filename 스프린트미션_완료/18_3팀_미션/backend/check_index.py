"""인덱스 확인 스크립트"""

from app.database.DatabaseConnector import DatabaseConnector
from sqlalchemy import text

db = DatabaseConnector()
with db.engine.connect() as conn:
    result = conn.execute(
        text(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name='idx_reviews_tmdb_id'"
        )
    )
    row = result.fetchone()
    if row:
        print("✅ 인덱스가 존재합니다:")
        print(row[0])
    else:
        print("❌ 인덱스가 존재하지 않습니다. 수동 생성이 필요합니다.")
        print("\n다음 명령으로 생성:")
        print("CREATE INDEX idx_reviews_tmdb_id ON reviews(tmdb_id);")
