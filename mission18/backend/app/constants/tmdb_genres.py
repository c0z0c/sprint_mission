"""
TMDB 장르 매핑 상수
"""

from typing import List, Optional

# TMDB Genre ID → 한글 이름 매핑
# 출처: https://developers.themoviedb.org/3/genres/get-movie-list
TMDB_GENRE_MAP = {
    28: "액션",
    12: "모험",
    16: "애니메이션",
    35: "코미디",
    80: "범죄",
    99: "다큐멘터리",
    18: "드라마",
    10751: "가족",
    14: "판타지",
    36: "역사",
    27: "공포",
    10402: "음악",
    9648: "미스터리",
    10749: "로맨스",
    878: "SF",
    10770: "TV 영화",
    53: "스릴러",
    10752: "전쟁",
    37: "서부",
}


def convert_genre_ids_to_korean(genre_ids: Optional[List[int]]) -> Optional[str]:
    """
    TMDB 장르 ID 배열을 한글 텍스트로 변환

    Args:
        genre_ids: TMDB 장르 ID 리스트 (예: [28, 12, 878])

    Returns:
        Optional[str]: 한글 장르 문자열 (예: "액션, 모험, SF") 또는 None

    Examples:
        >>> convert_genre_ids_to_korean([28, 12, 878])
        '액션, 모험, SF'
        >>> convert_genre_ids_to_korean([])
        None
        >>> convert_genre_ids_to_korean(None)
        None
    """
    if not genre_ids:
        return None

    # 유효한 장르만 변환
    genre_names = []
    for genre_id in genre_ids:
        if genre_name := TMDB_GENRE_MAP.get(genre_id):
            genre_names.append(genre_name)

    return ", ".join(genre_names) if genre_names else None
