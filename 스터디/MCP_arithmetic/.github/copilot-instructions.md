# 1. 원칙 요약
- 개인 개발 환경(MVP) 최적화, 재현성(reproducibility) 우선.
- 헬퍼 모듈 우선 재사용, 문서-코드 동등성(즉시 게시 가능) 유지.
- 이모지 금지, 불필요한 장황함 금지.

# 2. 상호작용 프로토콜(설명 후 확인 → 코딩)
1) 요구 요약: 목표/제약/산출물.  
2) 접근 전략 설명: 선택지·트레이드오프, 변경 파일, 성능/재현성 영향.  
3) 확인 질문: “이 전략으로 코딩 진행할까요? (예/아니오/수정)”  
4) 승인 후 코딩: 최소 실행 예제, 경로/의존성/헬퍼 호출, 간단 검증 포함.  

# 3. Coding Standards
- PEP 8 + Black + isort
- `pathlib` 우선, f-string, 상수 UPPER_SNAKE_CASE
- 타입 힌트 + Docstring (Google/NumPy)
- 로깅 우선 (print 최소화)

# 4. Documentation (GitHub Pages)
- 목차: `1.` `1.1.` `1.1.1`
- Mermaid: 노드 라벨 큰따옴표 `A["노드"]`
- 수식: `$$` 블록 우선
- 용어: 한영 병기 (normalization, 노멀라이제이션)
- 이모지 금지
