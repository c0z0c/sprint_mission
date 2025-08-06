
## 🗂️ Adult Dataset 필드 설명

| 필드명 (영문)       | 한글 발음        | 설명 |
|--------------------|------------------|------|
| `age`              | 에이지           | 개인의 나이 |
| `workclass`        | 워크클래스       | 고용 형태 (예: 사기업, 자영업, 공공기관 등) |
| `fnlwgt`           | 에프엔엘더블유지티 | 인구조사 샘플이 대표하는 인구 수 |
| `education`        | 에듀케이션       | 교육 수준 (예: 대학 학위, 고등학교 졸업 등) |
| `education-num`    | 에듀케이션넘     | 교육 수준을 숫자로 표현한 값 |
| `marital-status`   | 매리털스태터스    | 결혼 상태 (예: 결혼, 이혼, 미혼 등) |
| `occupation`       | 오큐페이션       | 직업군 (예: 기술자, 관리자, 판매직 등) |
| `relationship`     | 릴레이션십       | 가족 내 관계 (예: 배우자, 자녀, 부모 등) |
| `race`             | 레이스           | 인종 구분 (예: 백인, 아시아인 등) |
| `sex`              | 섹스             | 성별 (`Male` 또는 `Female`) |
| `capital-gain`     | 캐피털게인       | 자본 이득 (예: 주식 또는 부동산 수익) |
| `capital-loss`     | 캐피털로스       | 자본 손실 (예: 자산 손해) |
| `hours-per-week`   | 아워스퍼위크     | 주당 근무 시간 |
| `native-country`   | 네이티브컨트리   | 출신 국가 (예: 미국, 멕시코, 필리핀 등) |
| `income`           | 인컴             | 연간 수입 (`<=50K`, `>50K`) 분류 타겟 |

---

## 📘 Adult Dataset 필드별 값 설명

| 필드명 | 주요 값 예시 | 설명 |
|--------|--------------|------|
| `age` | 17 ~ 90 | 개인의 나이 (연속형 변수) |
| `workclass` | `Private`, `Self-emp-not-inc`, `Federal-gov`, `Without-pay` 등 | 고용 형태: 사기업, 자영업, 정부기관, 무급 등 |
| `fnlwgt` | 예: 77516, 83311 | 인구조사 샘플이 대표하는 인구 수 (가중치) |
| `education` | `Bachelors`, `HS-grad`, `Some-college`, `Doctorate` 등 | 교육 수준: 학사, 고졸, 일부 대학, 박사 등 |
| `education-num` | 1 ~ 16 | 교육 수준을 숫자로 표현 (예: `Preschool`=1, `Doctorate`=16) |
| `marital-status` | `Never-married`, `Married-civ-spouse`, `Divorced` 등 | 결혼 상태: 미혼, 결혼, 이혼 등 |
| `occupation` | `Tech-support`, `Sales`, `Exec-managerial`, `Armed-Forces` 등 | 직업군: 기술 지원, 영업, 관리자, 군인 등 |
| `relationship` | `Husband`, `Not-in-family`, `Own-child` 등 | 가족 내 관계: 배우자, 가족 외, 자녀 등 |
| `race` | `White`, `Black`, `Asian-Pac-Islander`, `Other` 등 | 인종 구분 |
| `sex` | `Male`, `Female` | 성별 |
| `capital-gain` | 예: 0, 14084, 99999 | 자본 이득 (연속형 변수, 대부분 0) |
| `capital-loss` | 예: 0, 1902, 1977 | 자본 손실 (연속형 변수, 대부분 0) |
| `hours-per-week` | 예: 1 ~ 99 | 주당 근무 시간 |
| `native-country` | `United-States`, `Mexico`, `Philippines`, `Germany` 등 | 출신 국가 |
| `income` | `<=50K`, `>50K` | 연간 수입이 $50,000 이하 또는 초과 (타겟 변수) |

---

### 🧩 참고 사항

- `workclass`, `occupation`, `native-country`에는 **결측값**이 존재할 수 있으며, `"?"`로 표시됩니다.
- `education`과 `education-num`은 서로 연관된 변수로, 하나만 사용해도 무방합니다.
- `capital-gain`과 `capital-loss`는 대부분 0이며, 일부 고소득자에게만 값이 존재합니다.
- `fnlwgt`는 분석 목적에 따라 사용 여부가 달라질 수 있습니다. 일반적으로는 제거하거나 가중치로 활용합니다.

---
