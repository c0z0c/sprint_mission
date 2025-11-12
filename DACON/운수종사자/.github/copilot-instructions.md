# 1. 원칙 요약
- 개인 개발 환경(MVP) 최적화, 재현성(reproducibility) 우선.
- 헬퍼 모듈 우선 재사용, 문서-코드 동등성(즉시 게시 가능) 유지.
- 이모지 금지, 불필요한 장황함 금지.
- 코드는 항상 새로 읽어서 사용.
- 코드 위치 제안시 셀기준 코드 위치
- 장문 코드를 제안하기 전에는 항상 물어 볼 것.

2. 프로젝트 개요
[배경]
교통사고는 차량·도로 환경뿐 아니라 운수종사자의 인지 특성에 크게 좌우됩니다. 
실제로 운수종사자는 신규 진입 시 자격 검사를 받고, 이후 정기적으로 자격 유지 검사를 통해 인지 능력과 안전 운전 역량을 점검받습니다.
이러한 자격 검사 데이터를 활용해 사고 위험도를 예측하는 AI 모델을 개발함으로써, 교통사고 예방과 맞춤형 안전 관리 체계 구축에 기여할 수 있습니다.

[대회 방식]
본 대회는 1차 평가, 2차 평가순으로 진행됩니다.
🔹1차 평가: 최종 Public 리더보드 기준 상위 15팀을 2차 평가 진출팀으로 선정합니다.
🔹2차 평가: 진출팀은 '모델 개발 보고서'와 '데이터 분석 보고서'를 작성하여 제출해야 하며, 이를 종합적으로 평가하여 최종 상위 7팀을 수상팀으로 선정합니다.


[주제]
운수종사자 인지적 특성 데이터를 활용한 교통사고 위험 예측 AI 모델 개발

[설명]
운수종사자 자격검사(A: 신규자격, B: 자격유지) 과정에서 수집된 인지·반응 관련 세부 검사 데이터를 활용하여, 검사 결과 기준 교통사고 위험군에 속할 확률을 예측하는 AI 모델을 개발합니다.
참가자는 각 운수종사자의 인지적 특성을 종합적으로 분석하여 교통사고 발생 가능성을 정량적으로 추정할 수 있는 예측 모델을 구축해야 합니다.
대회 종료 후 모델 개발 결과는 1)모델 개발 보고서와 2)데이터 분석 보고서의 형태로 제출되며, 모델의 성능뿐 아니라 데이터 이해도와 분석 과정의 논리성 또한 함께 평가됩니다.

# 3. 기본 스택(우선 import)
- DL: torch, torchvision, torchaudio, transformers, datasets, accelerate, peft, sentencepiece
- Data: numpy, pandas, scikit-learn, scipy
- Viz: matplotlib, seaborn, plotly(선택)
- Exp: wandb, tqdm, pytz, rich(선택)
- Dev: jupyter, ipykernel, python-dotenv, pathlib(표준), json/pickle/yaml(선택)
- Lint/Type: black, isort, ruff/flake8, mypy(선택)

# 4. 헬퍼 모듈 사용 규칙
- 중복 구현 금지: 동일 기능은 헬퍼로 해결.
- helper_utils.py 파일을 읽어서 선언된 함수를 사용하여 기능을 구현.
- helper_c0z0c_dev.py 파일을 읽어서 선언된 함수를 사용하여 기능을 구현.
- 경로/저장 로직: get_path_modeling(), save_model_dict() 패턴 준수.
- Colab/로컬 분기: drive_root() 사용.
```python
# 헬퍼 로드(필수 스니펫)
from urllib.request import urlretrieve; urlretrieve("https://raw.githubusercontent.com/c0z0c/jupyter_hangul/refs/heads/beta/helper_utils.py", "helper_utils.py")
import importlib, helper_utils as hu
importlib.reload(hu); from helper_utils import *
```
# 5. Debugging
- WandB 충돌 → `reinit=True` + 고유 `name`
- 한글 폰트 → `helper.setup()`

# 6. Documentation (GitHub Pages)
- 목차: `1.` `1.1.` `1.1.1`
- Mermaid: 노드 라벨 큰따옴표 `A["노드"]`
- 수식: `$$` 블록 우선
- 용어: 한영 병기 (normalization, 노멀라이제이션)
- 이모지 금지

# 7. Checklist
- [ ] `from helper_utils import *` 로드
- [ ] 전역 변수 설정 (`__device`, `__kst`)
- [ ] 시드 고정 (`torch.manual_seed(42)`)
- [ ] WandB init (entity, project, name)
- [ ] 경로는 `get_path_modeling()` 사용
