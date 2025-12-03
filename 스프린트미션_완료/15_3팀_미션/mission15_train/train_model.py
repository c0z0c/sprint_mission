import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# 기본 스트림 핸들러 설정 — 터미널에 INFO 이상 로그 출력
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 경고 무시 설정 (옵션)
import warnings
warnings.filterwarnings('ignore')

# 0. 환경 감지
# Dockerfile의 ENV 설정 덕분에 컨테이너 실행 시 True로 설정됨
IS_DOCKER = os.environ.get('RUNNING_IN_DOCKER', 'False') == 'True'

# 1. 경로 설정
# Dockerfile 또는 docker-compose에서 데이터가 COPY되거나 마운트될 경로
DATA_DIR = r'..\data' if not IS_DOCKER else '/app/data'
TRAIN_FILE = os.path.join(DATA_DIR, 'mission15_train.csv')


# 모델 파일이 저장되어 연구자 2에게 공유될 경로 (Named Volume 마운트 지점)
# model.pkl은 이 경로에 저장됩니다.
MODEL_OUTPUT_DIR = r'..\data' if not IS_DOCKER else '/app/data'
MODEL_FILENAME = os.path.join(MODEL_OUTPUT_DIR, 'model.pkl')

logger.info("모델 학습 시작")
logger.info(f"데이터 파일 경로: {TRAIN_FILE}")
logger.info(f"모델 저장 경로: {MODEL_FILENAME}")
logger.info("-" * 30)

# 2. 데이터 로드 및 분리
if not os.path.exists(TRAIN_FILE):
    logger.error(f"오류: 훈련 데이터 파일 '{TRAIN_FILE}'을 찾을 수 없습니다.")
    exit(1)

df = pd.read_csv(TRAIN_FILE)

# 특성(X)과 타겟(y) 분리
X = df.drop('Performance Index', axis=1)
y = df['Performance Index']

# 학습 및 검증 데이터 분리 (모델 평가용)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
logger.info(f"훈련 데이터 크기: {X_train.shape}")


# 3. 파이프라인 구축 (전처리 + 모델링)
# 범주형 변수 처리 설정
# Yes/No 형태의 'Extracurricular Activities' 변수를 OneHotEncoding
categorical_features = ['Extracurricular Activities']
preprocessor = ColumnTransformer(
    transformers=[
        # OneHotEncoder를 사용하며, 첫 번째 카테고리를 제거(drop='first')하여 다중공선성 방지
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
    ],
    remainder='passthrough'
)

# 모델 정의 (선형 회귀)
model = LinearRegression()

# 전처리 + 모델링 파이프라인
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', model)
])

logger.info("파이프라인 구축 완료.")


# 4. 모델 학습
pipeline.fit(X_train, y_train)
logger.info("모델 학습 완료.")


# 5. 모델 성능 평가 (RMSE)
y_pred = pipeline.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
logger.info(f"검증 세트 RMSE: {rmse:.4f}")

# 6. 모델 저장 (model.pkl)
# 모델 저장 경로 폴더가 없으면 생성
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# 최종 파이프라인 모델을 joblib을 사용하여 저장
tmp_path = MODEL_FILENAME + ".tmp"
joblib.dump(pipeline, tmp_path)
os.replace(tmp_path, MODEL_FILENAME)
logger.info(f"모델이 '{MODEL_FILENAME}'에 성공적으로 저장되었습니다.")
