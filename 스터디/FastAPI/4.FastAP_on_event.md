---
layout: default
title: "FastAPI - FastAPI `on_event` 가이드 문서"
description: "FastAPI - FastAPI `on_event` 가이드 문서"
date: 2025-12-15
cache-control: no-cache
expires: 0
pragma: no-cache
author: "김명환"
---

## 📄 1. FastAPI `on_event` 가이드 문서

FastAPI의 `on_event` 데코레이터는 애플리케이션의 수명 주기(lifecycle) 중 특정 시점에 실행되어야 하는 함수를 등록하는 데 사용됩니다. 주로 서버가 요청을 처리하기 전에 필요한 초기화 작업이나 서버 종료 시 필요한 자원 정리 작업을 수행하는 데 활용됩니다.

### 1.1. `on_event`의 종류 및 실행 시점

| 이벤트 종류 | 실행 시점 | 용도 |
| :--- | :--- | :--- |
| **`"startup"`** | 애플리케이션이 시작될 때, 첫 번째 요청을 처리하기 **직전** | 초기 설정, 자원 로드, 연결 수립 |
| **`"shutdown"`** | 애플리케이션이 종료될 때, 모든 요청 처리가 **완료된 후** | 자원 해제, 연결 종료, 정리 작업 |

### 1.2. `on_event("startup")`

#### 1.2.1. 용도 (Use Cases)

FastAPI 애플리케이션이 요청을 처리할 준비를 할 때 필요한 모든 준비 작업을 수행합니다.

  * **모델 로드 (Model Loading)**: ONNX, PyTorch, TensorFlow 등 머신러닝 모델을 디스크에서 읽어 메모리에 로드하여 요청 시 바로 사용할 수 있도록 준비합니다. (제공된 코드의 예시)
  * **데이터베이스 연결 (Database Connection)**: 데이터베이스 풀(connection pool)을 초기화하고 연결을 설정합니다.
  * **캐시 초기화 (Cache Initialization)**: Redis나 Memcached와 같은 캐시 시스템과의 연결을 설정합니다.
  * **설정 파일 로드 (Configuration Loading)**: 애플리케이션 전반에 걸쳐 사용될 설정(Configuration)을 로드합니다.

#### 1.2.2. 사용법 (Usage)

```python
from fastapi import FastAPI
import onnxruntime

app = FastAPI()

# onnxruntime 세션은 애플리케이션 수명 동안 유지됨
ort_session = None 

@app.on_event("startup")
def load_resources():
    global ort_session
    # 서버 시작 시 모델을 로드
    ort_session = onnxruntime.InferenceSession("model.onnx")
    print("INFO: Model loaded successfully.")

# 이후 엔드포인트에서 ort_session을 사용
@app.post("/predict")
def predict():
    # ... 예측 로직 ...
    return {"status": "ok"}
```

### 1.3. `on_event("shutdown")`

#### 1.3.1. 용도 (Use Cases)

애플리케이션이 정상적으로 종료될 때 사용했던 자원을 깨끗하게 해제하여 메모리 누수(memory leak)나 불완전한 상태를 방지합니다.

  * **데이터베이스 연결 해제 (Database Connection Closing)**: 활성화된 데이터베이스 연결 풀을 안전하게 종료합니다.
  * **로그 및 통계 저장 (Log and Statistics Saving)**: 종료 직전까지의 로그나 통계 데이터를 저장소에 기록합니다.
  * **임시 파일 정리 (Temporary File Cleanup)**: 실행 중에 생성된 임시 파일이나 캐시를 삭제합니다.

#### 1.3.2. 사용법 (Usage)

```python
from fastapi import FastAPI

app = FastAPI()

# 예시: 가상의 DB 연결 객체
db_connection = None

@app.on_event("shutdown")
def close_resources():
    # 서버 종료 시 DB 연결을 해제
    if db_connection:
        db_connection.close()
        print("INFO: Database connection closed.")
```

### 1.4. 사용 시 주의점 (Cautions)

#### 1.4.1. 비동기 처리 (Asynchronous Handling)

`on_event`에 등록되는 함수는 \*\*일반 함수(`def`)\*\*와 **비동기 함수(`async def`)** 모두 가능합니다.

  * **일반 함수**: 동기적으로 실행되며, 해당 함수가 완료될 때까지 서버 시작/종료 프로세스가 **대기**합니다. 파일 시스템 I/O (예: 모델 로드)와 같이 블로킹(blocking) 작업이 포함될 때 사용될 수 있습니다.
  * **비동기 함수**: 비동기적으로 실행되며, 주로 `await`을 사용하여 비동기 I/O 작업(예: `asyncpg`를 사용한 DB 연결)을 처리할 때 사용됩니다.

#### 1.4.2. 실행 순서 및 에러 처리 (Execution Order and Error Handling)

  * **`startup`**: 등록된 모든 `startup` 함수가 **순차적으로** 실행됩니다. 만약 하나의 `startup` 함수에서 예외(Exception)가 발생하면, 나머지 `startup` 함수는 실행되지 않고 서버가 **시작되지 않은 채** 종료됩니다.
  * **`shutdown`**: 등록된 모든 `shutdown` 함수도 **순차적으로** 실행됩니다. `shutdown` 함수 내에서 예외가 발생하더라도, 나머지 `shutdown` 함수는 **계속 실행**됩니다. 이는 자원 정리 작업의 중요성 때문에 설계된 방식입니다.

#### 1.4.3. 대규모 프로젝트 대체 (Alternatives in Large Projects)

FastAPI는 Starlette 기반이므로 `on_event`를 사용합니다. 그러나 프로젝트 규모가 커지거나 의존성 주입(Dependency Injection)과 더 잘 통합되기를 원한다면, **`lifespan` 컨텍스트 관리자**를 사용하는 것이 권장됩니다.

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

# @asynccontextmanager를 사용하여 startup/shutdown 로직 통합
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. 'startup' 로직
    print("INFO: Server starting up...")
    yield # 이 시점에 서버가 요청 처리를 시작함
    # 2. 'shutdown' 로직
    print("INFO: Server shutting down...")

app = FastAPI(lifespan=lifespan)
```

이는 최신 ASGI(Asynchronous Server Gateway Interface) 표준에 더 부합하며, 비동기 자원 관리에 더 유연합니다.

### 1.5. Mermaid 다이어그램: 서버 수명 주기

```mermaid
graph TD
    A[서버 시작 명령어 실행] --> B{FastAPI 초기화};
    B --> C{on_event("startup") 함수 실행};
    C --> D{모든 startup 함수 완료?};
    D -- No --> E[서버 시작 실패/종료];
    D -- Yes --> F[클라이언트 요청 처리 시작];
    F --> G{서버 종료 명령어/신호 수신};
    G --> H{on_event("shutdown") 함수 실행};
    H --> I[모든 shutdown 함수 완료];
    I --> J[서버 프로세스 종료];
```
