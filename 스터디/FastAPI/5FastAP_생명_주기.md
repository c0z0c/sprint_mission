---
layout: default
title: "FastAPI - FastAPI에서 사용하는 생명 주기(Lifecycle) 이벤트"
description: "FastAPI - FastAPI에서 사용하는 생명 주기(Lifecycle) 이벤트"
date: 2025-12-15
cache-control: no-cache
expires: 0
pragma: no-cache
author: "김명환"
---

## 🔄 FastAPI에서 사용하는 생명 주기(Lifecycle) 이벤트 정리

FastAPI는 Starlette을 기반으로 하며, 애플리케이션의 시작부터 종료까지의 흐름을 관리하기 위해 **생명 주기 이벤트 핸들러**를 사용합니다. 현재는 \*\*`lifespan`\*\*이 가장 권장되는 표준 방식이며, 이전에는 `on_event`가 사용되었습니다.

### 1\. 권장 표준: `lifespan` (ASGI 표준 준수)

`lifespan` 이벤트 핸들러는 Python의 **비동기 컨텍스트 관리자**(`asynccontextmanager`)를 사용하여 서버 시작 및 종료 시의 작업을 하나의 구조 내에서 관리합니다.

#### 1.1. 개요 및 구조

| 구성 요소 | 설명 | `on_event` 대응 |
| :--- | :--- | :--- |
| **`@asynccontextmanager`** | `lifespan` 함수를 컨텍스트 관리자로 정의하는 데코레이터입니다. | 해당 없음 |
| **`yield` 이전 코드** | 서버가 요청을 처리하기 **직전**에 실행되는 시작(Startup) 로직입니다. 모델 로드, DB 연결 초기화 등의 작업을 수행합니다. | `on_event("startup")` |
| **`yield`** | 이 키워드를 만나면 FastAPI 애플리케이션이 **요청 처리를 시작**하고, 서버가 실행 상태에 들어갑니다. | 서버 실행 시작 |
| **`yield` 이후 코드** | 서버가 종료 신호를 받은 후, 모든 요청 처리가 완료된 **직후**에 실행되는 종료(Shutdown) 로직입니다. 자원 해제, 연결 종료 등의 작업을 수행합니다. | `on_event("shutdown")` |

#### 1.2. 사용 예시

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
import some_db_library # 가상의 비동기 DB 라이브러리

db_pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- [ 🚀 Startup Logic (서버 시작) ] ---
    # 비동기 DB 연결 풀 초기화 (DB 연결은 리소스 소모가 큼)
    global db_pool
    db_pool = await some_db_library.connect_pool() 
    print("INFO: Database Pool Initialized.")
    
    yield # <<< 서버가 요청 처리를 시작

    # --- [ 🛑 Shutdown Logic (서버 종료) ] ---
    # DB 연결 풀 해제 및 정리
    await db_pool.close()
    print("INFO: Database Pool Closed.")

app = FastAPI(lifespan=lifespan)
```

### 2\. 이전 방식: `on_event` (Deprecated)

이 방식은 구버전 Starlette 및 FastAPI에서 사용되었으며, 현재는 사용 중단(Deprecated)이 권고됩니다.

#### 2.1. 이벤트 유형

| 이벤트 종류 | 실행 시점 | 함수 형식 |
| :--- | :--- | :--- |
| **`"startup"`** | 서버가 요청을 처리하기 **직전** | `def` 또는 `async def` |
| **`"shutdown"`** | 서버가 요청 처리를 마친 **직후** | `def` 또는 `async def` |

#### 2.2. 사용 예시 (이전 방식)

```python
# (이 코드는 현재 권장되지 않습니다)
from fastapi import FastAPI
app = FastAPI()

@app.on_event("startup")
def load_model():
    print("Startup: 모델 로드 시작")

@app.on_event("shutdown")
async def close_database():
    print("Shutdown: DB 연결 종료")
```

### 3\. FastAPI 생명 주기 이벤트의 중요성

생명 주기 이벤트를 활용하는 것은 마이크로서비스 설계에서 매우 중요합니다.

1.  **성능 최적화**: 모델이나 DB 연결과 같이 **초기화 비용이 높은 자원**을 서버 시작 시 한 번만 로드하여, 개별 HTTP 요청 처리 시간을 단축합니다.
2.  **자원 관리**: 서버가 비정상적으로 종료되는 것을 대비하여 `shutdown` 이벤트를 통해 열려 있는 파일, 소켓, DB 연결 등을 안전하게 \*\*해제(Close)\*\*하여 메모리 누수나 자원 고갈을 방지합니다.
3.  **환경 설정**: 애플리케이션 전역에서 사용될 환경 변수나 설정값을 로드하는 시점을 명확하게 정의할 수 있습니다.
