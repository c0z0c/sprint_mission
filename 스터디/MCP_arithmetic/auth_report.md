---
layout: default
title: "MCP Arithmetic 보안 분석 보고서"
description: "MCP Arithmetic 보안 분석 보고서"
author: "김명환"
cache-control: no-cache
date: 2025-10-14
expires: 0
pragma: no-cache
version: 1.0
---

# MCP Arithmetic 보안 분석 보고서

## 1. 요약 (Executive Summary)

### 1.1. 핵심 결론

**현재 OAuth 보안 적용 상태: 미적용 (Not Implemented)**

- **전송 계층 (Transport Layer)**: stdio 기반 로컬 프로세스 통신, HTTP 서버는 개발용으로 인증 없음
- **인증 (Authentication)**: 없음
- **인가 (Authorization)**: 없음
- **암호화 (Encryption)**: 없음 (HTTP는 평문, stdio는 로컬 프로세스 격리에 의존)

### 1.2. 위험도 평가

| 실행 모드 | 현재 위험도 | 위험 요인 | 권장 조치 |
|----------|-----------|----------|---------|
| **stdio 서버** (Claude Desktop) | **낮음** (Low) | 로컬 프로세스 간 통신, 물리적 접근 필요 | 현재 유지 가능 |
| **HTTP 서버** (개발용) | **중간** (Medium) | localhost:8000 바인딩, 네트워크 노출 가능 | 프로덕션 배포 시 인증 필수 |
| **원격 배포 시** | **높음** (High) | 인증 없이 공개 네트워크 노출 | OAuth 2.0 또는 API Key 즉시 적용 |

---

## 2. 상세 분석 (Detailed Analysis)

### 2.1. 파일 구조 분석

```
MCP_arithmetic/
├── mcp_server.py          # stdio MCP 서버 (보안 없음)
├── app.py                 # HTTP REST API (보안 없음)
├── mcp_client.py          # 테스트 클라이언트 (보안 없음)
├── tools/arithmetic.py    # 비즈니스 로직 (보안 로직 없음)
├── requirements.txt       # 보안 관련 라이브러리 없음
└── claude_desktop_config.json  # Claude Desktop 설정 (인증 설정 없음)
```

### 2.2. 보안 메커니즘 부재 확인

#### 2.2.1. 인증 (Authentication) 관련 코드 검색 결과

```bash
# 검색 키워드: oauth, authentication, token, jwt, api_key, bearer
# 결과: No matches found
```

**결론**: 인증 로직이 전혀 구현되지 않음.

#### 2.2.2. 인가 (Authorization) 관련 코드 검색 결과

```bash
# 검색 키워드: security, authorize, permission
# 결과: No matches found
```

**결론**: 접근 제어 로직이 전혀 구현되지 않음.

#### 2.2.3. 의존성 분석

**`requirements.txt` 내용**:
```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
mcp>=1.0.0
```

**결론**: 보안 관련 라이브러리 없음 (authlib, python-jose, passlib 등 부재).

---

## 3. 실행 모드별 보안 분석

### 3.1. stdio MCP 서버 (`mcp_server.py`)

#### 3.1.1. 통신 구조

```mermaid
graph LR
    A["Claude Desktop<br/>(부모 프로세스)"] -->|"stdin/stdout<br/>(로컬 IPC)"| B["mcp_server.py<br/>(자식 프로세스)"]
    B --> C["tools/arithmetic.py<br/>(비즈니스 로직)"]
```

#### 3.1.2. 보안 특성

| 계층 | 보안 메커니즘 | 상태 | 설명 |
|------|-------------|-----|------|
| **전송 (Transport)** | stdio (표준 입출력) | ✅ 안전 | 프로세스 간 격리, 네트워크 노출 없음 |
| **인증 (Authentication)** | 없음 | ⚠️ 불필요 | 운영체제 프로세스 권한으로 제어 |
| **암호화 (Encryption)** | 없음 | ⚠️ 불필요 | 로컬 메모리 내 통신 |
| **인가 (Authorization)** | 없음 | ⚠️ 불필요 | 단일 사용자 환경 |

#### 3.1.3. 위협 모델 (Threat Model)

```mermaid
graph TD
    A["공격자"] -->|"물리적 접근"| B["사용자 컴퓨터"]
    B --> C["Claude Desktop<br/>프로세스 탈취"]
    C --> D["mcp_server.py<br/>임의 명령 실행"]
    
    E["네트워크 공격자"] -->|"차단됨"| F["stdio<br/>(네트워크 비노출)"]
```

**결론**: stdio 모드는 **로컬 환경에서 안전**하며 OAuth 불필요.

#### 3.1.4. 코드 분석: `mcp_server.py`

```python
# 인증 로직 없음
async def main():
    """stdio 서버 실행."""
    logger.info("Starting MCP Arithmetic Server (stdio)")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )
```

**분석**:
- stdin/stdout으로만 통신 (네트워크 소켓 비사용)
- Claude Desktop이 서버를 직접 실행하므로 프로세스 권한으로 접근 제어
- 외부 네트워크 노출 없음

---

### 3.2. HTTP REST API 서버 (`app.py`)

#### 3.2.1. 통신 구조

```mermaid
graph LR
    A["클라이언트<br/>(브라우저/curl)"] -->|"HTTP:8000<br/>(평문)"| B["FastAPI 서버<br/>(app.py)"]
    B --> C["tools/arithmetic.py<br/>(비즈니스 로직)"]
    
    D["악의적 클라이언트"] -.->|"인증 없이 접근 가능"| B
```

#### 3.2.2. 보안 특성

| 계층 | 보안 메커니즘 | 상태 | 위험도 |
|------|-------------|-----|--------|
| **전송 (Transport)** | HTTP (비암호화) | ❌ 위험 | 중간 트래픽 도청 가능 |
| **인증 (Authentication)** | 없음 | ❌ 위험 | 누구나 API 호출 가능 |
| **인가 (Authorization)** | 없음 | ❌ 위험 | 모든 작업 무제한 허용 |
| **입력 검증 (Validation)** | pydantic | ✅ 양호 | 타입 안전성 보장 |

#### 3.2.3. 취약점 분석

**코드 분석: `app.py`**:

```python
@app.post("/invoke", response_model=ToolResponse)
def invoke(req: ToolRequest) -> ToolResponse:
    """도구 호출 엔드포인트."""
    # ❌ 인증 없음: 누구나 호출 가능
    # ❌ Rate Limiting 없음: DoS 공격 취약
    # ✅ 입력 검증: pydantic으로 타입 체크
    
    if req.tool != "arithmetic":
        raise HTTPException(status_code=400, detail=f"Unknown tool: {req.tool}")
    
    tool = ArithmeticTool()
    result = tool.run(req.operation, req.operands)
    return ToolResponse(success=True, result=result, operation=req.operation)
```

**취약점 목록**:

1. **인증 부재 (No Authentication)**:
   ```bash
   # 누구나 실행 가능
   curl -X POST http://localhost:8000/invoke \
     -H "Content-Type: application/json" \
     -d '{"tool":"arithmetic","operation":"add","operands":[1,2]}'
   ```

2. **암호화 부재 (No Encryption)**:
   - HTTP 평문 통신 → 중간자 공격 (MITM) 가능

3. **Rate Limiting 부재**:
   - 무제한 요청 → 서비스 거부 공격 (DoS) 취약

4. **감사 로그 부족 (Limited Audit Trail)**:
   - 요청자 식별 불가 (IP만 로그)

#### 3.2.4. 위협 시나리오

```mermaid
sequenceDiagram
    participant Attacker as 공격자
    participant Server as HTTP 서버<br/>(인증 없음)
    participant Tool as ArithmeticTool
    
    Note over Attacker,Tool: 시나리오 1: 무단 접근
    Attacker->>Server: POST /invoke (인증 헤더 없음)
    Server->>Tool: 연산 실행
    Tool-->>Server: 결과 반환
    Server-->>Attacker: 200 OK (성공)
    
    Note over Attacker,Tool: 시나리오 2: DoS 공격
    loop 무제한 요청
        Attacker->>Server: POST /invoke
        Server->>Tool: 연산 실행 (부하 증가)
    end
    Server--xAttacker: 서버 과부하
```

---

## 4. OAuth 구현 가이드

### 4.1. OAuth 2.0 개요

#### 4.1.1. OAuth 2.0 플로우 비교

```mermaid
graph TB
    subgraph "현재 (인증 없음)"
        A1["클라이언트"] -->|"직접 호출"| B1["API 서버"]
    end
    
    subgraph "OAuth 2.0 적용 후"
        A2["클라이언트"] -->|"1. 로그인 요청"| C["OAuth 제공자<br/>(Google/GitHub)"]
        C -->|"2. Access Token 발급"| A2
        A2 -->|"3. Bearer Token 포함 요청"| B2["API 서버"]
        B2 -->|"4. Token 검증"| C
        C -->|"5. 검증 결과"| B2
        B2 -->|"6. 응답"| A2
    end
```

#### 4.1.2. OAuth 2.0 Grant Types

| Grant Type | 사용 시나리오 | 보안 수준 | 권장 여부 |
|-----------|------------|---------|----------|
| **Authorization Code** | 웹 앱, 백엔드 서버 | 높음 | ✅ 권장 |
| **Client Credentials** | 서버 간 통신 (M2M) | 중간 | ✅ 권장 |
| **Implicit** | SPA (구형) | 낮음 | ❌ 비권장 (deprecated) |
| **Password Grant** | 신뢰 앱 | 낮음 | ❌ 비권장 |
| **Device Code** | IoT 디바이스 | 중간 | 상황별 |

### 4.2. 구현 옵션

#### 4.2.1. 옵션 A: API Key 방식 (간단)

**장점**:
- 구현 간단 (1-2시간)
- 외부 의존성 없음

**단점**:
- 기본적인 보안만 제공
- 토큰 만료/갱신 로직 수동 구현 필요

**코드 예시**:

```python
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

# 환경 변수로 관리
VALID_API_KEYS = {
    "sk_test_1234567890abcdef": "user_alice",
    "sk_prod_abcdef1234567890": "user_bob",
}

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """
    API Key 검증.
    
    Args:
        credentials: Authorization 헤더의 Bearer 토큰
    
    Returns:
        str: 인증된 사용자 ID
    
    Raises:
        HTTPException: 인증 실패 시 401
    """
    token = credentials.credentials
    
    if token not in VALID_API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return VALID_API_KEYS[token]


@app.post("/invoke", response_model=ToolResponse)
def invoke(
    req: ToolRequest,
    user_id: str = Depends(verify_api_key)  # 의존성 주입
) -> ToolResponse:
    """도구 호출 (API Key 인증 필요)."""
    logger.info(f"User {user_id} called {req.operation}")
    
    tool = ArithmeticTool()
    result = tool.run(req.operation, req.operands)
    return ToolResponse(success=True, result=result, operation=req.operation)
```

**사용 예시**:
```bash
curl -X POST http://localhost:8000/invoke \
  -H "Authorization: Bearer sk_test_1234567890abcdef" \
  -H "Content-Type: application/json" \
  -d '{"tool":"arithmetic","operation":"add","operands":[1,2]}'
```

---

#### 4.2.2. 옵션 B: OAuth 2.0 with Google/GitHub (프로덕션)

**장점**:
- 업계 표준 (RFC 6749)
- 토큰 만료/갱신 자동 처리
- 사용자 정보 연동 (이메일, 프로필)

**단점**:
- 구현 복잡 (1-2일)
- 외부 서비스 의존성

**필수 라이브러리**:
```txt
# requirements.txt에 추가
authlib==1.3.0
python-jose[cryptography]==3.3.0
python-multipart==0.0.6
```

**코드 예시**:

```python
from authlib.integrations.starlette_client import OAuth
from starlette.middleware.sessions import SessionMiddleware
from jose import JWTError, jwt
from datetime import datetime, timedelta

# OAuth 설정
oauth = OAuth()
oauth.register(
    name='google',
    client_id='YOUR_GOOGLE_CLIENT_ID',
    client_secret='YOUR_GOOGLE_CLIENT_SECRET',
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'},
)

app.add_middleware(SessionMiddleware, secret_key="YOUR_SECRET_KEY")

# JWT 설정
SECRET_KEY = "your-256-bit-secret"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


def create_access_token(data: dict) -> str:
    """JWT Access Token 생성."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> dict:
    """
    JWT 토큰 검증.
    
    Args:
        token: JWT Access Token
    
    Returns:
        dict: 토큰 페이로드 (사용자 정보)
    
    Raises:
        HTTPException: 토큰 검증 실패 시 401
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Token validation failed")


@app.get("/login")
async def login(request: Request):
    """Google OAuth 로그인 시작."""
    redirect_uri = request.url_for('auth_callback')
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get("/auth/callback")
async def auth_callback(request: Request):
    """Google OAuth 콜백 처리."""
    token = await oauth.google.authorize_access_token(request)
    user = token.get('userinfo')
    
    # JWT 생성
    access_token = create_access_token(data={"sub": user['email']})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user,
    }


@app.post("/invoke", response_model=ToolResponse)
def invoke(
    req: ToolRequest,
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> ToolResponse:
    """도구 호출 (OAuth 인증 필요)."""
    # 토큰 검증
    payload = verify_token(credentials.credentials)
    user_email = payload["sub"]
    
    logger.info(f"User {user_email} called {req.operation}")
    
    tool = ArithmeticTool()
    result = tool.run(req.operation, req.operands)
    return ToolResponse(success=True, result=result, operation=req.operation)
```

**사용 플로우**:

```mermaid
sequenceDiagram
    participant User as 사용자
    participant Client as 클라이언트 앱
    participant API as FastAPI 서버
    participant Google as Google OAuth
    
    User->>Client: 1. 로그인 요청
    Client->>API: 2. GET /login
    API->>Google: 3. OAuth 인증 페이지로 리다이렉트
    User->>Google: 4. 구글 계정 로그인
    Google->>API: 5. Authorization Code 반환
    API->>Google: 6. Code로 Access Token 요청
    Google-->>API: 7. Access Token + User Info
    API->>API: 8. JWT 생성
    API-->>Client: 9. JWT 반환
    
    Client->>API: 10. POST /invoke<br/>(Authorization: Bearer JWT)
    API->>API: 11. JWT 검증
    API-->>Client: 12. 연산 결과 반환
```

---

#### 4.2.3. 옵션 C: MCP 프로토콜 확장 (stdio 모드)

**현재 MCP 표준 (v1.0.0)**:
- stdio 통신만 정의
- 인증 메커니즘 없음

**커스텀 인증 추가 (비표준)**:

```python
# mcp_server.py에 추가
import hashlib
import hmac

SHARED_SECRET = "your-shared-secret-key"


def verify_request_signature(data: dict, signature: str) -> bool:
    """
    요청 서명 검증 (HMAC-SHA256).
    
    Args:
        data: 요청 본문
        signature: 클라이언트가 제공한 서명
    
    Returns:
        bool: 서명 유효 여부
    """
    expected = hmac.new(
        SHARED_SECRET.encode(),
        json.dumps(data, sort_keys=True).encode(),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """도구 호출 핸들러 (서명 검증 추가)."""
    # 요청 서명 확인 (커스텀 헤더에서)
    signature = arguments.pop("_signature", None)
    if not signature or not verify_request_signature(arguments, signature):
        return [
            TextContent(
                type="text",
                text=json.dumps({"success": False, "error": "Invalid signature"}),
            )
        ]
    
    # 기존 로직 실행
    # ...
```

**주의사항**:
- MCP 표준에 포함되지 않음
- Claude Desktop이 서명을 추가하도록 수정 불가
- **실용성 낮음** → HTTP 모드에서 OAuth 사용 권장

---

### 4.3. 권장 구현 단계

```mermaid
graph LR
    A["1단계<br/>API Key<br/>(1-2시간)"] --> B["2단계<br/>HTTPS 적용<br/>(0.5일)"]
    B --> C["3단계<br/>Rate Limiting<br/>(0.5일)"]
    C --> D["4단계<br/>OAuth 2.0<br/>(1-2일)"]
    D --> E["5단계<br/>감사 로그<br/>(0.5일)"]
```

#### 4.3.1. 1단계: API Key (필수)

**목표**: 기본 인증 추가 (1-2시간)

**구현**:
1. `app.py`에 `verify_api_key` 함수 추가
2. `/invoke` 엔드포인트에 `Depends(verify_api_key)` 추가
3. 환경 변수로 API Key 관리

**테스트**:
```bash
# 인증 없이 요청 → 401 Unauthorized
curl -X POST http://localhost:8000/invoke \
  -H "Content-Type: application/json" \
  -d '{"tool":"arithmetic","operation":"add","operands":[1,2]}'

# 인증 포함 요청 → 200 OK
curl -X POST http://localhost:8000/invoke \
  -H "Authorization: Bearer sk_test_1234567890abcdef" \
  -H "Content-Type: application/json" \
  -d '{"tool":"arithmetic","operation":"add","operands":[1,2]}'
```

---

#### 4.3.2. 2단계: HTTPS 적용 (필수)

**목표**: 전송 계층 암호화 (0.5일)

**옵션 A: 로컬 개발 (Self-Signed Certificate)**:
```bash
# 인증서 생성
openssl req -x509 -newkey rsa:4096 -nodes \
  -keyout key.pem -out cert.pem -days 365 \
  -subj "/CN=localhost"

# uvicorn 실행
uvicorn app:app --host 0.0.0.0 --port 8443 \
  --ssl-keyfile key.pem --ssl-certfile cert.pem
```

**옵션 B: 프로덕션 (Let's Encrypt)**:
```bash
# Certbot으로 무료 인증서 발급
certbot certonly --standalone -d yourdomain.com
```

**테스트**:
```bash
curl -k https://localhost:8443/invoke \
  -H "Authorization: Bearer sk_test_1234567890abcdef" \
  -H "Content-Type: application/json" \
  -d '{"tool":"arithmetic","operation":"add","operands":[1,2]}'
```

---

#### 4.3.3. 3단계: Rate Limiting (권장)

**목표**: DoS 공격 방어 (0.5일)

**라이브러리**:
```txt
# requirements.txt에 추가
slowapi==0.1.9
```

**코드**:
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.post("/invoke", response_model=ToolResponse)
@limiter.limit("10/minute")  # 분당 10회 제한
def invoke(
    request: Request,
    req: ToolRequest,
    user_id: str = Depends(verify_api_key)
) -> ToolResponse:
    """도구 호출 (Rate Limiting 적용)."""
    # 기존 로직
    # ...
```

**테스트**:
```bash
# 11번째 요청 시 429 Too Many Requests
for i in {1..11}; do
  curl -X POST http://localhost:8000/invoke \
    -H "Authorization: Bearer sk_test_1234567890abcdef" \
    -H "Content-Type: application/json" \
    -d '{"tool":"arithmetic","operation":"add","operands":[1,2]}'
done
```

---

#### 4.3.4. 4단계: OAuth 2.0 (프로덕션)

**목표**: 산업 표준 인증 (1-2일)

**구현**: 섹션 4.2.2 참조

---

#### 4.3.5. 5단계: 감사 로그 (규정 준수)

**목표**: 모든 요청 추적 (0.5일)

**코드**:
```python
import logging
from datetime import datetime

audit_logger = logging.getLogger("audit")
audit_logger.setLevel(logging.INFO)

# 파일 핸들러 추가
handler = logging.FileHandler("audit.log")
handler.setFormatter(logging.Formatter(
    '{"timestamp": "%(asctime)s", "user": "%(user)s", "operation": "%(operation)s", "result": "%(result)s"}'
))
audit_logger.addHandler(handler)


@app.post("/invoke", response_model=ToolResponse)
def invoke(
    req: ToolRequest,
    user_id: str = Depends(verify_api_key)
) -> ToolResponse:
    """도구 호출 (감사 로그 추가)."""
    tool = ArithmeticTool()
    result = tool.run(req.operation, req.operands)
    
    # 감사 로그 기록
    audit_logger.info(
        "API call",
        extra={
            "user": user_id,
            "operation": req.operation,
            "result": "success",
        }
    )
    
    return ToolResponse(success=True, result=result, operation=req.operation)
```

**로그 예시** (`audit.log`):
```json
{"timestamp": "2025-10-14 10:30:45", "user": "user_alice", "operation": "add", "result": "success"}
{"timestamp": "2025-10-14 10:31:12", "user": "user_bob", "operation": "multiply", "result": "success"}
```

---

## 5. 환경별 보안 전략

### 5.1. 로컬 개발 환경

```mermaid
graph LR
    A["개발자<br/>컴퓨터"] --> B["stdio 서버<br/>(인증 없음)"]
    A --> C["HTTP 서버<br/>(localhost:8000)"]
    C --> D["API Key<br/>(선택사항)"]
```

**권장 구성**:
- **stdio 서버**: 인증 불필요 (로컬 프로세스 격리)
- **HTTP 서버**: API Key 추가 (개발 습관 형성)
- **네트워크**: localhost 바인딩 (`--host 127.0.0.1`)

**설정 예시**:
```python
# run_server.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="127.0.0.1",  # localhost만 허용
        port=8000,
        reload=True,
    )
```

---

### 5.2. 프로덕션 환경

```mermaid
graph TB
    A["인터넷"] --> B["리버스 프록시<br/>(Nginx/Caddy)"]
    B --> C["HTTPS<br/>(TLS 1.3)"]
    C --> D["FastAPI 서버"]
    D --> E["OAuth 2.0<br/>검증"]
    E --> F["Rate Limiter"]
    F --> G["비즈니스 로직"]
    
    H["감사 로그"] --> I["SIEM<br/>(Splunk/ELK)"]
    D --> H
```

**필수 구성**:
1. **HTTPS (TLS 1.3)**: 전송 계층 암호화
2. **OAuth 2.0**: 사용자 인증
3. **Rate Limiting**: DoS 방어
4. **감사 로그**: 규정 준수 (GDPR, SOC 2)
5. **방화벽**: 허용 IP 화이트리스트

**Nginx 설정 예시**:
```nginx
upstream fastapi_backend {
    server 127.0.0.1:8000;
}

server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    # TLS 설정
    ssl_certificate /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;
    ssl_protocols TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Rate Limiting (Nginx 레벨)
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req zone=api_limit burst=20 nodelay;

    location / {
        proxy_pass http://fastapi_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

### 5.3. Claude Desktop 통합 환경

```mermaid
graph LR
    A["Claude Desktop"] -->|"stdio<br/>(로컬 IPC)"| B["mcp_server.py"]
    B --> C["tools/arithmetic.py"]
    
    D["물리적 보안"] --> E["컴퓨터 잠금<br/>계정 암호"]
    E --> A
```

**권장 구성**:
- **인증**: 불필요 (운영체제 프로세스 권한으로 제어)
- **물리적 보안**: 
  - 컴퓨터 잠금 (Windows Hello, Touch ID)
  - 디스크 암호화 (BitLocker, FileVault)
- **네트워크 격리**: 외부 노출 없음

**Claude Desktop 설정**:
```json
{
  "mcpServers": {
    "arithmetic": {
      "command": "python",
      "args": ["d:\\path\\to\\mcp_server.py"],
      // 인증 설정 없음 (stdio는 불필요)
    }
  }
}
```

---

## 6. 보안 체크리스트

### 6.1. 개발 단계

- [ ] **코드 리뷰**: 인증 로직 검증
- [ ] **의존성 스캔**: `pip-audit` 또는 `safety` 사용
  ```bash
  pip install pip-audit
  pip-audit
  ```
- [ ] **정적 분석**: `bandit`으로 취약점 스캔
  ```bash
  pip install bandit
  bandit -r . -ll
  ```
- [ ] **비밀 관리**: `.env` 파일 사용, `.gitignore`에 추가
  ```bash
  # .env
  API_KEY=sk_test_1234567890abcdef
  OAUTH_CLIENT_SECRET=your_secret_here
  ```

---

### 6.2. 배포 단계

- [ ] **HTTPS 강제**: HTTP 요청 자동 리다이렉트
- [ ] **CORS 설정**: 허용 오리진 화이트리스트
  ```python
  from fastapi.middleware.cors import CORSMiddleware
  
  app.add_middleware(
      CORSMiddleware,
      allow_origins=["https://yourdomain.com"],  # 특정 도메인만
      allow_credentials=True,
      allow_methods=["POST"],  # 필요한 메서드만
      allow_headers=["Authorization", "Content-Type"],
  )
  ```
- [ ] **환경 변수**: 민감 정보 외부화
- [ ] **로그 모니터링**: 실시간 알림 설정
- [ ] **백업**: 정기 백업 + 복구 테스트

---

### 6.3. 운영 단계

- [ ] **침투 테스트**: 연 1회 보안 감사
- [ ] **취약점 패치**: CVE 모니터링 + 즉시 업데이트
- [ ] **사고 대응 계획**: 인증 정보 유출 시 프로토콜 수립
- [ ] **규정 준수**: GDPR, PCI-DSS, SOC 2 (필요 시)

---

## 7. 비용-효과 분석

### 7.1. 구현 비용

| 보안 계층 | 구현 시간 | 연간 유지비용 | ROI (투자 대비 효과) |
|----------|---------|-------------|---------------------|
| **API Key** | 1-2시간 | 무료 | 매우 높음 (기본 인증) |
| **HTTPS** | 0.5일 | 무료 (Let's Encrypt) | 매우 높음 (MITM 방어) |
| **Rate Limiting** | 0.5일 | 무료 | 높음 (DoS 방어) |
| **OAuth 2.0** | 1-2일 | 무료 (Google/GitHub) | 중간 (복잡도 증가) |
| **감사 로그** | 0.5일 | \$10-50/월 (로그 저장) | 중간 (규정 준수) |
| **침투 테스트** | - | \$1,000-5,000/회 | 높음 (취약점 조기 발견) |

**총 초기 투자**: 2-4일 (16-32시간)  
**연간 운영 비용**: \$120-600 (로그 + 도구)

---

### 7.2. 위험 비용 (보안 미적용 시)

| 사고 유형 | 발생 확률 | 예상 피해액 | 방어 방법 |
|----------|---------|-----------|---------|
| **무단 접근** | 높음 | \$0-10,000 (데이터 유출) | API Key |
| **DoS 공격** | 중간 | \$1,000-50,000 (서비스 중단) | Rate Limiting |
| **MITM 공격** | 중간 | \$5,000-100,000 (데이터 도청) | HTTPS |
| **계정 탈취** | 낮음 | \$10,000-500,000 (법적 책임) | OAuth 2.0 |

**예상 연간 손실** (보안 미적용): \$16,000-660,000  
**보안 투자 대비 효과**: **1:40 ~ 1:1,100** (ROI 4,000% ~ 110,000%)

---

## 8. 결론 및 권장 사항

### 8.1. 즉시 조치 사항 (Critical)

1. **HTTP 서버에 API Key 추가** (1-2시간)
   - 현재: 누구나 접근 가능 (보안 위험)
   - 목표: 기본 인증 추가

2. **HTTPS 적용** (0.5일)
   - 현재: 평문 통신 (MITM 취약)
   - 목표: TLS 1.3 암호화

3. **Rate Limiting 추가** (0.5일)
   - 현재: 무제한 요청 (DoS 취약)
   - 목표: 분당 10-100회 제한

---

### 8.2. 중기 개선 사항 (1-2주)

4. **OAuth 2.0 통합** (1-2일)
   - 현재: 정적 API Key
   - 목표: Google/GitHub 소셜 로그인

5. **감사 로그 시스템** (0.5일)
   - 현재: 기본 로깅만
   - 목표: 모든 API 호출 추적

---

### 8.3. 장기 전략 (1-3개월)

6. **침투 테스트**: 외부 보안 업체에 의뢰
7. **버그 바운티 프로그램**: HackerOne 또는 자체 운영
8. **ISO 27001 인증**: 엔터프라이즈 고객 대응

---

### 8.4. stdio 서버 (Claude Desktop)

**현재 상태: 유지 권장**
- stdio 통신은 로컬 프로세스 간 격리로 충분히 안전
- OAuth 추가는 불필요 (오버 엔지니어링)
- 물리적 보안 (컴퓨터 잠금)으로 충분

---

## 9. 참고 자료

### 9.1. 공식 문서

- [OAuth 2.0 RFC 6749](https://datatracker.ietf.org/doc/html/rfc6749)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [MCP Protocol Specification](https://modelcontextprotocol.io/)

### 9.2. 라이브러리

- [authlib](https://docs.authlib.org/): OAuth 1.0/2.0 클라이언트/서버
- [python-jose](https://python-jose.readthedocs.io/): JWT 생성/검증
- [slowapi](https://github.com/laurentS/slowapi): FastAPI Rate Limiting

### 9.3. 도구

- [pip-audit](https://pypi.org/project/pip-audit/): 의존성 취약점 스캔
- [bandit](https://bandit.readthedocs.io/): Python 보안 정적 분석
- [Postman](https://www.postman.com/): API 테스트
- [OWASP ZAP](https://www.zaproxy.org/): 웹 애플리케이션 취약점 스캐너

---

## 10. 부록

### 10.1. 용어 정의

- **OAuth 2.0**: Open Authorization 2.0, 인증 및 인가 프레임워크
- **JWT**: JSON Web Token, 토큰 기반 인증 표준 (RFC 7519)
- **MITM**: Man-in-the-Middle Attack, 중간자 공격
- **DoS**: Denial of Service, 서비스 거부 공격
- **TLS**: Transport Layer Security, 전송 계층 보안 (HTTPS의 기반)
- **CORS**: Cross-Origin Resource Sharing, 교차 출처 리소스 공유
- **Rate Limiting**: 요청 빈도 제한 (DDoS 방어)
