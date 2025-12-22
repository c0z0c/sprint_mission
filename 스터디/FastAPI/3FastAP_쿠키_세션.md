---
layout: default
title: "FastAPI - 쿠키, 캐시, 세션, JWT"
description: "FastAPI - 쿠키, 캐시, 세션, JWT"
date: 2025-12-15
cache-control: no-cache
expires: 0
pragma: no-cache
author: "김명환"
---

FastAPI에서 자주 쓰이는 **쿠키, 캐시, 세션, JWT 등 상태 관리 기능**

---

# 🍪 쿠키 (Cookies)

## 개념
- 클라이언트(브라우저)에 저장되는 작은 데이터  
- 로그인 상태 유지, 사용자 맞춤 설정 등에 활용  

## 예제
```python
from fastapi import FastAPI, Response, Request

app = FastAPI()

@app.post("/set-cookie")
def set_cookie(response: Response):
    response.set_cookie(key="session_id", value="abc123", httponly=True)
    return ⦃"msg": "쿠키 설정 완료"❵

@app.get("/get-cookie")
def get_cookie(request: Request):
    session_id = request.cookies.get("session_id")
    return ⦃"session_id": session_id❵
```
👉 `/set-cookie`로 쿠키를 설정하고, `/get-cookie`로 읽을 수 있습니다.

---

# ⚡ 캐시 (Cache)

## 개념
- 서버나 외부 저장소(Redis, Memcached)에 데이터를 임시 저장  
- 자주 쓰이는 데이터를 빠르게 제공해 성능 향상  

## 예제 (Redis 활용)
```python
import aioredis
from fastapi import FastAPI

app = FastAPI()
redis = aioredis.from_url("redis://localhost")

@app.get("/cached")
async def cached_data():
    data = await redis.get("mykey")
    if data:
        return ⦃"cached": data❵
    new_data = "fresh result"
    await redis.set("mykey", new_data, ex=60)  # TTL 60초
    return ⦃"cached": new_data❵
```
👉 처음 요청 시 새 데이터를 저장하고, 이후 60초 동안은 캐시된 데이터를 반환합니다.

---

# 🔑 세션 (Sessions)

## 개념
- 쿠키는 클라이언트에 저장, 세션은 서버에 저장된 상태를 참조  
- 로그인 상태 유지, 사용자별 데이터 관리에 활용  

## 예제 (간단 구현)
```python
from fastapi import FastAPI, Request

app = FastAPI()
sessions = ⦃❵

@app.post("/login")
def login(request: Request):
    user_id = "u001"
    sessions[user_id] = ⦃"logged_in": True❵
    return ⦃"msg": "로그인 성공", "session": sessions[user_id]❵

@app.get("/session/⦃user_id❵")
def get_session(user_id: str):
    return sessions.get(user_id, ⦃"msg": "세션 없음"❵)
```
👉 실제 운영에서는 Redis 같은 중앙 저장소를 사용해 세션을 관리합니다.

---

# 🔐 JWT (JSON Web Token)

## 개념
- 세션 대신 자주 쓰이는 인증 방식  
- 클라이언트가 토큰을 보관하고 요청 시 헤더에 포함  

## 예제
```python
from fastapi import FastAPI, Depends
from fastapi.security import OAuth2PasswordBearer

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.get("/secure")
def secure_endpoint(token: str = Depends(oauth2_scheme)):
    return ⦃"msg": "토큰 인증 성공", "token": token❵
```
👉 클라이언트가 `Authorization: Bearer <token>` 헤더를 보내면 인증됩니다.

---

# 📑 정리
- **쿠키** → 클라이언트 상태 관리 (브라우저 저장)  
- **캐시** → 서버 성능 최적화 (Redis 등 외부 저장소)  
- **세션** → 서버 상태 관리 (로그인 유지, 사용자별 데이터)  
- **JWT** → 토큰 기반 인증 (세션 대체, 분산 환경에 적합)  

