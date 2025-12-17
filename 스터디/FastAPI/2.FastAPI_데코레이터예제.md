---
layout: default
title: "FastAPI - 데코레이터 심플 예제"
description: "FastAPI - 데코레이터 심플 예제"
date: 2025-12-15
cache-control: no-cache
expires: 0
pragma: no-cache
author: "김명환"
---

# 📖 FastAPI 데코레이터 심플 예제

## 1. `@app.get` (조회)
- **상황**: 사용자 목록을 가져오기  
```python
@app.get("/users")
def get_users():
    return list(users.values())
```
브라우저에서 `/users`로 접속하면 전체 사용자 목록을 JSON으로 확인할 수 있습니다.  

---

## 2. `@app.post` (생성)
- **상황**: 새로운 사용자 등록  
```python
@app.post("/users")
def create_user(user: dict):
    users[user["id"]] = user
    return {"msg": "사용자 생성 완료", "user": user}
```
클라이언트가 JSON 본문으로 `{ "id": "u003", "name": "홍길동" }`을 보내면 새로운 사용자가 추가됩니다.  

---

## 3. `@app.put` (전체 수정)
- **상황**: 특정 사용자 정보를 전체 교체  
```python
@app.put("/users/{user_id}")
def update_user(user_id: str, user: dict):
    users[user_id] = user
    return {"msg": "사용자 전체 수정 완료", "user": user}
```
기존 사용자 데이터를 완전히 새 데이터로 교체합니다.  

---

## 4. `@app.patch` (부분 수정)
- **상황**: 특정 사용자 정보 일부만 수정  
```python
@app.patch("/users/{user_id}")
def partial_update(user_id: str, user: dict):
    users[user_id].update(user)
    return {"msg": "사용자 일부 수정 완료", "user": users[user_id]}
```
예를 들어 `{"age": 35}`만 보내면 해당 사용자 나이만 변경됩니다.  

---

## 5. `@app.delete` (삭제)
- **상황**: 특정 사용자 삭제  
```python
@app.delete("/users/{user_id}")
def delete_user(user_id: str):
    del users[user_id]
    return {"msg": f"{user_id} 삭제 완료"}
```
`/users/u001`로 DELETE 요청을 보내면 해당 사용자가 제거됩니다.  

---

## 6. `@app.options` (옵션 확인)
- **상황**: 클라이언트가 지원되는 메서드 확인  
```python
@app.options("/users")
def options_users():
    return {"allow": ["GET", "POST", "PUT", "PATCH", "DELETE"]}
```
브라우저나 클라이언트가 서버가 어떤 메서드를 지원하는지 확인할 때 사용됩니다.  

---

## 7. `@app.head` (헤더만 반환)
- **상황**: 사용자 API의 메타데이터 확인  
```python
@app.head("/users")
def head_users():
    return {"X-Total-Count": len(users)}
```
본문 없이 헤더만 반환하여, 전체 사용자 수 같은 정보를 확인할 수 있습니다.  

---

# 📑 정리
- **GET** → 데이터 조회 (목록, 상세 보기)  
- **POST** → 데이터 생성 (회원가입, 글 작성)  
- **PUT** → 전체 수정 (프로필 전체 변경)  
- **PATCH** → 부분 수정 (프로필 일부 변경)  
- **DELETE** → 데이터 삭제 (계정 삭제)  
- **OPTIONS** → 서버가 지원하는 메서드 확인  
- **HEAD** → 본문 없이 헤더만 반환 (리소스 존재 여부, 메타데이터 확인)  
