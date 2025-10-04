---
title: "나만의 MCP 서버 만들기: 사칙연산 도구 개발"
date: 2025-10-05
categories: [AI, MCP, Tutorial, Python]
tags: [claude, mcp, python, custom-server, development]
---

# 나만의 MCP 서버 만들기: 사칙연산 도구 개발

## 목차

1. [개요](#1-개요)<br/>
   1. [커스텀 MCP 서버의 필요성](#11-커스텀-mcp-서버의-필요성)<br/>
   1. [학습 목표](#12-학습-목표)<br/>
1. [MCP 서버 구조 이해](#2-mcp-서버-구조-이해)<br/>
   1. [기본 아키텍처](#21-기본-아키텍처)<br/>
   1. [핵심 구성 요소](#22-핵심-구성-요소)<br/>
   1. [통신 프로토콜](#23-통신-프로토콜)<br/>
1. [개발 환경 준비](#3-개발-환경-준비)<br/>
   1. [프로젝트 디렉토리 생성](#31-프로젝트-디렉토리-생성)<br/>
   1. [가상 환경 설정](#32-가상-환경-설정)<br/>
   1. [필요한 라이브러리 설치](#33-필요한-라이브러리-설치)<br/>
1. [사칙연산 MCP 서버 구현](#4-사칙연산-mcp-서버-구현)<br/>
   1. [기본 구조 작성](#41-기본-구조-작성)<br/>
   1. [도구 함수 구현](#42-도구-함수-구현)<br/>
   1. [서버 초기화 및 실행](#43-서버-초기화-및-실행)<br/>
1. [Claude Desktop 연결](#5-claude-desktop-연결)<br/>
   1. [설정 파일 수정](#51-설정-파일-수정)<br/>
   1. [다중 MCP 서버 관리](#52-다중-mcp-서버-관리)<br/>
1. [테스트 및 검증](#6-테스트-및-검증)<br/>
   1. [로컬 테스트](#61-로컬-테스트)<br/>
   1. [Claude와 통합 테스트](#62-claude와-통합-테스트)<br/>
   1. [오류 처리 테스트](#63-오류-처리-테스트)<br/>
1. [추가 제안: 보안 및 확장성](#7-추가-제안-보안-및-확장성)<br/>
   1. [인증 및 권한 관리](#71-인증-및-권한-관리)<br/>
   1. [서버 확장 전략](#72-서버-확장-전략)<br/>
1. [용어 목록](#8-용어-목록)<br/>

---

## 1. 개요

### 1.1 커스텀 MCP 서버의 필요성

AI 연구와 실험에서 외부 도구 호출을 직접 제어할 수 있는 MCP(Microsoft Copilot Plugin) 서버를 만드는 것은 모델-도구 통합을 이해하고 디버깅하는 데 결정적이다.<br/>
직접 구현하면 호출 규약, 입력 검증, 에러 경계 케이스를 체계적으로 학습할 수 있다.

### 1.2 학습 목표

- MCP 서버 기본 구조 이해  
- 사칙연산 도구(더하기, 빼기, 곱하기, 나누기) 구현  
- Claude Desktop 또는 유사한 클라이언트와 연동하는 설정 파일 구성  
- 로컬/통합 테스트 및 예외 처리 전략 습득

---

## 2. MCP 서버 구조 이해

### 2.1 기본 아키텍처

- 클라이언트(예: Claude Desktop) ↔ HTTP 엔드포인트(Plugin 서버) ↔ 도구 함수  
- 서버는 JSON 기반 요청/응답을 사용하고, 도구 실행 결과를 표준화된 스키마로 반환한다.

간단한 아키텍처 다이어그램 (Mermaid 사용 예시):

```mermaid
graph LR
  A["클라이언트"] --> B["MCP 서버 엔드포인트"]
  B["MCP 서버 엔드포인트"] --> C["사칙연산 도구 모듈"]
  C["사칙연산 도구 모듈"] --> D["결과 검증 및 응답"]
```

### 2.2 핵심 구성 요소

- 엔드포인트 라우터: 요청을 적절한 도구로 라우팅  
- 도구 인터페이스: 입력 스키마 검증과 출력 포맷 통일  
- 에러/로깅 모듈: 실패 원인 추적과 재현성 보장  
- 보안 구성: 인증, CORS, 입력 사이즈 제한

어려운 용어 표기는 본문에서 자연스럽게 발음 표기(예: 스키마(schema: 스키마))를 함께 제공한다.

### 2.3 통신 프로토콜

- HTTP/1.1 또는 HTTP/2 사용 가능  
- 요청/응답은 JSON; 요청 예시는 다음과 같다:

```json
{
  "tool": "arithmetic",
  "operation": "add",
  "operands": [1.5, 2.3]
}
```

- 응답은 표준화된 오브젝트로 반환:

```json
{
  "success": true,
  "result": 3.8,
  "meta": {"runtime_ms": 2}
}
```

---

## 3. 개발 환경 준비

### 3.1 프로젝트 디렉토리 생성

권장 디렉토리 구조:

- mcp-arith/  
  - app.py  
  - tools/  
    - arithmetic.py  
  - requirements.txt  
  - README.md

### 3.2 가상 환경 설정

- Python 가상환경 생성 권장: venv 또는 conda 사용  
- 예: python -m venv .venv; source .venv/bin/activate

### 3.3 필요한 라이브러리 설치

- 최소: Flask 또는 FastAPI, pydantic(입력 검증), gunicorn(프로덕션)  
- requirements.txt 예시:

```
fastapi
uvicorn
pydantic
```

---

## 4. 사칙연산 MCP 서버 구현

### 4.1 기본 구조 작성

FastAPI 기반 최소 서버 예시 (파일: app.py):

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tools.arithmetic import ArithmeticTool

app = FastAPI()

class ToolRequest(BaseModel):
    tool: str
    operation: str
    operands: list[float]

@app.post("/invoke")
def invoke(req: ToolRequest):
    if req.tool != "arithmetic":
        raise HTTPException(status_code=400, detail="Unknown tool")
    tool = ArithmeticTool()
    try:
        result = tool.run(req.operation, req.operands)
        return {"success": True, "result": result, "meta": {"tool": "arithmetic"}}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

### 4.2 도구 함수 구현

도구 모듈 (파일: tools/arithmetic.py):

```python
class ArithmeticTool:
    def run(self, operation: str, operands: list[float]) -> float:
        if not operands:
            raise ValueError("Operands required")
        if operation == "add":
            return sum(operands)
        if operation == "sub":
            from functools import reduce
            return reduce(lambda a, b: a - b, operands)
        if operation == "mul":
            from functools import reduce
            return reduce(lambda a, b: a * b, operands)
        if operation == "div":
            from functools import reduce
            def safe_div(a, b):
                if b == 0:
                    raise ZeroDivisionError("Division by zero")
                return a / b
            return reduce(safe_div, operands)
        raise ValueError("Unsupported operation")
```

- 입력 검증: pydantic을 이용해 타입과 범위를 엄격히 검사한다.  
- 예외 처리: 0으로 나누기, 비어있는 피연산자 등은 명확한 메시지로 반환한다.

### 4.3 서버 초기화 및 실행

로컬 실행:

- 개발: uvicorn app:app --reload --port 8000  
- 프로덕션: gunicorn -k uvicorn.workers.UvicornWorker app:app -w 4

---

## 5. Claude Desktop 연결

### 5.1 설정 파일 수정

Claude Desktop(또는 유사 클라이언트)에 플러그인으로 등록하려면 JSON 또는 YAML 형식의 설정 파일에 엔드포인트를 추가한다. 예시(간단한 표현):

```json
{
  "name": "Local Arithmetic MCP",
  "url": "http://localhost:8000/invoke",
  "auth": null
}
```

- 인증이 필요한 경우 Bearer 토큰 또는 API 키를 설정한다.

### 5.2 다중 MCP 서버 관리

- 여러 도구를 분리된 경로로 배치하거나 단일 서버에서 도구 네임스페이스로 관리한다.  
- 라우팅 규칙: /invoke?tool=arithmetic 또는 /tools/arithmetic/invoke 처럼 명확한 엔드포인트 설계 권장.

---

## 6. 테스트 및 검증

### 6.1 로컬 테스트

- curl 또는 HTTP 클라이언트를 사용한 단위 테스트:

```bash
curl -X POST http://localhost:8000/invoke \
  -H "Content-Type: application/json" \
  -d '{"tool":"arithmetic","operation":"add","operands":[1,2,3]}'
```

- 기대 응답: {"success":true,"result":6,...}

### 6.2 Claude와 통합 테스트

- Claude에서 도구 호출을 유도하는 프롬프트를 작성해 실제 호출 흐름 확인.  
- 요청/응답 로그를 남겨 latency와 실패 케이스를 분석한다.

### 6.3 오류 처리 테스트

- 경계 케이스: 빈 배열, 비수치 입력, 0으로 나누기, 매우 큰 수(오버플로우) 등 시나리오를 작성해 자동화된 테스트로 검증한다.  
- 예외 메시지는 클라이언트가 재시도/백오프/사용자 메시지로 변환하기 쉬운 구조로 유지한다.

---

## 7. 추가 제안 보안 및 확장성

### 7.1 인증 및 권한 관리

- 내부용이면 IP 화이트리스트, 외부용이면 API 키 또는 OAuth2 적용.  
- 권한 분리: 읽기 전용, 실행 권한을 도구별로 구분한다.

### 7.2 서버 확장 전략

- 수평 확장: 컨테이너화(Docker) 후 오토스케일링 그룹 배포  
- 캐싱: 동일 입력에 대한 결과를 캐시해 반복 호출 비용 절감  
- 모니터링: Prometheus + Grafana로 메트릭과 에러율 감시

---

## 8. 용어 목록

| 용어 | 정의 |
|---|---|
| MCP | Microsoft Copilot Plugin의 약어, 모델이 외부 도구를 호출하기 위한 플러그인 인터페이스 |
| 엔드포인트 | 서버에서 특정 기능을 제공하는 URL 경로 |
| 스키마 | 데이터의 구조와 타입을 정의한 규약 |
| pydantic | Python 데이터 검증/설계 라이브러리 |
| FastAPI | 비동기 Python 웹 프레임워크 |
| uvicorn | ASGI 서버, FastAPI와 함께 사용 |
| JSON | 데이터 교환 형식인 JavaScript Object Notation |
| 라우터 | 요청을 처리할 함수로 연결하는 구성 요소 |
| 오토스케일링 | 부하에 따라 서버 인스턴스를 자동으로 늘이거나 줄이는 기법 |
| 캐싱 | 계산 결과를 임시로 저장해 반복 계산을 줄이는 기법 |


