---
layout: default
title: "FastAPI - 기본 구조"
description: "FastAPI - 기본 구조"
date: 2025-12-15
cache-control: no-cache
expires: 0
pragma: no-cache
author: "김명환"
---

# ⚙️ 운영 환경에서의 Uvicorn 최적화 설정 방법

## 1. 실행 구조
- **개발 환경**: `uvicorn main:app --reload`  
- **운영 환경**: `gunicorn -k uvicorn.workers.UvicornWorker main:app --workers 4 --bind 0.0.0.0:8000`  
  - Gunicorn이 프로세스 관리와 로드 밸런싱을 담당  
  - Uvicorn은 ASGI 서버로 실제 요청 처리  

---

## 2. 주요 설정 포인트

### 🔹 워커(Workers) 관리
- 운영 환경에서는 **멀티 워커**를 반드시 사용해야 합니다.  
- CPU 코어 수에 따라 `workers = 2 * cores + 1` 공식이 자주 활용됩니다.  
- 예: 4코어 서버 → `workers=9`  

### 🔹 호스트와 포트
- `host="0.0.0.0"` → 외부 접속 허용  
- `port=8000` 또는 운영 환경에 맞는 포트 지정  

### 🔹 SSL/TLS 적용
- HTTPS를 직접 적용하려면 `--ssl-certfile`과 `--ssl-keyfile` 옵션을 사용  
- 보통은 **Nginx/Apache 리버스 프록시**에서 SSL을 처리하고, Uvicorn은 내부 통신만 담당  

### 🔹 로깅 및 모니터링
- `log_level="info"` 또는 `debug`로 설정  
- 운영 환경에서는 **access log**와 **error log**를 분리 관리  
- Prometheus, Grafana 같은 모니터링 툴과 연계 가능  

### 🔹 성능 최적화
- `uvloop`와 `httptools`를 활용하면 성능이 크게 향상됨 (Uvicorn 기본 내장)  
- Keep-alive, connection timeout 등을 적절히 조정  
- Docker 환경에서는 `--workers`와 `--threads`를 컨테이너 리소스에 맞게 조정  

---

## 3. 권장 배포 패턴
- **Nginx + Gunicorn + Uvicorn**  
  - Nginx: SSL 처리, 정적 파일 서빙, 리버스 프록시  
  - Gunicorn: 프로세스 관리 및 워커 실행  
  - Uvicorn: ASGI 서버로 FastAPI 실행  

---

## 4. 위험 요소 및 주의사항
- **단일 Uvicorn 실행**은 운영 환경에서 안정성이 떨어짐 (프로세스 크래시 시 복구 불가).  
- **`--reload` 옵션**은 운영 환경에서 절대 사용하지 말아야 함 (성능 저하 및 보안 위험).  
- **SSL 직접 적용**은 가능하지만, 일반적으로 Nginx 같은 프록시 서버에서 처리하는 것이 더 안전하고 효율적임.  
- **워커 수 과다 설정**은 메모리 사용량 급증을 초래할 수 있으므로 서버 리소스에 맞게 조정해야 함.  

---

## 📑 정리
- 운영 환경에서는 **Gunicorn + UvicornWorker** 조합이 표준  
- 워커 수는 CPU 코어 기반으로 최적화  
- SSL은 프록시 서버(Nginx)에서 처리하는 것이 일반적  
- 로깅, 모니터링, 성능 튜닝을 반드시 포함해야 안정적인 서비스 운영 가능  
