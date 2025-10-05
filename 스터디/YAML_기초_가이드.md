---
layout: default
title: "YAML 기초 가이드"
description: "YAML 기초 가이드"
date: 2025-10-05
cache-control: no-cache
expires: 0
pragma: no-cache
author: "김명환"
---

## 📘 YAML 기초 가이드

### 🧾 YAML이란?

**YAML**은 "YAML Ain’t Markup Language"의 약자로, 사람이 읽기 쉬운 데이터 직렬화 포맷입니다.<br/>
주로 설정 파일, 데이터 교환, 구성 정의 등에 사용되며, JSON보다 간결하고 직관적인 문법을 제공합니다.

---

### 📐 기본 문법

#### 1. **키-값 구조**
```yaml
name: dev
age: 30
```

#### 2. **들여쓰기 (공백 기반)**
- 들여쓰기는 **공백(space)**으로만 합니다. **탭(tab)은 금지**입니다.
```yaml
person:
  name: dev
  age: 30
```

#### 3. **리스트**
```yaml
languages:
  - Python
  - JavaScript
  - Go
```

#### 4. **중첩 구조**
```yaml
server:
  host: localhost
  port: 8080
```

#### 5. **주석**
```yaml
# 이건 주석입니다
name: dev
```

#### 6. **문자열 처리**
```yaml
title: "Hello, YAML!"
description: >
  여러 줄의 문자열을
  한 줄로 이어서 처리합니다.
note: |
  여러 줄의 문자열을
  줄바꿈 포함 그대로 유지합니다.
```

---

### 🧪 예시: 웹 서버 설정

```yaml
webserver:
  host: 127.0.0.1
  port: 80
  ssl: true
  paths:
    - /home
    - /login
    - /dashboard
```

---

### ✅ YAML 사용처

| 분야 | 사용 예 |
|------|---------|
| DevOps | Kubernetes, Ansible 설정 |
| 머신러닝 | YOLO, PyTorch 모델 구성 |
| 웹 개발 | API 설정, 환경 변수 |
| 시스템 관리 | Ubuntu Netplan 네트워크 설정 |

---
