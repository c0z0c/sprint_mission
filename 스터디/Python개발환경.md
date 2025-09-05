---
layout: default
title: "Python 개발환경"
description: "Python 개발환경"
date: 2025-09-05
cache-control: no-cache
expires: 0
pragma: no-cache
author: "김명환"
---

# Python 개발환경 관리 가이드 (Windows & macOS)

## 1. 개요

### 1.1. 가상환경의 필요성

Python 개발에서 가상환경(Virtual Environment)은 프로젝트별로 독립적인 패키지 환경을 구성하여 의존성 충돌을 방지하는 핵심 도구입니다. 특히 딥러닝 프로젝트에서는 PyTorch, TensorFlow 등의 라이브러리가 서로 다른 버전 요구사항을 가질 수 있어 가상환경 관리가 필수적입니다.

**주요 장점:**
- 프로젝트별 독립적인 패키지 관리
- 의존성 충돌(Dependency Conflict) 방지
- 재현 가능한 개발 환경 구성
- 시스템 Python 환경의 안정성 보장

### 1.2. 주요 도구 비교

| 도구 | 특징 | 적용 범위 | 패키지 관리 |
|------|------|-----------|-------------|
| **virtualenv** | 경량화된 가상환경 | Python 패키지만 | pip |
| **conda** | 언어 독립적 환경 관리 | Python + 다른 언어, 바이너리 | conda + pip |
| **venv** | Python 내장 모듈 | Python 패키지만 | pip |

### 1.3. 운영체제별 차이점

운영체제마다 경로 구분자, 실행 파일 위치, 셸 환경이 다르므로 각각의 명령어를 숙지해야 합니다.

## 2. Virtualenv 환경 관리

### 2.1. 설치 및 기본 설정

#### 2.1.1. Windows 설치

```cmd
# pip를 통한 virtualenv 설치
pip install virtualenv

# 설치 확인
virtualenv --version
```

#### 2.1.2. macOS 설치

```bash
# pip를 통한 virtualenv 설치
pip install virtualenv

# Homebrew를 통한 설치 (선택사항)
brew install virtualenv

# 설치 확인
virtualenv --version
```

### 2.2. 가상환경 생성

**Windows:**
```cmd
# 기본 가상환경 생성
virtualenv venv

# Python 버전 지정하여 생성
virtualenv --python=python3.9 myproject_env

# 특정 경로에 생성
virtualenv C:\projects\myproject\venv
```

**macOS:**
```bash
# 기본 가상환경 생성
virtualenv venv

# Python 버전 지정하여 생성
virtualenv --python=python3.9 myproject_env

# 특정 경로에 생성
virtualenv ~/projects/myproject/venv
```

### 2.3. 환경 활성화/비활성화

#### 2.3.1. Windows 활성화

**Command Prompt:**
```cmd
# 활성화
venv\Scripts\activate

# 비활성화
deactivate
```

**PowerShell:**
```powershell
# 실행 정책 설정 (최초 1회)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 활성화
venv\Scripts\Activate.ps1

# 비활성화
deactivate
```

#### 2.3.2. macOS 활성화

```bash
# 활성화
source venv/bin/activate

# 비활성화
deactivate
```

### 2.4. 환경 삭제

**Windows & macOS:**
```bash
# 가상환경 폴더 직접 삭제
rm -rf venv          # macOS/Linux
rmdir /s venv        # Windows
```

## 3. Conda 환경 관리

### 3.1. Anaconda/Miniconda 설치

#### 3.1.1. Windows 설치 가이드

```cmd
# Miniconda 다운로드 후 설치
# https://docs.conda.io/en/latest/miniconda.html

# 설치 확인
conda --version

# conda 초기화 (Anaconda Prompt에서)
conda init
```

#### 3.1.2. macOS 설치 가이드

```bash
# Homebrew를 통한 설치
brew install --cask miniconda

# 또는 wget을 통한 설치
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh

# 셸 초기화
conda init zsh  # zsh 사용 시
conda init bash # bash 사용 시
```

### 3.2. 환경 생성 및 설정

**Windows & macOS 공통:**
```bash
# 기본 환경 생성
conda create --name myenv

# Python 버전 지정하여 생성
conda create --name deeplearning python=3.9

# 패키지와 함께 생성
conda create --name pytorch_env python=3.9 pytorch torchvision -c pytorch

# YAML 파일로 환경 생성
conda env create -f environment.yml
```

### 3.3. 환경 관리 명령어

**Windows & macOS 공통:**
```bash
# 환경 활성화
conda activate myenv

# 환경 비활성화
conda deactivate

# 환경 목록 확인
conda env list

# 환경 삭제
conda env remove --name myenv

# 환경 복제
conda create --name newenv --clone myenv
```

### 3.4. 채널 및 패키지 관리

**Windows & macOS 공통:**
```bash
# 채널 추가
conda config --add channels conda-forge
conda config --add channels pytorch

# 패키지 설치
conda install numpy pandas matplotlib

# 패키지 업데이트
conda update numpy

# 환경 내 패키지 목록
conda list

# 환경 export
conda env export > environment.yml
```

## 4. 라이브러리 의존성 관리

### 4.1. pip freeze를 통한 패키지 목록 추출

**Windows:**
```cmd
# 현재 환경의 모든 패키지 출력
pip freeze

# requirements.txt 파일 생성
pip freeze > requirements.txt

# 특정 패키지만 필터링
pip freeze | findstr "torch"
```

**macOS:**
```bash
# 현재 환경의 모든 패키지 출력
pip freeze

# requirements.txt 파일 생성
pip freeze > requirements.txt

# 특정 패키지만 필터링
pip freeze | grep torch
```

### 4.2. requirements.txt 생성 및 활용

**파일 구조 예시:**
```txt
# 기본 requirements.txt
torch==1.12.0
torchvision==0.13.0
numpy>=1.21.0
pandas==1.5.2
matplotlib>=3.5.0
jupyter

# 개발용 requirements-dev.txt
-r requirements.txt
pytest>=7.0.0
black==22.10.0
flake8>=5.0.0
```

**버전 지정 방법:**
```txt
# 정확한 버전 지정
torch==1.12.0

# 최소 버전 지정
numpy>=1.21.0

# 버전 범위 지정
pandas>=1.4.0,<2.0.0

# 호환 버전 지정 (틸드)
scipy~=1.9.0  # 1.9.x 버전 중 최신
```

### 4.3. pip install -r을 통한 일괄 설치

**Windows & macOS 공통:**
```bash
# requirements.txt 일괄 설치
pip install -r requirements.txt

# 개발용 패키지 포함 설치
pip install -r requirements-dev.txt

# 업그레이드와 함께 설치
pip install -r requirements.txt --upgrade

# 캐시 없이 설치
pip install -r requirements.txt --no-cache-dir
```

### 4.4. 버전 관리 모범 사례

**프로덕션 환경 고려사항:**
- 정확한 버전 지정 (`==`) 사용으로 재현성 보장
- 주요 라이브러리는 범위 지정으로 보안 업데이트 허용
- 개발/테스트/프로덕션 환경별 requirements 분리

```txt
# requirements/base.txt - 공통 의존성
numpy==1.24.1
pandas==1.5.2

# requirements/production.txt - 프로덕션용
-r base.txt
gunicorn==20.1.0

# requirements/development.txt - 개발용
-r base.txt
pytest==7.2.0
jupyter==1.0.0
```

## 5. 실무 워크플로우

### 5.1. 프로젝트 시작 시 환경 설정

#### Virtualenv 워크플로우

**Windows:**
```cmd
# 1. 프로젝트 디렉터리 생성
mkdir my_deeplearning_project
cd my_deeplearning_project

# 2. 가상환경 생성
virtualenv venv

# 3. 환경 활성화
venv\Scripts\activate

# 4. 필요한 패키지 설치
pip install torch torchvision numpy pandas matplotlib jupyter

# 5. requirements.txt 생성
pip freeze > requirements.txt
```

**macOS:**
```bash
# 1. 프로젝트 디렉터리 생성
mkdir my_deeplearning_project
cd my_deeplearning_project

# 2. 가상환경 생성
virtualenv venv

# 3. 환경 활성화
source venv/bin/activate

# 4. 필요한 패키지 설치
pip install torch torchvision numpy pandas matplotlib jupyter

# 5. requirements.txt 생성
pip freeze > requirements.txt
```

#### Conda 워크플로우

**Windows & macOS 공통:**
```bash
# 1. 프로젝트별 환경 생성
conda create --name my_project python=3.9

# 2. 환경 활성화
conda activate my_project

# 3. 주요 패키지 설치
conda install pytorch torchvision -c pytorch
conda install numpy pandas matplotlib jupyter -c conda-forge

# 4. 환경 정보 export
conda env export > environment.yml
```

### 5.2. 협업을 위한 환경 공유

#### Git과 함께 사용하기

**.gitignore 설정:**
```gitignore
# 가상환경 폴더 제외
venv/
env/
.env

# Conda 환경 제외
.conda/

# IDE 설정 파일
.vscode/
.idea/

# 캐시 파일
__pycache__/
*.pyc
.pytest_cache/
```

**README.md에 환경 설정 가이드 포함:**
```markdown
## 환경 설정

### Virtualenv 사용 시
```bash
virtualenv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Conda 사용 시
```bash
conda env create -f environment.yml
conda activate project_name
```
```

### 5.3. 환경 복원 및 재생산

#### 다른 머신에서 환경 복원

**Virtualenv 환경 복원:**
```bash
# 1. 저장소 클론
git clone <repository-url>
cd project-directory

# 2. 가상환경 생성 및 활성화
virtualenv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 의존성 설치
pip install -r requirements.txt
```

**Conda 환경 복원:**
```bash
# 1. 저장소 클론 후 이동
cd project-directory

# 2. environment.yml로 환경 생성
conda env create -f environment.yml

# 3. 환경 활성화
conda activate project_name
```

## 6. 운영체제별 주요 차이점 비교

| 항목 | Windows | macOS |
|------|---------|-------|
| **활성화 스크립트** | `Scripts\activate` | `bin/activate` |
| **경로 구분자** | `\` (백슬래시) | `/` (슬래시) |
| **기본 셸** | Command Prompt/PowerShell | Terminal/Bash/Zsh |
| **실행 파일 확장자** | `.bat`, `.exe` | 확장자 없음 |
| **Python 실행파일** | `python.exe` | `python3` |
| **PowerShell 실행 정책** | 설정 필요 | 해당 없음 |
| **환경변수 설정** | `set VARIABLE=value` | `export VARIABLE=value` |
| **패키지 설치 경로** | `Lib\site-packages` | `lib/python3.x/site-packages` |

### 6.1. PowerShell 실행 정책 (Windows 전용)

Windows PowerShell에서 가상환경을 사용할 때는 실행 정책 설정이 필요합니다:

```powershell
# 현재 실행 정책 확인
Get-ExecutionPolicy

# 실행 정책 변경 (관리자 권한 불필요)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 가상환경 활성화
.\venv\Scripts\Activate.ps1
```

### 6.2. macOS Zsh 설정

macOS Catalina 이후 기본 셸이 Zsh로 변경되어 conda 초기화 시 주의가 필요합니다:

```bash
# Zsh 사용자의 경우
conda init zsh

# 설정 파일 새로고침
source ~/.zshrc

# Bash 사용자의 경우
conda init bash
source ~/.bashrc
```

---

- 코렙 25.08.27
    - [requirements_colab_250827_3_12.txt](https://github.com/c0z0c/sprint_mission/blob/master/%EC%8A%A4%ED%84%B0%EB%94%94/requirements_colab_250827_3_12.txt)

---

## 용어 목록

- **Virtual Environment**: 프로젝트별로 독립적인 Python 패키지 환경을 제공하는 도구
- **Dependency Conflict**: 서로 다른 패키지들이 같은 라이브러리의 호환되지 않는 버전을 요구할 때 발생하는 충돌
- **Requirements.txt**: pip에서 사용하는 패키지 의존성 목록 파일
- **Environment.yml**: Conda에서 사용하는 환경 설정 및 패키지 목록 파일
- **Activate/Deactivate**: 가상환경을 활성화하거나 비활성화하는 명령
- **Pip freeze**: 현재 환경에 설치된 모든 패키지와 버전 정보를 출력하는 명령
- **Conda channel**: Conda 패키지를 배포하는 저장소
- **Site-packages**: Python 패키지가 설치되는 디렉터리
- **Execution Policy**: Windows PowerShell에서 스크립트 실행을 제어하는 보안 설정
- **Shell initialization**: 셸 시작 시 conda 명령을 사용할 수 있도록 설정하는 과정