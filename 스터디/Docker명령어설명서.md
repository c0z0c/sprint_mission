---
layout: default
title: "Docker 명령어 설명서"
description: "Docker 명령어 설명서"
date: 2025-12-10
cache-control: no-cache
expires: 0
pragma: no-cache
author: "김명환"
---

# Docker 명령어 설명서

## 1. 기본 명령어 (Basic Commands)

### 1.1. 이미지 관리 (Image Management)

#### 1.1.1. docker pull

원격 레지스트리 (registry)에서 이미지를 다운로드합니다.

```bash
docker pull [OPTIONS] NAME[:TAG|@DIGEST]
```

**주요 옵션 (Options)**

- `-a, --all-tags`: 모든 태그 (tag)된 이미지를 다운로드
- `--platform`: 특정 플랫폼 (platform) 이미지 지정 (예: linux/amd64, linux/arm64)
- `-q, --quiet`: 출력 최소화

**사용 예시**

```bash
# 최신 버전 다운로드
docker pull ubuntu:latest

# 특정 버전 다운로드
docker pull python:3.9

# 특정 플랫폼 이미지 다운로드
docker pull --platform linux/amd64 mysql:8.0
```

#### 1.1.2. docker images

로컬에 저장된 이미지 목록을 조회합니다.

```bash
docker images [OPTIONS] [REPOSITORY[:TAG]]
```

**주요 옵션**

- `-a, --all`: 중간 이미지 (intermediate images)를 포함한 모든 이미지 표시
- `-q, --quiet`: 이미지 ID만 표시
- `--filter`: 조건에 따른 필터링 (예: `dangling=true`)
- `--format`: 출력 형식 지정

**사용 예시**

```bash
# 모든 이미지 조회
docker images

# 특정 이미지만 조회
docker images ubuntu

# 사용하지 않는 이미지 조회
docker images --filter "dangling=true"

# 이미지 ID만 출력
docker images -q
```

#### 1.1.3. docker rmi

로컬 이미지를 삭제합니다.

```bash
docker rmi [OPTIONS] IMAGE [IMAGE...]
```

**주요 옵션**

- `-f, --force`: 강제 삭제
- `--no-prune`: 태그되지 않은 부모 이미지 (parent images) 삭제 방지

**사용 예시**

```bash
# 이미지 삭제
docker rmi ubuntu:latest

# 여러 이미지 동시 삭제
docker rmi image1 image2 image3

# 사용하지 않는 모든 이미지 삭제
docker rmi $(docker images -f "dangling=true" -q)

# 강제 삭제
docker rmi -f nginx:latest
```

#### 1.1.4. docker build

Dockerfile로부터 이미지를 빌드 (build)합니다.

```bash
docker build [OPTIONS] PATH | URL | -
```

**주요 옵션**

- `-t, --tag`: 이미지 이름과 태그 지정
- `-f, --file`: Dockerfile 경로 지정 (기본값: PATH/Dockerfile)
- `--build-arg`: 빌드 시 변수 전달
- `--no-cache`: 캐시 (cache)를 사용하지 않고 빌드
- `--target`: 멀티스테이지 빌드 (multi-stage build)에서 특정 스테이지 지정

**사용 예시**

```bash
# 현재 디렉토리의 Dockerfile로 빌드
docker build -t myapp:1.0 .

# 특정 Dockerfile 지정
docker build -f Dockerfile.prod -t myapp:prod .

# 빌드 인자 전달
docker build --build-arg VERSION=1.0 -t myapp:1.0 .

# 캐시 없이 빌드
docker build --no-cache -t myapp:latest .

# 멀티스테이지에서 특정 스테이지만 빌드
docker build --target builder -t myapp:builder .
```

### 1.2. 컨테이너 관리 (Container Management)

#### 1.2.1. docker run

새로운 컨테이너 (container)를 생성하고 실행합니다.

```bash
docker run [OPTIONS] IMAGE [COMMAND] [ARG...]
```

**주요 옵션**

- `-d, --detach`: 백그라운드 (background) 모드로 실행
- `-p, --publish`: 포트 매핑 (port mapping) (호스트:컨테이너)
- `-v, --volume`: 볼륨 (volume) 마운트 (mount)
- `-e, --env`: 환경변수 (environment variable) 설정
- `--name`: 컨테이너 이름 지정
- `-it`: 대화형 터미널 (interactive terminal) 모드
- `--rm`: 컨테이너 종료 시 자동 삭제
- `--network`: 네트워크 (network) 지정
- `--restart`: 재시작 정책 (restart policy) 설정

**사용 예시**

```bash
# 기본 실행
docker run ubuntu

# 백그라운드에서 실행
docker run -d nginx

# 포트 매핑과 이름 지정
docker run -d -p 8080:80 --name webserver nginx

# 볼륨 마운트
docker run -d -v /host/path:/container/path nginx

# 환경변수 설정
docker run -e MYSQL_ROOT_PASSWORD=secret -d mysql

# 대화형 터미널
docker run -it ubuntu /bin/bash

# 종료 시 자동 삭제
docker run --rm ubuntu echo "Hello World"

# 재시작 정책 설정
docker run -d --restart unless-stopped nginx
```

#### 1.2.2. docker ps

실행 중인 컨테이너 목록을 조회합니다.

```bash
docker ps [OPTIONS]
```

**주요 옵션**

- `-a, --all`: 중지된 컨테이너 (stopped containers)를 포함한 모든 컨테이너 표시
- `-q, --quiet`: 컨테이너 ID만 표시
- `-f, --filter`: 조건에 따른 필터링
- `--format`: 출력 형식 지정
- `-l, --latest`: 가장 최근에 생성된 컨테이너 표시

**사용 예시**

```bash
# 실행 중인 컨테이너 조회
docker ps

# 모든 컨테이너 조회
docker ps -a

# 컨테이너 ID만 출력
docker ps -q

# 특정 조건으로 필터링
docker ps --filter "status=exited"
docker ps --filter "name=web"
```

#### 1.2.3. docker stop

실행 중인 컨테이너를 중지합니다.

```bash
docker stop [OPTIONS] CONTAINER [CONTAINER...]
```

**주요 옵션**

- `-t, --time`: 강제 종료 (SIGKILL) 전 대기 시간 (기본값: 10초)

**사용 예시**

```bash
# 컨테이너 중지
docker stop webserver

# 여러 컨테이너 동시 중지
docker stop container1 container2

# 모든 실행 중인 컨테이너 중지
docker stop $(docker ps -q)

# 대기 시간 지정
docker stop -t 30 webserver
```

#### 1.2.4. docker start

중지된 컨테이너를 시작합니다.

```bash
docker start [OPTIONS] CONTAINER [CONTAINER...]
```

**주요 옵션**

- `-a, --attach`: 표준 출력 (stdout)/표준 에러 (stderr)를 연결
- `-i, --interactive`: 표준 입력 (stdin)을 연결

**사용 예시**

```bash
# 컨테이너 시작
docker start webserver

# 여러 컨테이너 동시 시작
docker start container1 container2

# 시작하면서 출력 확인
docker start -a webserver
```

#### 1.2.5. docker restart

컨테이너를 재시작합니다.

```bash
docker restart [OPTIONS] CONTAINER [CONTAINER...]
```

**주요 옵션**

- `-t, --time`: 강제 종료 전 대기 시간

**사용 예시**

```bash
# 컨테이너 재시작
docker restart webserver

# 대기 시간 지정
docker restart -t 30 webserver
```

#### 1.2.6. docker rm

컨테이너를 삭제합니다.

```bash
docker rm [OPTIONS] CONTAINER [CONTAINER...]
```

**주요 옵션**

- `-f, --force`: 실행 중인 컨테이너 강제 삭제
- `-v, --volumes`: 연결된 익명 볼륨 (anonymous volumes)도 함께 삭제

**사용 예시**

```bash
# 컨테이너 삭제
docker rm webserver

# 강제 삭제
docker rm -f webserver

# 중지된 모든 컨테이너 삭제
docker rm $(docker ps -aq -f status=exited)

# 볼륨과 함께 삭제
docker rm -v webserver
```

#### 1.2.7. docker exec

실행 중인 컨테이너 내부에서 명령어를 실행합니다.

```bash
docker exec [OPTIONS] CONTAINER COMMAND [ARG...]
```

**주요 옵션**

- `-i, --interactive`: 표준 입력 연결
- `-t, --tty`: 가상 터미널 (pseudo-TTY) 할당
- `-d, --detach`: 백그라운드에서 실행
- `-e, --env`: 환경변수 설정
- `-w, --workdir`: 작업 디렉토리 (working directory) 지정
- `-u, --user`: 사용자 (user) 지정

**사용 예시**

```bash
# 대화형 쉘 실행
docker exec -it webserver /bin/bash

# 단일 명령어 실행
docker exec webserver ls /app

# 백그라운드에서 명령어 실행
docker exec -d webserver /app/script.sh

# 특정 사용자로 명령어 실행
docker exec -u root webserver apt-get update
```

#### 1.2.8. docker logs

컨테이너의 로그 (logs)를 조회합니다.

```bash
docker logs [OPTIONS] CONTAINER
```

**주요 옵션**

- `-f, --follow`: 실시간 로그 스트림 (stream) 출력
- `--tail`: 마지막 N줄만 출력
- `-t, --timestamps`: 타임스탬프 (timestamp) 표시
- `--since`: 특정 시간 이후의 로그만 출력
- `--until`: 특정 시간 이전의 로그만 출력

**사용 예시**

```bash
# 전체 로그 조회
docker logs webserver

# 실시간 로그 확인
docker logs -f webserver

# 마지막 100줄만 조회
docker logs --tail 100 webserver

# 타임스탬프와 함께 조회
docker logs -t webserver

# 특정 시간 이후 로그
docker logs --since 2025-12-10T10:00:00 webserver
docker logs --since 30m webserver
```

### 1.3. 정보 조회 (Information Retrieval)

#### 1.3.1. docker inspect

Docker 객체 (object)의 상세 정보를 JSON 형식으로 조회합니다.

```bash
docker inspect [OPTIONS] NAME|ID [NAME|ID...]
```

**주요 옵션**

- `-f, --format`: Go 템플릿 (template)을 사용하여 출력 형식 지정
- `--type`: 객체 타입 지정 (container, image, volume, network)

**사용 예시**

```bash
# 컨테이너 상세 정보
docker inspect webserver

# 특정 필드만 조회
docker inspect -f '{{.State.Status}}' webserver
docker inspect -f '{{.NetworkSettings.IPAddress}}' webserver

# 이미지 정보 조회
docker inspect nginx:latest
```

#### 1.3.2. docker stats

컨테이너의 리소스 (resource) 사용량을 실시간으로 모니터링합니다.

```bash
docker stats [OPTIONS] [CONTAINER...]
```

**주요 옵션**

- `-a, --all`: 모든 컨테이너 표시
- `--no-stream`: 실시간 스트림 없이 한 번만 출력
- `--format`: 출력 형식 지정

**사용 예시**

```bash
# 모든 실행 중인 컨테이너 모니터링
docker stats

# 특정 컨테이너만 모니터링
docker stats webserver db

# 한 번만 출력
docker stats --no-stream
```

## 2. 추가 명령어 (Advanced Commands)

### 2.1. 네트워크 관리 (Network Management)

#### 2.1.1. docker network create

사용자 정의 네트워크를 생성합니다.

```bash
docker network create [OPTIONS] NETWORK
```

**주요 옵션**

- `-d, --driver`: 네트워크 드라이버 (driver) 지정 (bridge, overlay, host 등)
- `--subnet`: 서브넷 (subnet) CIDR 형식으로 지정
- `--gateway`: 게이트웨이 (gateway) 주소 지정
- `--ip-range`: 컨테이너에 할당할 IP 범위 지정

**사용 예시**

```bash
# 기본 브리지 네트워크 생성
docker network create mynetwork

# 서브넷 지정
docker network create --subnet=172.20.0.0/16 --gateway=172.20.0.1 mynetwork

# overlay 네트워크 생성 (Swarm 모드)
docker network create -d overlay myoverlay
```

#### 2.1.2. docker network ls

네트워크 목록을 조회합니다.

```bash
docker network ls [OPTIONS]
```

**사용 예시**

```bash
# 모든 네트워크 조회
docker network ls

# 특정 드라이버 필터링
docker network ls --filter driver=bridge
```

#### 2.1.3. docker network connect

실행 중인 컨테이너를 네트워크에 연결합니다.

```bash
docker network connect [OPTIONS] NETWORK CONTAINER
```

**주요 옵션**

- `--ip`: 컨테이너에 할당할 특정 IP 주소 지정
- `--alias`: 네트워크 별칭 (alias) 지정

**사용 예시**

```bash
# 컨테이너를 네트워크에 연결
docker network connect mynetwork webserver

# 특정 IP로 연결
docker network connect --ip 172.20.0.10 mynetwork webserver
```

#### 2.1.4. docker network disconnect

컨테이너를 네트워크에서 분리합니다.

```bash
docker network disconnect [OPTIONS] NETWORK CONTAINER
```

**사용 예시**

```bash
# 네트워크에서 분리
docker network disconnect mynetwork webserver

# 강제 분리
docker network disconnect -f mynetwork webserver
```

#### 2.1.5. docker network rm

네트워크를 삭제합니다.

```bash
docker network rm NETWORK [NETWORK...]
```

**사용 예시**

```bash
# 네트워크 삭제
docker network rm mynetwork

# 사용하지 않는 모든 네트워크 삭제
docker network prune
```

### 2.2. 볼륨 관리 (Volume Management)

#### 2.2.1. docker volume create

볼륨을 생성합니다.

```bash
docker volume create [OPTIONS] [VOLUME]
```

**주요 옵션**

- `-d, --driver`: 볼륨 드라이버 지정 (기본값: local)
- `--label`: 메타데이터 (metadata) 레이블 추가
- `-o, --opt`: 드라이버 옵션 설정

**사용 예시**

```bash
# 기본 볼륨 생성
docker volume create myvolume

# 레이블과 함께 생성
docker volume create --label env=production myvolume

# NFS 볼륨 생성
docker volume create --driver local \
  --opt type=nfs \
  --opt o=addr=192.168.1.1,rw \
  --opt device=:/path/to/dir \
  nfsvolume
```

#### 2.2.2. docker volume ls

볼륨 목록을 조회합니다.

```bash
docker volume ls [OPTIONS]
```

**주요 옵션**

- `-f, --filter`: 조건에 따른 필터링
- `-q, --quiet`: 볼륨 이름만 표시

**사용 예시**

```bash
# 모든 볼륨 조회
docker volume ls

# dangling 볼륨 조회
docker volume ls -f dangling=true
```

#### 2.2.3. docker volume inspect

볼륨의 상세 정보를 조회합니다.

```bash
docker volume inspect [OPTIONS] VOLUME [VOLUME...]
```

**사용 예시**

```bash
# 볼륨 정보 조회
docker volume inspect myvolume

# 특정 필드만 조회
docker volume inspect -f '{{.Mountpoint}}' myvolume
```

#### 2.2.4. docker volume rm

볼륨을 삭제합니다.

```bash
docker volume rm [OPTIONS] VOLUME [VOLUME...]
```

**사용 예시**

```bash
# 볼륨 삭제
docker volume rm myvolume

# 여러 볼륨 삭제
docker volume rm volume1 volume2

# 사용하지 않는 모든 볼륨 삭제
docker volume prune
```

### 2.3. 이미지 관리 고급 (Advanced Image Management)

#### 2.3.1. docker tag

이미지에 새로운 태그를 지정합니다.

```bash
docker tag SOURCE_IMAGE[:TAG] TARGET_IMAGE[:TAG]
```

**사용 예시**

```bash
# 태그 추가
docker tag myapp:1.0 myapp:latest

# 레지스트리 경로를 포함한 태그
docker tag myapp:1.0 myregistry.com/myapp:1.0
```

#### 2.3.2. docker push

이미지를 레지스트리로 업로드 (upload)합니다.

```bash
docker push [OPTIONS] NAME[:TAG]
```

**주요 옵션**

- `-a, --all-tags`: 모든 태그를 업로드
- `--disable-content-trust`: 이미지 서명 (signing) 비활성화

**사용 예시**

```bash
# 이미지 푸시
docker push myregistry.com/myapp:1.0

# 모든 태그 푸시
docker push -a myregistry.com/myapp
```

**Docker Hub 푸시 실전 예시**

Docker Hub에 이미지를 푸시할 때는 전체 레지스트리 주소를 명시해야 합니다.

```bash
# 백엔드 이미지 태깅
docker tag c0z0c/mis18_backend:latest index.docker.io/c0z0c/mis18_backend:v1.1

# 백엔드 푸시
docker push index.docker.io/c0z0c/mis18_backend:v1.1

# 프론트엔드 이미지 태깅
docker tag c0z0c/mis18_frontend:latest index.docker.io/c0z0c/mis18_frontend:v1.1

# 프론트엔드 푸시
docker push index.docker.io/c0z0c/mis18_frontend:v1.1
```

**참고**: `index.docker.io`는 Docker Hub의 공식 레지스트리 주소입니다. 이 주소를 명시하면 클라우드 플랫폼에서 이미지를 정확하게 참조할 수 있습니다.

#### 2.3.3. docker save

이미지를 tar 아카이브 (archive)로 저장합니다.

```bash
docker save [OPTIONS] IMAGE [IMAGE...]
```

**주요 옵션**

- `-o, --output`: 출력 파일 지정

**사용 예시**

```bash
# 이미지를 파일로 저장
docker save -o myapp.tar myapp:1.0

# 여러 이미지를 하나의 파일로 저장
docker save -o images.tar myapp:1.0 nginx:latest

# 파이프를 사용한 저장
docker save myapp:1.0 | gzip > myapp.tar.gz
```

#### 2.3.4. docker load

tar 아카이브에서 이미지를 로드 (load)합니다.

```bash
docker load [OPTIONS]
```

**주요 옵션**

- `-i, --input`: 입력 파일 지정
- `-q, --quiet`: 출력 최소화

**사용 예시**

```bash
# 파일에서 이미지 로드
docker load -i myapp.tar

# 파이프를 사용한 로드
gunzip -c myapp.tar.gz | docker load
```

#### 2.3.5. docker export

컨테이너의 파일시스템 (filesystem)을 tar 아카이브로 내보내기 (export)합니다.

```bash
docker export [OPTIONS] CONTAINER
```

**주요 옵션**

- `-o, --output`: 출력 파일 지정

**사용 예시**

```bash
# 컨테이너를 파일로 내보내기
docker export -o container.tar webserver

# 파이프를 사용한 내보내기
docker export webserver | gzip > container.tar.gz
```

#### 2.3.6. docker import

tar 아카이브에서 이미지를 생성합니다.

```bash
docker import [OPTIONS] file|URL|- [REPOSITORY[:TAG]]
```

**주요 옵션**

- `-c, --change`: 생성될 이미지에 Dockerfile 명령어 적용
- `-m, --message`: 커밋 메시지 (commit message) 설정

**사용 예시**

```bash
# 파일에서 이미지 생성
docker import container.tar myapp:imported

# 파이프를 사용한 가져오기
cat container.tar.gz | docker import - myapp:imported

# Dockerfile 명령어와 함께 가져오기
docker import -c "CMD /start.sh" container.tar myapp:imported
```

### 2.4. 시스템 관리 (System Management)

#### 2.4.1. docker system df

Docker가 사용 중인 디스크 공간 (disk space)을 확인합니다.

```bash
docker system df [OPTIONS]
```

**주요 옵션**

- `-v, --verbose`: 상세 정보 표시

**사용 예시**

```bash
# 디스크 사용량 요약
docker system df

# 상세 정보
docker system df -v
```

#### 2.4.2. docker system prune

사용하지 않는 Docker 객체를 정리 (cleanup)합니다.

```bash
docker system prune [OPTIONS]
```

**주요 옵션**

- `-a, --all`: 사용하지 않는 이미지도 모두 삭제
- `-f, --force`: 확인 없이 삭제
- `--volumes`: 볼륨도 함께 삭제

**사용 예시**

```bash
# 기본 정리 (중지된 컨테이너, 사용하지 않는 네트워크, dangling 이미지)
docker system prune

# 모든 사용하지 않는 이미지 포함
docker system prune -a

# 볼륨 포함 정리
docker system prune --volumes

# 확인 없이 정리
docker system prune -f
```

#### 2.4.3. docker version

Docker 클라이언트 (client)와 서버 (server)의 버전 정보를 표시합니다.

```bash
docker version [OPTIONS]
```

**주요 옵션**

- `-f, --format`: 출력 형식 지정

**사용 예시**

```bash
# 버전 정보 조회
docker version

# JSON 형식으로 출력
docker version --format '{{json .}}'
```

#### 2.4.4. docker info

Docker 시스템 전반의 정보를 표시합니다.

```bash
docker info [OPTIONS]
```

**주요 옵션**

- `-f, --format`: 출력 형식 지정

**사용 예시**

```bash
# 시스템 정보 조회
docker info

# 특정 필드만 조회
docker info --format '{{.ServerVersion}}'
```

### 2.5. 컨테이너 관리 고급 (Advanced Container Management)

#### 2.5.1. docker cp

컨테이너와 호스트 간 파일/디렉토리를 복사합니다.

```bash
docker cp [OPTIONS] CONTAINER:SRC_PATH DEST_PATH
docker cp [OPTIONS] SRC_PATH CONTAINER:DEST_PATH
```

**주요 옵션**

- `-a, --archive`: 아카이브 모드 (소유권과 권한 보존)
- `-L, --follow-link`: 심볼릭 링크 (symbolic link)를 따라감

**사용 예시**

```bash
# 컨테이너에서 호스트로 복사
docker cp webserver:/app/logs/. ./logs

# 호스트에서 컨테이너로 복사
docker cp ./config.json webserver:/app/config.json

# 권한을 보존하며 복사
docker cp -a webserver:/app/backup ./backup
```

#### 2.5.2. docker commit

컨테이너의 변경사항 (changes)으로부터 새로운 이미지를 생성합니다.

```bash
docker commit [OPTIONS] CONTAINER [REPOSITORY[:TAG]]
```

**주요 옵션**

- `-a, --author`: 작성자 정보 설정
- `-m, --message`: 커밋 메시지 설정
- `-c, --change`: 생성될 이미지에 Dockerfile 명령어 적용
- `-p, --pause`: 커밋 중 컨테이너 일시정지 (기본값: true)

**사용 예시**

```bash
# 컨테이너를 이미지로 커밋
docker commit webserver myapp:snapshot

# 메시지와 작성자 정보 포함
docker commit -m "Added new features" -a "John Doe" webserver myapp:v2

# Dockerfile 명령어 적용
docker commit -c "ENV DEBUG=true" webserver myapp:debug
```

#### 2.5.3. docker top

컨테이너 내부에서 실행 중인 프로세스 (processes)를 조회합니다.

```bash
docker top CONTAINER [ps OPTIONS]
```

**사용 예시**

```bash
# 실행 중인 프로세스 조회
docker top webserver

# ps 옵션 사용
docker top webserver aux
```

#### 2.5.4. docker pause / docker unpause

컨테이너의 모든 프로세스를 일시정지하거나 재개합니다.

```bash
docker pause CONTAINER [CONTAINER...]
docker unpause CONTAINER [CONTAINER...]
```

**사용 예시**

```bash
# 컨테이너 일시정지
docker pause webserver

# 컨테이너 재개
docker unpause webserver
```

#### 2.5.5. docker rename

컨테이너의 이름을 변경합니다.

```bash
docker rename CONTAINER NEW_NAME
```

**사용 예시**

```bash
# 컨테이너 이름 변경
docker rename old_name new_name
```

#### 2.5.6. docker wait

컨테이너가 중지될 때까지 대기하고 종료 코드 (exit code)를 반환합니다.

```bash
docker wait CONTAINER [CONTAINER...]
```

**사용 예시**

```bash
# 컨테이너 종료 대기
docker wait webserver

# 종료 코드를 변수에 저장
EXIT_CODE=$(docker wait webserver)
echo "Container exited with code: $EXIT_CODE"
```

#### 2.5.7. docker attach

실행 중인 컨테이너의 표준 입력/출력/에러에 연결합니다.

```bash
docker attach [OPTIONS] CONTAINER
```

**주요 옵션**

- `--no-stdin`: 표준 입력을 연결하지 않음
- `--sig-proxy`: 시그널 (signal)을 프록시 (proxy) (기본값: true)

**사용 예시**

```bash
# 컨테이너에 연결
docker attach webserver

# 연결 해제: Ctrl+P, Ctrl+Q (컨테이너는 계속 실행)
# 컨테이너 종료: Ctrl+C
```

#### 2.5.8. docker diff

컨테이너의 파일시스템 변경사항을 검사합니다.

```bash
docker diff CONTAINER
```

**출력 형식**

- A: 파일 또는 디렉토리 추가 (Added)
- D: 파일 또는 디렉토리 삭제 (Deleted)
- C: 파일 또는 디렉토리 변경 (Changed)

**사용 예시**

```bash
# 파일시스템 변경사항 확인
docker diff webserver
```

### 2.6. 레지스트리 관리 (Registry Management)

#### 2.6.1. docker login

Docker 레지스트리에 로그인합니다.

```bash
docker login [OPTIONS] [SERVER]
```

**주요 옵션**

- `-u, --username`: 사용자명
- `-p, --password`: 비밀번호 (보안상 권장하지 않음)
- `--password-stdin`: 표준 입력에서 비밀번호 읽기

**사용 예시**

```bash
# Docker Hub 로그인 (대화형)
docker login

# 특정 레지스트리 로그인
docker login myregistry.com

# 표준 입력으로 비밀번호 전달 (보안)
echo "$PASSWORD" | docker login -u myuser --password-stdin myregistry.com
```

#### 2.6.2. docker logout

Docker 레지스트리에서 로그아웃합니다.

```bash
docker logout [SERVER]
```

**사용 예시**

```bash
# Docker Hub 로그아웃
docker logout

# 특정 레지스트리 로그아웃
docker logout myregistry.com
```

#### 2.6.3. docker search

Docker Hub에서 이미지를 검색합니다.

```bash
docker search [OPTIONS] TERM
```

**주요 옵션**

- `--filter`: 필터 조건 (stars, is-official, is-automated)
- `--limit`: 최대 검색 결과 수 (기본값: 25)
- `--no-trunc`: 출력을 잘라내지 않음

**사용 예시**

```bash
# 이미지 검색
docker search ubuntu

# 별 개수 필터링
docker search --filter stars=100 ubuntu

# 공식 이미지만 검색
docker search --filter is-official=true nginx

# 검색 결과 제한
docker search --limit 10 python
```

## 3. Dockerfile

### 3.1. Dockerfile 개요

Dockerfile은 Docker 이미지를 빌드하기 위한 명령어 (instructions)를 포함하는 텍스트 파일입니다. 각 명령어는 이미지에 레이어 (layer)를 추가합니다.

### 3.2. Dockerfile 구조

```mermaid
graph TD
    A["FROM: 베이스 이미지"]
    B["RUN: 명령어 실행"]
    C["COPY/ADD: 파일 복사"]
    D["WORKDIR: 작업 디렉토리"]
    E["ENV: 환경변수"]
    F["EXPOSE: 포트 노출"]
    G["CMD/ENTRYPOINT: 실행 명령"]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
```

### 3.3. Dockerfile 명령어

#### 3.3.1. FROM

베이스 이미지 (base image)를 지정합니다. 모든 Dockerfile은 FROM으로 시작해야 합니다.

```dockerfile
FROM [--platform=<platform>] <image>[:<tag>] [AS <name>]
```

**사용 예시**

```dockerfile
# 기본 사용
FROM ubuntu:22.04

# 멀티스테이지 빌드
FROM golang:1.21 AS builder
FROM alpine:3.18 AS runtime

# 특정 플랫폼 지정
FROM --platform=linux/amd64 python:3.11
```

**특징**

- ARG를 제외한 모든 명령어보다 먼저 와야 함
- 멀티스테이지 빌드에서 여러 번 사용 가능
- AS 키워드로 스테이지 이름 지정 가능

#### 3.3.2. RUN

이미지 빌드 시 명령어를 실행합니다. 새로운 레이어를 생성합니다.

```dockerfile
# 쉘 형식 (shell form)
RUN <command>

# exec 형식 (exec form)
RUN ["executable", "param1", "param2"]
```

**사용 예시**

```dockerfile
# 쉘 형식 (기본 쉘: /bin/sh -c)
RUN apt-get update && apt-get install -y \
    package1 \
    package2 \
    && rm -rf /var/lib/apt/lists/*

# exec 형식
RUN ["/bin/bash", "-c", "echo hello"]

# 여러 명령어를 && 로 연결하여 레이어 최소화
RUN apt-get update && \
    apt-get install -y curl && \
    curl -o /tmp/file.tar.gz https://example.com/file.tar.gz && \
    tar -xzf /tmp/file.tar.gz -C /opt && \
    rm /tmp/file.tar.gz
```

**모범 사례 (Best Practices)**

- 레이어 수를 줄이기 위해 관련 명령어를 && 로 연결
- apt-get update와 install을 동일한 RUN에서 실행
- 불필요한 파일은 동일 레이어에서 삭제
- 긴 명령어는 백슬래시 (\\) 로 줄바꿈

#### 3.3.3. CMD

컨테이너 시작 시 실행할 기본 명령어를 지정합니다. Dockerfile에서 한 번만 사용 가능합니다.

```dockerfile
# exec 형식 (권장)
CMD ["executable","param1","param2"]

# 쉘 형식
CMD command param1 param2

# ENTRYPOINT의 기본 파라미터
CMD ["param1","param2"]
```

**사용 예시**

```dockerfile
# exec 형식 (권장)
CMD ["python", "app.py"]
CMD ["nginx", "-g", "daemon off;"]

# 쉘 형식
CMD python app.py

# ENTRYPOINT와 함께 사용
ENTRYPOINT ["python"]
CMD ["app.py"]
```

**특징**

- docker run 시 명령어를 지정하면 CMD는 무시됨
- 여러 개 정의 시 마지막 CMD만 적용
- ENTRYPOINT와 함께 사용 시 기본 인자 역할

#### 3.3.4. ENTRYPOINT

컨테이너 시작 시 항상 실행될 명령어를 지정합니다.

```dockerfile
# exec 형식 (권장)
ENTRYPOINT ["executable", "param1", "param2"]

# 쉘 형식
ENTRYPOINT command param1 param2
```

**사용 예시**

```dockerfile
# 기본 사용
ENTRYPOINT ["python", "app.py"]

# CMD와 조합
ENTRYPOINT ["python"]
CMD ["app.py"]  # docker run에서 오버라이드 가능

# 스크립트 실행
ENTRYPOINT ["/entrypoint.sh"]
```

**CMD와의 차이점**

| 특성 | CMD | ENTRYPOINT |
|------|-----|------------|
| docker run 명령어로 오버라이드 | 쉬움 | 어려움 (--entrypoint 필요) |
| 주 용도 | 기본 명령어 또는 ENTRYPOINT 인자 | 컨테이너의 실행파일 |
| 실행 보장 | 오버라이드 가능 | 항상 실행 |

**조합 예시**

```dockerfile
# ENTRYPOINT와 CMD 조합
ENTRYPOINT ["python", "app.py"]
CMD ["--port", "8080"]

# docker run myapp 실행 시
# 실행 명령: python app.py --port 8080

# docker run myapp --port 3000 실행 시
# 실행 명령: python app.py --port 3000
```

#### 3.3.5. COPY

호스트의 파일이나 디렉토리를 이미지로 복사합니다.

```dockerfile
COPY [--chown=<user>:<group>] <src>... <dest>
COPY [--chown=<user>:<group>] ["<src>",... "<dest>"]
```

**사용 예시**

```dockerfile
# 기본 복사
COPY app.py /app/app.py

# 여러 파일 복사
COPY app.py requirements.txt /app/

# 디렉토리 복사
COPY ./src /app/src

# 소유권 지정
COPY --chown=appuser:appgroup app.py /app/

# 와일드카드 사용
COPY *.py /app/

# 멀티스테이지 빌드에서 다른 스테이지로부터 복사
COPY --from=builder /app/dist /app/dist
```

**특징**

- 로컬 파일시스템에서만 복사 가능
- URL은 지원하지 않음 (ADD 사용)
- .dockerignore 파일을 참조하여 제외할 파일 지정
- 메타데이터 (타임스탬프 등) 보존

#### 3.3.6. ADD

COPY와 유사하지만 추가 기능이 있습니다.

```dockerfile
ADD [--chown=<user>:<group>] <src>... <dest>
ADD [--chown=<user>:<group>] ["<src>",... "<dest>"]
```

**사용 예시**

```dockerfile
# 기본 복사 (COPY와 동일)
ADD app.py /app/

# tar 파일 자동 압축 해제
ADD archive.tar.gz /app/

# URL에서 다운로드
ADD https://example.com/file.tar.gz /tmp/

# 소유권 지정
ADD --chown=appuser:appgroup archive.tar.gz /app/
```

**COPY와의 차이점**

| 특성 | COPY | ADD |
|------|------|-----|
| 로컬 파일 복사 | 지원 | 지원 |
| URL 다운로드 | 미지원 | 지원 |
| tar 자동 압축 해제 | 미지원 | 지원 |
| 권장 사용 | 일반적인 복사 | 특수 기능이 필요할 때만 |

**모범 사례**

- 일반적인 파일 복사는 COPY 사용 권장
- ADD는 tar 자동 압축 해제가 필요할 때만 사용
- URL 다운로드는 RUN + curl/wget 사용 권장 (레이어 제어 가능)

#### 3.3.7. WORKDIR

작업 디렉토리를 설정합니다. RUN, CMD, ENTRYPOINT, COPY, ADD 명령어의 기준 경로가 됩니다.

```dockerfile
WORKDIR /path/to/workdir
```

**사용 예시**

```dockerfile
# 절대 경로
WORKDIR /app

# 상대 경로 (이전 WORKDIR 기준)
WORKDIR /app
WORKDIR subdir  # /app/subdir

# 환경변수 사용
ENV APPDIR=/app
WORKDIR ${APPDIR}

# 여러 번 사용 가능
WORKDIR /app
RUN touch file1.txt
WORKDIR /tmp
RUN touch file2.txt
```

**특징**

- 디렉토리가 없으면 자동 생성
- 여러 번 사용 가능
- 상대 경로는 이전 WORKDIR 기준
- RUN cd 대신 WORKDIR 사용 권장

#### 3.3.8. ENV

환경변수를 설정합니다. 빌드 시와 런타임 시 모두 사용됩니다.

```dockerfile
# 단일 변수
ENV <key>=<value>

# 여러 변수
ENV <key1>=<value1> <key2>=<value2> ...
```

**사용 예시**

```dockerfile
# 단일 변수
ENV NODE_ENV=production

# 여러 변수
ENV PORT=8080 \
    DEBUG=false \
    APP_HOME=/app

# 환경변수 참조
ENV APP_HOME=/app
ENV CONFIG_PATH=${APP_HOME}/config

# Dockerfile 내에서 사용
ENV VERSION=1.0.0
RUN echo "Building version ${VERSION}"
```

**특징**

- 빌드 타임과 런타임 모두에서 유효
- docker run -e 옵션으로 오버라이드 가능
- ${변수명} 형식으로 참조 가능
- ARG와 달리 이미지에 포함됨

#### 3.3.9. ARG

빌드 시에만 사용되는 변수를 정의합니다. 이미지에는 포함되지 않습니다.

```dockerfile
ARG <name>[=<default value>]
```

**사용 예시**

```dockerfile
# 기본값이 있는 ARG
ARG VERSION=1.0.0
ARG BUILD_DATE

# FROM 이전에 사용 (글로벌 스코프)
ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

# FROM 이후 재선언 필요
ARG VERSION
RUN echo "Building version ${VERSION}"

# --build-arg로 오버라이드
# docker build --build-arg VERSION=2.0.0 .

# 멀티스테이지 빌드에서 사용
ARG ENVIRONMENT=production
FROM node:18 AS builder
ARG ENVIRONMENT
RUN npm run build:${ENVIRONMENT}
```

**ENV와의 차이점**

| 특성 | ARG | ENV |
|------|-----|-----|
| 사용 시점 | 빌드 타임만 | 빌드 타임 + 런타임 |
| 이미지 포함 | 미포함 | 포함 |
| 오버라이드 | --build-arg | -e 또는 --env |
| 보안 | 히스토리에 남음 | 런타임에 노출 |

**보안 주의사항**

```dockerfile
# 나쁜 예: 민감한 정보를 ARG로 전달
ARG PASSWORD=secret123  # docker history에 남음!

# 좋은 예: 빌드 시 시크릿 마운트 사용 (BuildKit)
RUN --mount=type=secret,id=mysecret \
    PASSWORD=$(cat /run/secrets/mysecret) && \
    # 비밀번호 사용
```

#### 3.3.10. EXPOSE

컨테이너가 리스닝할 포트 (port)를 문서화합니다. 실제로 포트를 열지는 않습니다.

```dockerfile
EXPOSE <port> [<port>/<protocol>...]
```

**사용 예시**

```dockerfile
# 기본 포트 (TCP)
EXPOSE 8080

# 여러 포트
EXPOSE 8080 8443

# 프로토콜 지정
EXPOSE 80/tcp
EXPOSE 53/udp

# 범위 지정
EXPOSE 8000-8010
```

**특징**

- 문서화 목적 (실제로 포트를 열지 않음)
- docker run -p 또는 -P 옵션과 함께 사용
- -P 옵션 시 EXPOSE된 포트를 자동으로 매핑

**사용 예시**

```bash
# EXPOSE 8080이 정의된 이미지
# 모든 EXPOSE 포트를 자동 매핑
docker run -P myapp

# 특정 포트 매핑
docker run -p 3000:8080 myapp
```

#### 3.3.11. VOLUME

마운트 포인트 (mount point)를 생성하고 익명 볼륨으로 표시합니다.

```dockerfile
VOLUME ["/data"]
VOLUME /data
```

**사용 예시**

```dockerfile
# 단일 볼륨
VOLUME /var/lib/mysql

# 여러 볼륨
VOLUME ["/data", "/logs", "/config"]

# 배열 형식 (권장)
VOLUME ["/var/lib/postgresql/data"]
```

**특징**

- 컨테이너 실행 시 자동으로 익명 볼륨 생성
- 데이터 영속성 (persistence)을 위해 사용
- docker run -v 옵션으로 오버라이드 가능

**사용 시나리오**

```dockerfile
# 데이터베이스 데이터 디렉토리
VOLUME /var/lib/mysql

# 로그 디렉토리
VOLUME /var/log/nginx

# 설정 파일 디렉토리
VOLUME /etc/app/config
```

#### 3.3.12. USER

명령어를 실행할 사용자 (또는 UID) 및 그룹 (또는 GID)을 설정합니다.

```dockerfile
USER <user>[:<group>]
USER <UID>[:<GID>]
```

**사용 예시**

```dockerfile
# 사용자명으로 지정
USER appuser

# 사용자와 그룹 지정
USER appuser:appgroup

# UID와 GID로 지정
USER 1000:1000

# 사용자 생성 후 전환
RUN addgroup -S appgroup && adduser -S appuser -G appgroup
USER appuser

# 권한 설정과 함께 사용
RUN mkdir /app && chown -R appuser:appgroup /app
USER appuser
WORKDIR /app
```

**보안 모범 사례**

```dockerfile
# 나쁜 예: root로 실행
FROM ubuntu:22.04
COPY app.py /app/
CMD ["python", "/app/app.py"]

# 좋은 예: 비특권 사용자로 실행
FROM ubuntu:22.04
RUN useradd -m -u 1000 appuser
COPY --chown=appuser:appuser app.py /app/
USER appuser
CMD ["python", "/app/app.py"]
```

#### 3.3.13. LABEL

이미지에 메타데이터를 추가합니다.

```dockerfile
LABEL <key>=<value> <key>=<value> ...
```

**사용 예시**

```dockerfile
# 단일 레이블
LABEL version="1.0.0"

# 여러 레이블
LABEL maintainer="user@example.com" \
      version="1.0.0" \
      description="My application" \
      vendor="Company Name"

# 표준 레이블 (OCI 표준)
LABEL org.opencontainers.image.authors="user@example.com"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.created="2025-12-10"
LABEL org.opencontainers.image.title="My App"
LABEL org.opencontainers.image.description="My application description"
```

**레이블 조회**

```bash
# 이미지 레이블 확인
docker inspect --format='{{json .Config.Labels}}' myapp:1.0.0

# 특정 레이블로 필터링
docker images --filter "label=version=1.0.0"
```

#### 3.3.14. HEALTHCHECK

컨테이너의 상태를 확인하는 명령어를 정의합니다.

```dockerfile
HEALTHCHECK [OPTIONS] CMD command
HEALTHCHECK NONE  # 헬스체크 비활성화
```

**주요 옵션**

- `--interval=<duration>`: 체크 간격 (기본값: 30s)
- `--timeout=<duration>`: 타임아웃 (기본값: 30s)
- `--start-period=<duration>`: 시작 대기 시간 (기본값: 0s)
- `--retries=<number>`: 실패 재시도 횟수 (기본값: 3)

**사용 예시**

```dockerfile
# HTTP 엔드포인트 체크
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# 데이터베이스 연결 체크
HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
  CMD pg_isready -U postgres || exit 1

# 파일 존재 여부 체크
HEALTHCHECK CMD test -f /app/ready || exit 1

# 헬스체크 비활성화
HEALTHCHECK NONE
```

**종료 코드 (Exit Codes)**

- 0: 정상 (healthy)
- 1: 비정상 (unhealthy)
- 2: 예약됨 (reserved)

**상태 확인**

```bash
# 컨테이너 상태 확인
docker ps
# STATUS 컬럼에 (healthy) 또는 (unhealthy) 표시

# 상세 헬스체크 로그 확인
docker inspect --format='{{json .State.Health}}' container_name
```

#### 3.3.15. SHELL

기본 쉘을 변경합니다.

```dockerfile
SHELL ["executable", "parameters"]
```

**사용 예시**

```dockerfile
# Windows에서 PowerShell 사용
SHELL ["powershell", "-Command"]

# Bash 사용 (기본값)
SHELL ["/bin/bash", "-c"]

# Bash strict mode 사용
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

# 변경 후 RUN 명령어는 새 쉘 사용
SHELL ["/bin/bash", "-c"]
RUN echo "Using bash"
```

#### 3.3.16. ONBUILD

이미지가 다른 빌드의 베이스로 사용될 때 실행할 트리거 (trigger) 명령어를 추가합니다.

```dockerfile
ONBUILD <INSTRUCTION>
```

**사용 예시**

```dockerfile
# 베이스 이미지 (mybase:1.0)
FROM node:18
WORKDIR /app
ONBUILD COPY package*.json ./
ONBUILD RUN npm install
ONBUILD COPY . .

# 이 베이스를 사용하는 이미지
FROM mybase:1.0
# ONBUILD 명령어들이 자동으로 실행됨
CMD ["npm", "start"]
```

**특징**

- 베이스 이미지에서 정의
- 자식 이미지 빌드 시 FROM 직후 실행
- ONBUILD 내에 ONBUILD는 사용 불가
- FROM과 MAINTAINER는 트리거 불가

### 3.4. 멀티스테이지 빌드 (Multi-stage Builds)

여러 FROM 명령어를 사용하여 빌드 과정을 여러 단계로 나누고, 필요한 결과물만 최종 이미지에 포함시킵니다.

#### 3.4.1. 기본 예시

```dockerfile
# 빌드 스테이지
FROM golang:1.21 AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o app

# 런타임 스테이지
FROM alpine:3.18
RUN apk --no-cache add ca-certificates
WORKDIR /root/
COPY --from=builder /app/app .
CMD ["./app"]
```

#### 3.4.2. Node.js 애플리케이션 예시

```dockerfile
# 빌드 스테이지
FROM node:18 AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# 런타임 스테이지
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY --from=builder /app/dist ./dist
EXPOSE 3000
CMD ["node", "dist/index.js"]
```

#### 3.4.3. Python 애플리케이션 예시

```dockerfile
# 빌드 스테이지 (의존성 설치)
FROM python:3.11 AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# 런타임 스테이지
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["python", "app.py"]
```

#### 3.4.4. 특정 스테이지만 빌드

```bash
# builder 스테이지만 빌드
docker build --target builder -t myapp:builder .

# 테스트 스테이지 실행
docker build --target test -t myapp:test .
```

#### 3.4.5. 외부 이미지에서 복사

```dockerfile
# 외부 이미지에서 파일 복사
FROM alpine:3.18
COPY --from=nginx:latest /etc/nginx/nginx.conf /nginx.conf
```

### 3.5. .dockerignore 파일

빌드 컨텍스트 (build context)에서 제외할 파일과 디렉토리를 지정합니다.

#### 3.5.1. 문법

```dockerignore
# 주석
# 특정 파일
README.md

# 특정 디렉토리
node_modules/
.git/

# 패턴 매칭
*.log
*.tmp
temp*

# 제외 규칙 (! 사용)
!important.log

# 중첩 디렉토리
**/logs
```

#### 3.5.2. 예시

```dockerignore
# 버전 관리 시스템
.git
.gitignore
.gitattributes

# CI/CD
.github
.gitlab-ci.yml
.travis.yml

# 의존성 디렉토리
node_modules
venv
__pycache__

# 테스트 파일
*.test.js
*_test.go
tests/

# 문서
README.md
docs/
*.md

# 빌드 산출물
dist/
build/
*.o
*.so

# 환경 설정
.env
.env.local
*.key
*.pem

# IDE 설정
.vscode
.idea
*.swp
*.swo

# 로그 및 임시 파일
*.log
*.tmp
*.temp
logs/
```

### 3.6. Dockerfile 모범 사례 (Best Practices)

#### 3.6.1. 레이어 최적화

```dockerfile
# 나쁜 예: 많은 레이어 생성
FROM ubuntu:22.04
RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y git
RUN apt-get install -y vim

# 좋은 예: 레이어 최소화
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y \
    curl \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*
```

#### 3.6.2. 캐시 활용

```dockerfile
# 나쁜 예: 캐시 효율 낮음
FROM node:18
WORKDIR /app
COPY . .
RUN npm install

# 좋은 예: 캐시 효율 높음
FROM node:18
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
```

#### 3.6.3. 최소 베이스 이미지 사용

```dockerfile
# 표준 이미지 (큼)
FROM node:18              # ~900MB

# Slim 이미지 (중간)
FROM node:18-slim         # ~200MB

# Alpine 이미지 (작음)
FROM node:18-alpine       # ~170MB

# Distroless 이미지 (최소)
FROM gcr.io/distroless/nodejs18  # ~100MB
```

#### 3.6.4. 보안 강화

```dockerfile
# 1. 비특권 사용자 사용
FROM node:18-alpine
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001
USER nodejs

# 2. 최신 베이스 이미지 사용
FROM python:3.11-slim

# 3. 불필요한 패키지 제거
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. 민감한 정보 제외
# .dockerignore에 .env, *.key 추가

# 5. HEALTHCHECK 추가
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:3000/health || exit 1
```

#### 3.6.5. 빌드 인자 활용

```dockerfile
# 환경별 빌드
ARG ENVIRONMENT=production
FROM node:18-alpine AS base

# 개발 환경
FROM base AS development
ENV NODE_ENV=development
RUN npm install

# 프로덕션 환경
FROM base AS production
ENV NODE_ENV=production
RUN npm ci --only=production

# 최종 스테이지
FROM ${ENVIRONMENT} AS final
COPY . .
CMD ["node", "app.js"]
```

**빌드 명령어**

```bash
# 개발 환경 빌드
docker build --build-arg ENVIRONMENT=development -t myapp:dev .

# 프로덕션 환경 빌드
docker build --build-arg ENVIRONMENT=production -t myapp:prod .
```

### 3.7. 완전한 예시

#### 3.7.1. Node.js 프로덕션 Dockerfile

```dockerfile
# 멀티스테이지 빌드
FROM node:18-alpine AS base
RUN apk add --no-cache libc6-compat
WORKDIR /app

# 의존성 설치
FROM base AS deps
COPY package*.json ./
RUN npm ci

# 빌드
FROM base AS builder
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

# 프로덕션 이미지
FROM base AS runner
ENV NODE_ENV=production

RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

COPY --from=builder --chown=nextjs:nodejs /app/dist ./dist
COPY --from=builder --chown=nextjs:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=nextjs:nodejs /app/package.json ./package.json

USER nextjs

EXPOSE 3000

ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD node healthcheck.js

CMD ["node", "dist/index.js"]
```

#### 3.7.2. Python FastAPI Dockerfile

```dockerfile
# 빌드 스테이지
FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir --no-warn-script-location -r requirements.txt

# 런타임 스테이지
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/root/.local/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser

COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local
COPY --chown=appuser:appuser . .

USER appuser

ENV PATH=/home/appuser/.local/bin:$PATH

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 4. docker-compose.yml

### 4.1. Docker Compose 개요

Docker Compose는 여러 컨테이너로 구성된 애플리케이션을 정의하고 실행하기 위한 도구입니다. YAML 파일로 서비스 (services), 네트워크, 볼륨을 구성합니다.

### 4.2. 기본 구조

```yaml
version: '3.8'  # Compose 파일 버전

services:       # 서비스 정의
  service1:
    # 서비스 설정
  service2:
    # 서비스 설정

networks:       # 네트워크 정의 (선택사항)
  network1:
    # 네트워크 설정

volumes:        # 볼륨 정의 (선택사항)
  volume1:
    # 볼륨 설정

configs:        # 설정 정의 (선택사항)
  config1:
    # 설정

secrets:        # 시크릿 정의 (선택사항)
  secret1:
    # 시크릿 설정
```

### 4.3. 서비스 설정 (Service Configuration)

#### 4.3.1. image

사용할 이미지를 지정합니다.

```yaml
services:
  web:
    image: nginx:latest

  db:
    image: postgres:15-alpine

  app:
    image: myregistry.com/myapp:1.0.0
```

#### 4.3.2. build

Dockerfile로부터 이미지를 빌드합니다.

```yaml
services:
  web:
    # 간단한 형식
    build: .

  app:
    # 확장 형식
    build:
      context: ./app
      dockerfile: Dockerfile.prod
      args:
        - VERSION=1.0.0
        - ENVIRONMENT=production
      target: production
      cache_from:
        - myapp:cache
      labels:
        - "com.example.version=1.0.0"
```

**주요 옵션**

- `context`: 빌드 컨텍스트 경로
- `dockerfile`: Dockerfile 경로
- `args`: 빌드 인자
- `target`: 멀티스테이지 빌드 타겟
- `cache_from`: 캐시 소스 이미지
- `labels`: 이미지 레이블

#### 4.3.3. container_name

컨테이너 이름을 지정합니다.

```yaml
services:
  web:
    image: nginx
    container_name: my_nginx_server
```

#### 4.3.4. ports

포트 매핑을 정의합니다.

```yaml
services:
  web:
    image: nginx
    ports:
      # HOST:CONTAINER
      - "8080:80"
      - "8443:443"

      # 호스트 포트를 지정하지 않으면 랜덤 할당
      - "3000"

      # 프로토콜 지정
      - "53:53/udp"

      # IP 주소 지정
      - "127.0.0.1:8080:80"

      # 범위 지정
      - "8000-8010:8000-8010"
```

#### 4.3.5. expose

다른 서비스에 포트를 노출합니다 (호스트에는 노출하지 않음).

```yaml
services:
  api:
    image: myapi
    expose:
      - "3000"
      - "8080"
```

#### 4.3.6. volumes

볼륨 마운트를 정의합니다.

```yaml
services:
  db:
    image: postgres:15
    volumes:
      # 이름 있는 볼륨
      - db_data:/var/lib/postgresql/data

      # 바인드 마운트 (절대 경로)
      - /host/path:/container/path

      # 바인드 마운트 (상대 경로)
      - ./config:/etc/app/config

      # 읽기 전용
      - ./config:/etc/app/config:ro

      # 익명 볼륨
      - /container/path

      # 확장 형식
      - type: volume
        source: db_data
        target: /var/lib/postgresql/data
        volume:
          nocopy: true

      - type: bind
        source: ./config
        target: /etc/app/config
        read_only: true

volumes:
  db_data:
```

**마운트 타입**

- `volume`: 명명된 볼륨 또는 익명 볼륨
- `bind`: 호스트 경로 바인드
- `tmpfs`: 임시 파일시스템 (메모리)

#### 4.3.7. environment

환경변수를 설정합니다.

```yaml
services:
  app:
    image: myapp
    environment:
      # 키-값 형식
      NODE_ENV: production
      DEBUG: "false"
      PORT: 3000

      # 배열 형식
      # - NODE_ENV=production
      # - DEBUG=false

      # 호스트 환경변수 참조
      API_KEY: ${API_KEY}

      # 기본값 설정
      LOG_LEVEL: ${LOG_LEVEL:-info}
```

#### 4.3.8. env_file

환경변수 파일을 지정합니다.

```yaml
services:
  app:
    image: myapp
    env_file:
      # 단일 파일
      - .env

      # 여러 파일 (나중 파일이 우선)
      # - .env
      # - .env.production
```

**.env 파일 예시**

```env
NODE_ENV=production
DEBUG=false
PORT=3000
DATABASE_URL=postgres://user:pass@db:5432/mydb
```

#### 4.3.9. command

컨테이너 시작 시 실행할 명령어를 오버라이드합니다.

```yaml
services:
  app:
    image: myapp
    # 문자열 형식
    command: python app.py --port 8080

    # 배열 형식 (권장)
    # command: ["python", "app.py", "--port", "8080"]
```

#### 4.3.10. entrypoint

컨테이너의 엔트리포인트를 오버라이드합니다.

```yaml
services:
  app:
    image: myapp
    entrypoint: /entrypoint.sh

    # 배열 형식
    # entrypoint: ["/entrypoint.sh"]

    # command와 함께 사용
    # entrypoint: ["python"]
    # command: ["app.py", "--port", "8080"]
```

#### 4.3.11. depends_on

서비스 간 시작 순서 및 의존성을 정의합니다.

```yaml
services:
  web:
    image: nginx
    depends_on:
      - app
      - cache

  app:
    image: myapp
    depends_on:
      # 확장 형식 (v3.9+)
      db:
        condition: service_healthy
      cache:
        condition: service_started

  db:
    image: postgres:15
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  cache:
    image: redis:7-alpine
```

**조건 (Conditions)**

- `service_started`: 서비스가 시작됨 (기본값)
- `service_healthy`: 서비스가 healthy 상태
- `service_completed_successfully`: 서비스가 성공적으로 완료됨

#### 4.3.12. networks

서비스가 연결할 네트워크를 지정합니다.

```yaml
services:
  web:
    image: nginx
    networks:
      - frontend
      - backend

  app:
    image: myapp
    networks:
      backend:
        aliases:
          - api
        ipv4_address: 172.20.0.10

  db:
    image: postgres:15
    networks:
      - backend

networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

#### 4.3.13. restart

재시작 정책을 설정합니다.

```yaml
services:
  web:
    image: nginx
    restart: always

  app:
    image: myapp
    restart: on-failure

  worker:
    image: worker
    restart: unless-stopped
```

**정책 옵션**

- `no`: 재시작하지 않음 (기본값)
- `always`: 항상 재시작
- `on-failure`: 실패 시에만 재시작
- `unless-stopped`: 수동으로 중지하지 않는 한 재시작

#### 4.3.14. healthcheck

헬스체크를 정의합니다.

```yaml
services:
  web:
    image: nginx
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  db:
    image: postgres:15
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  # 헬스체크 비활성화
  cache:
    image: redis:7
    healthcheck:
      disable: true
```

#### 4.3.15. labels

메타데이터 레이블을 추가합니다.

```yaml
services:
  web:
    image: nginx
    labels:
      com.example.description: "Web server"
      com.example.version: "1.0.0"
      com.example.team: "frontend"

      # Traefik 라우팅 설정
      traefik.enable: "true"
      traefik.http.routers.web.rule: "Host(`example.com`)"
```

#### 4.3.16. deploy

Swarm 모드 배포 설정입니다 (docker stack deploy에서 사용).

```yaml
services:
  web:
    image: nginx
    deploy:
      replicas: 3
      update_config:
        parallelism: 2
        delay: 10s
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M
      placement:
        constraints:
          - node.role == worker
```

#### 4.3.17. logging

로깅 설정을 정의합니다.

```yaml
services:
  web:
    image: nginx
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  app:
    image: myapp
    logging:
      driver: "syslog"
      options:
        syslog-address: "tcp://192.168.0.1:514"
```

**로깅 드라이버**

- `json-file`: JSON 파일 (기본값)
- `syslog`: Syslog
- `journald`: Systemd journal
- `gelf`: Graylog Extended Log Format
- `fluentd`: Fluentd
- `awslogs`: Amazon CloudWatch Logs

#### 4.3.18. 기타 주요 옵션

```yaml
services:
  app:
    image: myapp

    # 작업 디렉토리
    working_dir: /app

    # 사용자 지정
    user: "1000:1000"

    # 호스트네임
    hostname: myapp-server

    # DNS 설정
    dns:
      - 8.8.8.8
      - 8.8.4.4
    dns_search:
      - example.com

    # 추가 호스트
    extra_hosts:
      - "db.local:192.168.1.10"
      - "cache.local:192.168.1.11"

    # 보안 옵션
    security_opt:
      - no-new-privileges:true

    # 프로세스 리소스 제한
    ulimits:
      nproc: 65535
      nofile:
        soft: 20000
        hard: 40000

    # tmpfs 마운트
    tmpfs:
      - /tmp
      - /run
```

### 4.4. 네트워크 설정 (Networks Configuration)

```yaml
networks:
  # 기본 브리지 네트워크
  frontend:
    driver: bridge

  # 커스텀 서브넷
  backend:
    driver: bridge
    ipam:
      driver: default
      config:
        - subnet: 172.20.0.0/16
          gateway: 172.20.0.1

  # 외부 네트워크 사용
  external_net:
    external: true
    name: my_external_network

  # Overlay 네트워크 (Swarm)
  overlay_net:
    driver: overlay
    attachable: true
```

### 4.5. 볼륨 설정 (Volumes Configuration)

```yaml
volumes:
  # 기본 볼륨
  db_data:

  # 드라이버 옵션
  nfs_data:
    driver: local
    driver_opts:
      type: nfs
      o: addr=192.168.1.1,rw
      device: ":/path/to/dir"

  # 외부 볼륨
  external_volume:
    external: true
    name: my_external_volume

  # 레이블
  labeled_volume:
    driver: local
    labels:
      com.example.description: "Database volume"
      com.example.environment: "production"
```

### 4.6. Configs와 Secrets (Swarm 모드)

#### 4.6.1. Configs

```yaml
version: '3.8'

services:
  app:
    image: myapp
    configs:
      - source: app_config
        target: /etc/app/config.yml
        mode: 0440

configs:
  app_config:
    file: ./config.yml

  app_config_external:
    external: true
    name: prod_app_config
```

#### 4.6.2. Secrets

```yaml
version: '3.8'

services:
  app:
    image: myapp
    secrets:
      - source: db_password
        target: /run/secrets/db_password
        mode: 0400
      - api_key

secrets:
  db_password:
    file: ./secrets/db_password.txt

  api_key:
    external: true
    name: prod_api_key
```

### 4.7. 변수 치환 (Variable Substitution)

#### 4.7.1. 환경변수 사용

```yaml
services:
  web:
    image: nginx:${NGINX_VERSION:-latest}
    ports:
      - "${WEB_PORT:-8080}:80"
    environment:
      - API_URL=${API_URL}
      - ENV=${ENV:-development}
```

**.env 파일**

```env
NGINX_VERSION=1.25
WEB_PORT=8080
API_URL=http://api.example.com
ENV=production
```

#### 4.7.2. 환경 파일 우선순위

1. Compose 파일에 설정된 값
2. 쉘 환경변수
3. .env 파일
4. Dockerfile의 ENV
5. 기본값

### 4.8. 확장 필드 (Extension Fields)

재사용 가능한 설정을 정의합니다.

```yaml
version: '3.8'

# 확장 필드 정의 (x-로 시작)
x-common-variables: &common-vars
  TZ: Asia/Seoul
  LOG_LEVEL: info

x-common-healthcheck: &common-healthcheck
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s

x-logging: &default-logging
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"

services:
  web:
    image: nginx
    environment:
      <<: *common-vars
      SERVICE_NAME: web
    healthcheck:
      <<: *common-healthcheck
      test: ["CMD", "curl", "-f", "http://localhost"]
    logging: *default-logging

  app:
    image: myapp
    environment:
      <<: *common-vars
      SERVICE_NAME: app
    healthcheck:
      <<: *common-healthcheck
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
    logging: *default-logging
```

### 4.9. Docker Compose 명령어

#### 4.9.1. docker compose up

서비스를 생성하고 시작합니다.

```bash
# 기본 실행
docker compose up

# 백그라운드 실행
docker compose up -d

# 특정 서비스만 시작
docker compose up web db

# 빌드 후 시작
docker compose up --build

# 강제 재생성
docker compose up --force-recreate

# 스케일 조정
docker compose up --scale web=3 --scale worker=5
```

#### 4.9.2. docker compose down

서비스를 중지하고 제거합니다.

```bash
# 기본 종료 (네트워크도 제거)
docker compose down

# 볼륨도 함께 제거
docker compose down -v

# 이미지도 함께 제거
docker compose down --rmi all

# 모두 제거
docker compose down -v --rmi all --remove-orphans
```

#### 4.9.3. docker compose ps

실행 중인 서비스를 조회합니다.

```bash
# 모든 서비스 조회
docker compose ps

# 모든 컨테이너 조회 (중지된 것 포함)
docker compose ps -a

# 특정 서비스만 조회
docker compose ps web
```

#### 4.9.4. docker compose logs

서비스의 로그를 조회합니다.

```bash
# 모든 서비스 로그
docker compose logs

# 실시간 로그
docker compose logs -f

# 특정 서비스 로그
docker compose logs web

# 마지막 100줄
docker compose logs --tail=100

# 타임스탬프 포함
docker compose logs -t
```

#### 4.9.5. docker compose exec

실행 중인 서비스에서 명령어를 실행합니다.

```bash
# 대화형 쉘
docker compose exec web /bin/bash

# 단일 명령어
docker compose exec db psql -U postgres

# 특정 사용자로 실행
docker compose exec -u root web apt-get update
```

#### 4.9.6. docker compose build

서비스 이미지를 빌드합니다.

```bash
# 모든 서비스 빌드
docker compose build

# 특정 서비스만 빌드
docker compose build web

# 캐시 없이 빌드
docker compose build --no-cache

# 병렬 빌드
docker compose build --parallel
```

#### 4.9.7. docker compose restart

서비스를 재시작합니다.

```bash
# 모든 서비스 재시작
docker compose restart

# 특정 서비스 재시작
docker compose restart web

# 타임아웃 지정
docker compose restart -t 30 web
```

#### 4.9.8. docker compose stop / start

서비스를 중지하거나 시작합니다.

```bash
# 중지
docker compose stop
docker compose stop web

# 시작
docker compose start
docker compose start web
```

#### 4.9.9. docker compose config

Compose 파일의 설정을 검증하고 출력합니다.

```bash
# 설정 확인
docker compose config

# 간략한 출력
docker compose config --services

# 특정 서비스만 출력
docker compose config --services web
```

### 4.10. 완전한 예시

#### 4.10.1. 풀스택 웹 애플리케이션

```yaml
version: '3.8'

# 확장 필드
x-common-variables: &common-variables
  TZ: Asia/Seoul
  LOG_LEVEL: ${LOG_LEVEL:-info}

x-logging: &default-logging
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"

services:
  # 프론트엔드
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.prod
      args:
        - NODE_ENV=production
    container_name: frontend
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      <<: *common-variables
      NEXT_PUBLIC_API_URL: http://backend:4000
    networks:
      - frontend_network
    depends_on:
      backend:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    logging: *default-logging

  # 백엔드 API
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile.prod
    container_name: backend
    restart: unless-stopped
    ports:
      - "4000:4000"
    environment:
      <<: *common-variables
      NODE_ENV: production
      DATABASE_URL: postgres://postgres:${DB_PASSWORD}@db:5432/myapp
      REDIS_URL: redis://cache:6379
      JWT_SECRET: ${JWT_SECRET}
    env_file:
      - .env.production
    networks:
      - frontend_network
      - backend_network
    depends_on:
      db:
        condition: service_healthy
      cache:
        condition: service_started
    volumes:
      - ./backend/uploads:/app/uploads
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:4000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    logging: *default-logging

  # PostgreSQL 데이터베이스
  db:
    image: postgres:15-alpine
    container_name: postgres_db
    restart: unless-stopped
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: myapp
      PGDATA: /var/lib/postgresql/data/pgdata
    networks:
      - backend_network
    volumes:
      - db_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
    logging: *default-logging

  # Redis 캐시
  cache:
    image: redis:7-alpine
    container_name: redis_cache
    restart: unless-stopped
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    networks:
      - backend_network
    volumes:
      - cache_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    logging: *default-logging

  # Nginx 리버스 프록시
  nginx:
    image: nginx:alpine
    container_name: nginx_proxy
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - ./nginx/logs:/var/log/nginx
    networks:
      - frontend_network
    depends_on:
      - frontend
      - backend
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    logging: *default-logging

networks:
  frontend_network:
    driver: bridge
  backend_network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16

volumes:
  db_data:
    driver: local
  cache_data:
    driver: local
```

#### 4.10.2. 마이크로서비스 아키텍처

```yaml
version: '3.8'

services:
  # API Gateway
  gateway:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./gateway/nginx.conf:/etc/nginx/nginx.conf:ro
    networks:
      - public
    depends_on:
      - auth-service
      - user-service
      - order-service

  # 인증 서비스
  auth-service:
    build: ./services/auth
    environment:
      - SERVICE_NAME=auth
      - DATABASE_URL=postgres://postgres:secret@auth-db:5432/auth
      - REDIS_URL=redis://cache:6379/0
    networks:
      - public
      - auth-network
    depends_on:
      - auth-db
      - cache

  auth-db:
    image: postgres:15-alpine
    environment:
      POSTGRES_PASSWORD: secret
      POSTGRES_DB: auth
    volumes:
      - auth_db_data:/var/lib/postgresql/data
    networks:
      - auth-network

  # 사용자 서비스
  user-service:
    build: ./services/user
    environment:
      - SERVICE_NAME=user
      - DATABASE_URL=postgres://postgres:secret@user-db:5432/user
    networks:
      - public
      - user-network
    depends_on:
      - user-db

  user-db:
    image: postgres:15-alpine
    environment:
      POSTGRES_PASSWORD: secret
      POSTGRES_DB: user
    volumes:
      - user_db_data:/var/lib/postgresql/data
    networks:
      - user-network

  # 주문 서비스
  order-service:
    build: ./services/order
    environment:
      - SERVICE_NAME=order
      - DATABASE_URL=mongodb://order-db:27017/order
      - RABBITMQ_URL=amqp://rabbitmq:5672
    networks:
      - public
      - order-network
      - message-network
    depends_on:
      - order-db
      - rabbitmq

  order-db:
    image: mongo:7
    volumes:
      - order_db_data:/data/db
    networks:
      - order-network

  # 메시지 큐
  rabbitmq:
    image: rabbitmq:3-management-alpine
    ports:
      - "15672:15672"
    environment:
      RABBITMQ_DEFAULT_USER: admin
      RABBITMQ_DEFAULT_PASS: secret
    networks:
      - message-network
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq

  # 공유 캐시
  cache:
    image: redis:7-alpine
    networks:
      - auth-network
    volumes:
      - cache_data:/data

networks:
  public:
  auth-network:
  user-network:
  order-network:
  message-network:

volumes:
  auth_db_data:
  user_db_data:
  order_db_data:
  rabbitmq_data:
  cache_data:
```

#### 4.10.3. 개발 환경 설정

**docker-compose.yml** (기본)

```yaml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.dev
    volumes:
      - .:/app
      - /app/node_modules
    environment:
      - NODE_ENV=development
    command: npm run dev
```

**docker-compose.override.yml** (자동 적용)

```yaml
version: '3.8'

services:
  app:
    ports:
      - "3000:3000"
      - "9229:9229"  # Node.js 디버거
    environment:
      - DEBUG=*
```

**docker-compose.prod.yml** (프로덕션)

```yaml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.prod
    restart: always
    environment:
      - NODE_ENV=production
```

**명령어**

```bash
# 개발 환경 (docker-compose.yml + docker-compose.override.yml)
docker compose up

# 프로덕션 환경
docker compose -f docker-compose.yml -f docker-compose.prod.yml up
```

### 4.11. 모범 사례 (Best Practices)

#### 4.11.1. 환경 분리

```bash
# 파일 구조
.
├── docker-compose.yml          # 공통 설정
├── docker-compose.dev.yml      # 개발 환경
├── docker-compose.prod.yml     # 프로덕션 환경
├── .env.dev                    # 개발 환경변수
└── .env.prod                   # 프로덕션 환경변수
```

```bash
# 개발 환경
docker compose -f docker-compose.yml -f docker-compose.dev.yml --env-file .env.dev up

# 프로덕션 환경
docker compose -f docker-compose.yml -f docker-compose.prod.yml --env-file .env.prod up
```

#### 4.11.2. 헬스체크 활용

```yaml
services:
  web:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  app:
    depends_on:
      web:
        condition: service_healthy
```

#### 4.11.3. 리소스 제한

```yaml
services:
  app:
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M
```

#### 4.11.4. 보안 강화

```yaml
services:
  app:
    # 비특권 사용자
    user: "1000:1000"

    # 읽기 전용 루트 파일시스템
    read_only: true
    tmpfs:
      - /tmp
      - /app/temp

    # 보안 옵션
    security_opt:
      - no-new-privileges:true

    # 민감한 정보는 secrets 사용
    secrets:
      - db_password

    # 환경변수 검증
    environment:
      - DATABASE_URL=${DATABASE_URL:?DATABASE_URL must be set}
```

## 5. 클라우드 배포 (Cloud Deployment)

### 5.1. Google Cloud Run 배포

Google Cloud Run에 Docker 이미지를 배포하는 방법입니다.

#### 5.1.1. 기본 배포 명령어

```bash
gcloud run deploy [SERVICE_NAME] --image [IMAGE_URL] --region [REGION] --platform managed --allow-unauthenticated
```

**주요 옵션**

- `--image`: Docker Hub 또는 Container Registry의 이미지 URL
- `--region`: 배포할 리전 (예: asia-northeast3)
- `--cpu`: CPU 할당 (기본값: 1)
- `--memory`: 메모리 할당 (예: 2Gi, 512Mi)
- `--allow-unauthenticated`: 인증 없이 접근 허용
- `--platform`: managed (완전 관리형) 또는 gke (Kubernetes)

#### 5.1.2. 실전 배포 예시

**백엔드 배포**

```bash
gcloud run deploy mis18-backend --image index.docker.io/c0z0c/mis18_backend:v1.1 --region asia-northeast3 --cpu 1 --memory 2Gi --allow-unauthenticated
```

**프론트엔드 배포**

```bash
gcloud run deploy mis18-frontend --image index.docker.io/c0z0c/mis18_frontend:v1.1 --region asia-northeast3 --allow-unauthenticated
```

#### 5.1.3. 완전 자동화 스크립트

빌드부터 배포까지 한 번에 수행하는 스크립트입니다.

**백엔드 자동화**

```bash
# 1. Docker Compose로 빌드
docker-compose build backend

# 2. 이미지 태깅
docker tag c0z0c/mis18_backend:latest index.docker.io/c0z0c/mis18_backend:v1.1

# 3. Docker Hub에 푸시
docker push index.docker.io/c0z0c/mis18_backend:v1.1

# 4. Google Cloud Run에 배포
gcloud run deploy mis18-backend --image index.docker.io/c0z0c/mis18_backend:v1.1 --region asia-northeast3 --cpu 1 --memory 2Gi --platform managed --allow-unauthenticated
```

**프론트엔드 자동화**

```bash
# 1. Docker Compose로 빌드
docker-compose build frontend

# 2. 이미지 태깅
docker tag c0z0c/mis18_frontend:latest index.docker.io/c0z0c/mis18_frontend:v1.1

# 3. Docker Hub에 푸시
docker push index.docker.io/c0z0c/mis18_frontend:v1.1

# 4. Google Cloud Run에 배포
gcloud run deploy mis18-frontend --image index.docker.io/c0z0c/mis18_frontend:v1.1 --region asia-northeast3 --platform managed --allow-unauthenticated
```

#### 5.1.4. 환경 변수 설정

배포 시 환경 변수를 설정하려면:

```bash
gcloud run deploy mis18-backend --image index.docker.io/c0z0c/mis18_backend:v1.1 --region asia-northeast3 --set-env-vars="DATABASE_URL=postgresql://...,API_KEY=xxx" --allow-unauthenticated
```

또는 파일에서 환경 변수 로드:

```bash
gcloud run deploy mis18-backend --image index.docker.io/c0z0c/mis18_backend:v1.1 --region asia-northeast3 --env-vars-file=.env.yaml --allow-unauthenticated
```

#### 5.1.5. PowerShell 스크립트 예시

Windows PowerShell에서 사용할 수 있는 완전 자동화 스크립트:

```powershell
# deploy-to-cloud.ps1

# 변수 설정
$BACKEND_IMAGE = "index.docker.io/c0z0c/mis18_backend"
$FRONTEND_IMAGE = "index.docker.io/c0z0c/mis18_frontend"
$VERSION = "v1.1"
$REGION = "asia-northeast3"

# 백엔드 배포
Write-Host "Building backend..." -ForegroundColor Green
docker-compose build backend

Write-Host "Tagging backend image..." -ForegroundColor Green
docker tag c0z0c/mis18_backend:latest "${BACKEND_IMAGE}:${VERSION}"

Write-Host "Pushing backend to Docker Hub..." -ForegroundColor Green
docker push "${BACKEND_IMAGE}:${VERSION}"

Write-Host "Deploying backend to Cloud Run..." -ForegroundColor Green
gcloud run deploy mis18-backend --image "${BACKEND_IMAGE}:${VERSION}" --region $REGION --cpu 1 --memory 2Gi --platform managed --allow-unauthenticated

# 프론트엔드 배포
Write-Host "Building frontend..." -ForegroundColor Green
docker-compose build frontend

Write-Host "Tagging frontend image..." -ForegroundColor Green
docker tag c0z0c/mis18_frontend:latest "${FRONTEND_IMAGE}:${VERSION}"

Write-Host "Pushing frontend to Docker Hub..." -ForegroundColor Green
docker push "${FRONTEND_IMAGE}:${VERSION}"

Write-Host "Deploying frontend to Cloud Run..." -ForegroundColor Green
gcloud run deploy mis18-frontend --image "${FRONTEND_IMAGE}:${VERSION}" --region $REGION --platform managed --allow-unauthenticated

Write-Host "Deployment completed!" -ForegroundColor Green
```

#### 5.1.6. 배포 상태 확인

```bash
# 서비스 목록 조회
gcloud run services list --region asia-northeast3

# 특정 서비스 상세 정보
gcloud run services describe mis18-backend --region asia-northeast3

# 로그 확인
gcloud run logs read mis18-backend --region asia-northeast3 --limit 50

# 실시간 로그 스트리밍
gcloud run logs tail mis18-backend --region asia-northeast3
```

#### 5.1.7. 주의사항

1. **Docker Hub 로그인**: 푸시 전에 `docker login` 필요
2. **gcloud 인증**: `gcloud auth login` 및 `gcloud config set project [PROJECT_ID]` 필요
3. **이미지 경로**: `index.docker.io` 사용으로 레지스트리 명시
4. **리전 선택**: 서울 리전은 `asia-northeast3`
5. **비용**: Cloud Run은 사용한 만큼 과금 (무료 할당량 있음)

---

이상으로 Docker 명령어 설명서를 완성했습니다. 각 섹션은 다음을 포함합니다:

1. **기본 명령어**: 이미지/컨테이너 관리, 정보 조회
2. **추가 명령어**: 네트워크/볼륨 관리, 고급 기능, 시스템 관리
3. **Dockerfile**: 모든 명령어와 멀티스테이지 빌드, 모범 사례
4. **docker-compose.yml**: 서비스 구성, 네트워크/볼륨 설정, 실전 예시
5. **클라우드 배포**: Docker Hub 푸시 및 Google Cloud Run 배포 완전 가이드

각 명령어와 옵션에 대한 자세한 설명, 사용 예시, 그리고 실무에 적용할 수 있는 완전한 예제를 포함했습니다.