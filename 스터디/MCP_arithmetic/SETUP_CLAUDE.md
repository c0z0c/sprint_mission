# Claude Desktop MCP 연결 설정

## 1. 의존성 설치

```powershell
pip install -r requirements.txt
```

## 2. Claude Desktop 설정 파일 위치

### Windows
```
%APPDATA%\Claude\claude_desktop_config.json
```

실제 경로 예시:
```
C:\Users\사용자명\AppData\Roaming\Claude\claude_desktop_config.json
```

### macOS
```
~/Library/Application Support/Claude/claude_desktop_config.json
```

### Linux
```
~/.config/Claude/claude_desktop_config.json
```

## 3. 설정 파일 복사

### 방법 A: 자동 복사 (PowerShell)

```powershell
# 설정 디렉토리 생성
$configDir = "$env:APPDATA\Claude"
New-Item -ItemType Directory -Force -Path $configDir

# 설정 파일 복사
Copy-Item -Path ".\claude_desktop_config.json" -Destination "$configDir\claude_desktop_config.json" -Force

Write-Host "설정 파일 복사 완료: $configDir\claude_desktop_config.json"
```

### 방법 B: 수동 복사

1. `claude_desktop_config.json` 파일 열기
2. 내용 복사
3. `%APPDATA%\Claude\claude_desktop_config.json` 생성 또는 수정
4. 내용 붙여넣기

### 기존 설정 병합 (다른 MCP 서버가 있는 경우)

기존 파일이 있다면 `mcpServers` 객체에 `arithmetic` 항목만 추가:

```json
{
  "mcpServers": {
    "기존서버1": { ... },
    "기존서버2": { ... },
    "arithmetic": {
      "command": "python",
      "args": [
        "d:\\GoogleDrive\\homepage\\스프린트미션\\스터디\\MCP_SAMPLE\\mcp_server.py"
      ]
    }
  }
}
```

## 4. Claude Desktop 재시작

1. Claude Desktop 완전 종료 (트레이 아이콘 우클릭 → Quit)
2. Claude Desktop 재실행

## 5. 연결 확인

Claude Desktop에서 다음과 같이 테스트:

```
arithmetic 도구를 사용해서 12와 8을 더해줘
```

예상 응답:
```json
{
  "success": true,
  "operation": "add",
  "operands": [12, 8],
  "result": 20
}
```

## 6. 트러블슈팅

### 도구가 보이지 않는 경우

1. 설정 파일 경로 확인:
```powershell
Get-Content "$env:APPDATA\Claude\claude_desktop_config.json"
```

2. Python 경로 확인:
```powershell
Get-Command python | Select-Object -ExpandProperty Source
```

3. 서버 직접 실행 테스트:
```powershell
python mcp_server.py
```
(Ctrl+C로 종료)

4. Claude Desktop 로그 확인:
```
%APPDATA%\Claude\logs\
```

### 경로 문제

- `command`는 시스템 PATH에 있는 `python` 또는 절대 경로 사용
- `args`의 경로는 **역슬래시 이스케이프** (`\\`) 필요

가상 환경 사용 시:
```json
{
  "mcpServers": {
    "arithmetic": {
      "command": "d:\\GoogleDrive\\homepage\\스프린트미션\\스터디\\MCP_SAMPLE\\.venv\\Scripts\\python.exe",
      "args": [
        "d:\\GoogleDrive\\homepage\\스프린트미션\\스터디\\MCP_SAMPLE\\mcp_server.py"
      ]
    }
  }
}
```

## 7. HTTP 서버와 비교

| 항목 | HTTP 서버 (app.py) | stdio 서버 (mcp_server.py) |
|------|-------------------|---------------------------|
| 실행 | `python run_server.py` | Claude Desktop 자동 실행 |
| 프로토콜 | HTTP REST API | stdio JSON-RPC |
| 테스트 | 브라우저/curl | Claude Desktop 대화 |
| 포트 | 8000 | 없음 (stdin/stdout) |
| 용도 | 개발/디버깅 | 프로덕션 Claude 통합 |

두 서버는 **독립적**이며 동시 사용 가능.
