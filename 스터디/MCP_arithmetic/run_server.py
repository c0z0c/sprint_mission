"""서버 실행 스크립트 (Windows PowerShell)."""
import subprocess
import sys
from pathlib import Path

from ulogger import *
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def main():
    """FastAPI 서버 실행."""
    # 프로젝트 루트
    project_root = Path(__file__).parent

    logger.debug("=== MCP Arithmetic Server ===")
    logger.debug(f"Project root: {project_root}")
    logger.debug(f"Python: {sys.executable}")
    logger.debug(f"Server: http://localhost:8000")
    logger.debug(f"API Docs: http://localhost:8000/docs")
    logger.debug("Starting server...")
    logger.debug("")

    # uvicorn 실행
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app:app",
                "--reload",
                "--port",
                "8000",
                "--host",
                "127.0.0.1",
            ],
            cwd=str(project_root),
            check=True,
        )
    except KeyboardInterrupt:
        logger.error("Server stopped.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
