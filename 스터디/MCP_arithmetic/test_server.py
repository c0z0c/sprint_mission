"""서버 실행 및 기본 검증 스크립트."""
import sys
from pathlib import Path
from ulogger import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tools import ArithmeticTool

def test_arithmetic():
    """사칙연산 도구 기본 테스트."""
    tool = ArithmeticTool()

    # 테스트 케이스
    test_cases = [
        ("add", [10, 5, 3], 18.0),
        ("subtract", [100, 25, 5], 70.0),
        ("multiply", [2, 3, 4], 24.0),
        ("divide", [100, 4, 5], 5.0),
    ]

    logger.debug("")
    logger.debug("=== 사칙연산 도구 테스트 ===")

    for operation, operands, expected in test_cases:
        try:
            result = tool.run(operation, operands)
            status = "PASS" if abs(result - expected) < 1e-9 else "FAIL"
            logger.debug(f"{status}: {operation}({operands}) = {result} (expected: {expected})")
        except Exception as e:
            logger.debug(f"ERROR: {operation}({operands}) - {e}")

    # 오류 케이스\
    logger.debug("")
    logger.debug("=== 오류 처리 테스트 ===")

    error_cases = [
        ("divide", [10, 0], ZeroDivisionError),
        ("invalid", [1, 2], ValueError),
        ("add", [1], ValueError),
    ]

    for operation, operands, expected_error in error_cases:
        try:
            tool.run(operation, operands)
            logger.debug(f"FAIL: {operation}({operands}) - Expected {expected_error.__name__}")
        except expected_error as e:
            logger.debug(f"PASS: {operation}({operands}) - Caught {expected_error.__name__}: {e}")
        except Exception as e:
            logger.debug(f"FAIL: {operation}({operands}) - Unexpected error: {e}")


if __name__ == "__main__":
    test_arithmetic()
    logger.debug("")
    logger.debug("=== 테스트 완료 ===")
    logger.debug("서버 실행: python -m uvicorn app:app --reload --port 8000")
