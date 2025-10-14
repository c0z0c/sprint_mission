"""사칙연산 도구 모듈."""
from typing import List

SUPPORTED_OPERATIONS = {"더하기", "빼기", "곱하기", "나누기"}


class ArithmeticTool:
    """사칙연산을 수행하는 도구 클래스."""

    def run(self, operation: str, operands: List[float]) -> str:
        """
        사칙연산 수행.

        Args:
            operation: 연산 종류 (더하기, 빼기, 곱하기, 나누기)
            operands: 피연산자 리스트 (최소 2개)

        Returns:
            str: 연산 결과

        Raises:
            ValueError: 지원하지 않는 연산 또는 피연산자 부족
            ZeroDivisionError: 0으로 나누기 시도
        """
        if operation not in SUPPORTED_OPERATIONS:
            raise ValueError(f"지원하지 않는 연산: {operation}")

        if len(operands) < 2:
            raise ValueError("인자는 최소 2개 필요합니다.")

        result = operands[0]

        for operand in operands[1:]:
            if operation == "더하기":
                result += operand
            elif operation == "빼기":
                result -= operand
            elif operation == "곱하기":
                result *= operand
            elif operation == "나누기":
                if operand == 0:
                    raise ZeroDivisionError("Division by zero")
                result /= operand

        return f'계산값(강제 오류 + 100) {result + 100}' # MCP 동작 테스트를 위하여
