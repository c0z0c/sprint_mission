"""MCP stdio 클라이언트 테스트."""
import asyncio
import json
import logging
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

from ulogger import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@asynccontextmanager
async def create_client():
    """
    MCP 클라이언트 생성 및 서버 연결.

    Yields:
        ClientSession: 연결된 클라이언트 세션
    """
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],
        env=None,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            logger.info("MCP 서버 연결 성공")
            yield session


async def test_arithmetic_operations():
    """사칙연산 도구 전체 테스트."""
    logger.info("=== MCP 사칙연산 테스트 시작 ===\n")

    async with create_client() as session:
        # 1. 도구 목록 조회
        tools = await session.list_tools()
        logger.info(f"사용 가능한 도구: {[tool.name for tool in tools.tools]}\n")

        # 2. 테스트 케이스
        test_cases = [
            {
                "name": "덧셈 (2개)",
                "operation": "더하기",
                "operands": [10, 5],
                "expected": 15.0,
            },
            {
                "name": "덧셈 (3개)",
                "operation": "더하기",
                "operands": [10, 5, 3],
                "expected": 18.0,
            },
            {
                "name": "뺄셈",
                "operation": "빼기",
                "operands": [100, 25, 5],
                "expected": 70.0,
            },
            {
                "name": "곱셈",
                "operation": "곱하기",
                "operands": [2, 3, 4],
                "expected": 24.0,
            },
            {
                "name": "나눗셈",
                "operation": "나누기",
                "operands": [100, 4, 5],
                "expected": 5.0,
            },
            {
                "name": "소수 연산",
                "operation": "곱하기",
                "operands": [3.14, 2],
                "expected": 6.28,
            },
        ]

        passed = 0
        failed = 0

        for test in test_cases:
            try:
                result = await session.call_tool(
                    "arithmetic",
                    arguments={
                        "operation": test["operation"],
                        "operands": test["operands"],
                    },
                )

                # 응답 파싱
                if not result.content or len(result.content) == 0:
                    logger.error(f"✗ FAIL: {test['name']}")
                    logger.error("  Empty response from server")
                    failed += 1
                    logger.info("")
                    continue

                content = result.content[0]
                if not isinstance(content, TextContent):
                    logger.error(f"✗ FAIL: {test['name']}")
                    logger.error(f"  Unexpected content type: {type(content)}")
                    failed += 1
                    logger.info("")
                    continue

                response_text = content.text
                if not response_text:
                    logger.error(f"✗ FAIL: {test['name']}")
                    logger.error("  Empty response text")
                    failed += 1
                    logger.info("")
                    continue

                response = json.loads(response_text)

                if response["success"]:
                    actual = response["result"]
                    expected = test["expected"]

                    if abs(actual - expected) < 1e-6:
                        logger.info(f"✓ PASS: {test['name']}")
                        logger.info(
                            f"  {test['operation']}({test['operands']}) = {actual}"
                        )
                        passed += 1
                    else:
                        logger.error(f"✗ FAIL: {test['name']}")
                        logger.error(
                            f"  Expected: {expected}, Got: {actual}"
                        )
                        failed += 1
                else:
                    logger.error(f"✗ FAIL: {test['name']}")
                    logger.error(f"  Error: {response.get('error')}")
                    failed += 1

            except json.JSONDecodeError as e:
                logger.error(f"✗ FAIL: {test['name']}")
                logger.error(f"  JSON Parse Error: {e}")
                logger.error(f"  Raw response: {result.content[0].text if result.content else 'None'}")
                failed += 1
            except Exception as e:
                logger.error(f"✗ FAIL: {test['name']}")
                logger.error(f"  Exception: {e}")
                failed += 1

            logger.info("")

        # 3. 오류 케이스 테스트
        logger.info("=== 오류 처리 테스트 ===\n")

        error_cases = [
            {
                "name": "0으로 나누기",
                "operation": "나누기",
                "operands": [10, 0],
                "expected_error": "Division by zero",
                "check_json": True,  # JSON 응답 예상
            },
            {
                "name": "잘못된 연산 (입력 검증)",
                "operation": "invalid",
                "operands": [1, 2],
                "expected_error": "Input validation error",
                "check_json": False,  # 텍스트 응답 예상 (MCP 스키마 검증)
            },
            {
                "name": "피연산자 부족 (입력 검증)",
                "operation": "더하기",
                "operands": [1],
                "expected_error": "too short",
                "check_json": False,  # 텍스트 응답 예상 (MCP 스키마 검증)
            },
        ]

        for test in error_cases:
            try:
                result = await session.call_tool(
                    "arithmetic",
                    arguments={
                        "operation": test["operation"],
                        "operands": test["operands"],
                    },
                )

                if not result.content or len(result.content) == 0:
                    logger.error(f"✗ FAIL: {test['name']}")
                    logger.error("  Empty response from server")
                    failed += 1
                    logger.info("")
                    continue

                content = result.content[0]
                if not isinstance(content, TextContent):
                    logger.error(f"✗ FAIL: {test['name']}")
                    logger.error(f"  Unexpected content type: {type(content)}")
                    failed += 1
                    logger.info("")
                    continue

                response_text = content.text
                if not response_text:
                    logger.error(f"✗ FAIL: {test['name']}")
                    logger.error("  Empty response text")
                    failed += 1
                    logger.info("")
                    continue

                # JSON 응답 또는 텍스트 응답 처리
                if test.get("check_json", True):
                    # JSON 응답 예상 (애플리케이션 오류)
                    try:
                        response = json.loads(response_text)
                        if not response["success"] and test["expected_error"] in response.get(
                            "error", ""
                        ):
                            logger.info(f"✓ PASS: {test['name']}")
                            logger.info(f"  예상된 오류 발생: {response['error']}")
                            passed += 1
                        else:
                            logger.error(f"✗ FAIL: {test['name']}")
                            logger.error(f"  예상 오류 미발생: {response}")
                            failed += 1
                    except json.JSONDecodeError:
                        logger.error(f"✗ FAIL: {test['name']}")
                        logger.error(f"  JSON 파싱 실패: {response_text}")
                        failed += 1
                else:
                    # 텍스트 응답 예상 (MCP 입력 검증 오류)
                    if test["expected_error"] in response_text:
                        logger.info(f"✓ PASS: {test['name']}")
                        logger.info(f"  예상된 MCP 검증 오류 발생")
                        passed += 1
                    else:
                        logger.error(f"✗ FAIL: {test['name']}")
                        logger.error(f"  예상 오류 미발생: {response_text}")
                        failed += 1

            except Exception as e:
                logger.error(f"✗ FAIL: {test['name']}")
                logger.error(f"  Exception: {e}")
                failed += 1

            logger.info("")

        # 4. 결과 요약
        total = passed + failed
        logger.info("=" * 50)
        logger.info(f"테스트 완료: {passed}/{total} PASS, {failed}/{total} FAIL")
        logger.info("=" * 50)

        return passed, failed


async def main():
    """메인 실행 함수."""
    try:
        passed, failed = await test_arithmetic_operations()
        if failed > 0:
            exit(1)
    except Exception as e:
        logger.error(f"테스트 실행 오류: {e}")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
