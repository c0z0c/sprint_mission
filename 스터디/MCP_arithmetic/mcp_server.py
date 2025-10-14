"""MCP stdio 서버 (Claude Desktop 연동)."""
import asyncio
import json
import logging
import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from tools import ArithmeticTool
from ulogger import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Server 인스턴스 생성
app = Server("arithmetic-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """
    사용 가능한 도구 목록 반환.

    Returns:
        list[Tool]: 도구 정의 리스트
    """
    return [
        Tool(
            name="arithmetic",
            description="사칙연산 수행 (더하기, 빼기, 곱하기, 나누기)",
            inputSchema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["더하기", "빼기", "곱하기", "나누기"],
                        "description": "연산 종류",
                    },
                    "operands": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "description": "피연산자 리스트",
                    },
                },
                "required": ["operation", "operands"],
            },
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """
    도구 호출 핸들러.

    Args:
        name: 도구 이름
        arguments: 도구 입력 인자

    Returns:
        list[TextContent]: 실행 결과
    """
    try:
        if name != "arithmetic":
            raise ValueError(f"Unknown tool: {name}")

        operation = arguments.get("operation")
        operands = arguments.get("operands")

        logger.info(f"Calling {name}: operation={operation}, operands={operands}")

        tool = ArithmeticTool()
        result = tool.run(operation, operands)
        logger.info(f"Result: {result}")

        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "success": True,
                        "operation": operation,
                        "operands": operands,
                        "result": result,
                    },
                    ensure_ascii=False,
                ),
            )
        ]

    except (ValueError, ZeroDivisionError) as e:
        logger.error(f"Error: {e}")
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "success": False,
                        "error": str(e),
                        "operation": arguments.get("operation"),
                        "operands": arguments.get("operands"),
                    },
                    ensure_ascii=False,
                ),
            )
        ]

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "success": False,
                        "error": f"Unexpected error: {str(e)}",
                        "operation": arguments.get("operation"),
                        "operands": arguments.get("operands"),
                    },
                    ensure_ascii=False,
                ),
            )
        ]


async def main():
    """stdio 서버 실행."""
    logger.info("Starting MCP Arithmetic Server (stdio)")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
