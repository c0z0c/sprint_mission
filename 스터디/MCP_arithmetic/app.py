"""MCP 사칙연산 서버."""
import logging
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tools import ArithmeticTool

from ulogger import *
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# # 로깅 설정
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
# )
# logger = logging.getLogger(__name__)

app = FastAPI(
    title="MCP Arithmetic Server",
    description="사칙연산을 제공하는 MCP 서버",
    version="1.0.0",
)

class ToolRequest(BaseModel):
    """도구 호출 요청 스키마."""

    tool: str = Field(..., description="도구 이름 (arithmetic)")
    operation: str = Field(
        ..., description="연산 종류 (add, subtract, multiply, divide)"
    )
    operands: List[float] = Field(..., min_length=2, description="피연산자 리스트")


class ToolResponse(BaseModel):
    """도구 호출 응답 스키마."""

    success: bool
    result: float
    operation: str


@app.post("/invoke", response_model=ToolResponse)
def invoke(req: ToolRequest) -> ToolResponse:
    """
    도구 호출 엔드포인트.

    Args:
        req: 도구 요청 객체

    Returns:
        ToolResponse: 연산 결과

    Raises:
        HTTPException: 도구 오류 또는 연산 실패
    """
    logger.info(
        f"Received request: tool={req.tool}, operation={req.operation}, "
        f"operands={req.operands}"
    )

    if req.tool != "arithmetic":
        logger.error(f"Unknown tool: {req.tool}")
        raise HTTPException(status_code=400, detail=f"Unknown tool: {req.tool}")

    tool = ArithmeticTool()

    try:
        result = tool.run(req.operation, req.operands)
        logger.info(
            f"Operation successful: {req.operation}({req.operands}) = {result}"
        )
        return ToolResponse(success=True, result=result, operation=req.operation)

    except ZeroDivisionError as e:
        logger.error(f"Division by zero: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except ValueError as e:
        logger.error(f"Invalid operation: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/")
def root() -> dict:
    """
    루트 엔드포인트.

    Returns:
        dict: 서비스 정보
    """
    return {
        "service": "MCP Arithmetic Server",
        "version": "1.0.0",
        "endpoint": "/invoke",
        "docs": "/docs",
    }
