import ast
from datetime import datetime
import logging
import operator
from typing import Any
from zoneinfo import ZoneInfo

from src.retrieval.hybrid import _GLOBAL_DOCUMENT_STORE, hybrid_retrieve

logger = logging.getLogger(__name__)


async def search_knowledge_base(
    query: str,
    language: str = "auto",
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Search knowledge base using hybrid retrieval."""
    docs = await hybrid_retrieve(query=query, language=language, top_k=top_k)
    return [
        {
            "doc_id": d["doc_id"],
            "chunk_id": d["chunk_id"],
            "text": d["text"],
            "source": d["source"],
            "score": d["score"],
            "language": d["language"],
        }
        for d in docs
    ]


def get_document(doc_id: str) -> dict[str, Any]:
    """Retrieve full document by its doc_id."""
    if doc_id in _GLOBAL_DOCUMENT_STORE:
        doc = _GLOBAL_DOCUMENT_STORE[doc_id]
        return {
            "doc_id": doc_id,
            "title": doc.get("title", ""),
            "full_text": doc.get("text", ""),
            "metadata": doc.get("metadata", {}),
        }
    return {
        "doc_id": doc_id,
        "error": f"Document '{doc_id}' not found in knowledge base.",
    }


def get_current_datetime(timezone_name: str = "Asia/Qatar") -> str:
    """Return current ISO datetime string in the specified timezone."""
    try:
        tz = ZoneInfo(timezone_name)
    except Exception:
        from datetime import timezone as dt_timezone, timedelta
        if "qatar" in timezone_name.lower():
            tz = dt_timezone(timedelta(hours=3))
        else:
            tz = dt_timezone.utc

    now = datetime.now(tz)
    return now.isoformat()


# Safe math evaluation using AST
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_ast_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value}")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)
        return _ALLOWED_OPERATORS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _eval_ast_node(node.operand)
        return _ALLOWED_OPERATORS[op_type](operand)
    raise ValueError(f"Unsupported expression construct: {type(node).__name__}")


def calculate(expression: str) -> str:
    """Safely evaluate mathematical expressions."""
    try:
        # Clean expression
        clean_expr = expression.strip().replace("^", "**").replace("×", "*").replace("÷", "/")
        parsed = ast.parse(clean_expr, mode="eval")
        result = _eval_ast_node(parsed.body)
        return str(result)
    except Exception as e:
        logger.warning("Failed to evaluate expression '%s': %s", expression, e)
        return f"Error: Invalid mathematical expression ({e})"
