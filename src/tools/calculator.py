"""
安全计算器工具 —— 只允许纯数学运算，禁止任何 Python 代码注入
"""
import math
from src.tools.base import BaseTool


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "执行数学计算。支持 + - * / ** () 和 math 库函数（sqrt, sin, cos, log 等）"
    parameters = {
        "expression": {
            "type": "string",
            "description": "数学表达式，如 '1500 * 0.87' 或 'sqrt(144) + 2**10'",
        }
    }
    is_dangerous = False

    # 白名单：只允许这些内置函数和 math 函数
    _SAFE_GLOBALS = {
        "__builtins__": {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "int": int, "float": float, "str": str,
        },
        "math": math,  # sin, cos, sqrt, log 等
    }

    def execute(self, **kwargs) -> str:
        expression = kwargs.get("expression", "")
        if not expression:
            return "错误: 缺少表达式参数"

        # 安全检查：禁止危险字符
        dangerous = ["import", "exec", "eval", "open", "write", "system", "__"]
        expr_lower = expression.lower()
        for d in dangerous:
            if d in expr_lower:
                return f"错误: 表达式包含禁止的关键字 '{d}'。只允许纯数学运算。"

        try:
            result = eval(expression, self._SAFE_GLOBALS, {})
            return str(result)
        except Exception as e:
            return f"计算错误: {e}"
