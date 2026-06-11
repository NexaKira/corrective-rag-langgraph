"""
Tool 抽象基类 —— 所有工具遵循统一接口
"""

class BaseTool:
    """
    工具基类。每个工具需要定义：
    - name: 工具名称（LLM 用它来指定要用哪个工具）
    - description: 一句话描述工具能做什么
    - parameters: 参数 schema（告诉 LLM 要传什么参数）
    - is_dangerous: 是否需要人工确认（默认 False）
    """

    name: str = ""
    description: str = ""
    parameters: dict = {}
    is_dangerous: bool = False

    def execute(self, **kwargs) -> str:
        """执行工具，返回结果字符串"""
        raise NotImplementedError("子类必须实现 execute 方法")

    def to_prompt_description(self) -> str:
        """生成给 LLM 看的工具描述"""
        return f"- {self.name}: {self.description}\n  参数: {self.parameters}"
