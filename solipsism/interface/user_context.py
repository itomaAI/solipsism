import logging
from typing import List, Optional

from ..core.system import System
from ..core.tool import BaseTool

logger = logging.getLogger(__name__)


class UserContext:
    """
    ユーザーを代理する、システム内のメッセージング・エンドポイント。
    LLMContextと同様にツリー構造のノードとして振る舞う。
    """

    def __init__(self, context_id: str, parent_id: Optional[str] = None):
        self.id = context_id
        self.parent_id = parent_id      # 親ID属性を追加
        self.child_ids: List[str] = [] # 子IDリスト属性を追加
        self.system = System()
        self.system.context_id = self.id
        logger.info(f"UserContext '{self.id}' endpoint created.")

    def add_tool(self, tool: BaseTool):
        """自身のSystemにツールを登録する。"""
        self.system.add_tool(tool)