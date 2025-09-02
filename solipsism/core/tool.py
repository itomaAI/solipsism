import os
import importlib.util
import inspect
import logging
from abc import ABC, abstractmethod
from asyncio import Queue
from .lpml import Element
from typing import TYPE_CHECKING
from typing import Dict, Type, List

if TYPE_CHECKING:
    from .system import System

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """すべてのツールのための抽象基底クラス"""

    def __init__(self):
        self.system: 'System' = None

    @property
    @abstractmethod
    def name(self) -> str:
        """ツールの名前。LPMLのタグ名に対応します。"""
        pass

    @property
    @abstractmethod
    def definition(self) -> str:
        """ツールのに対応するタグの定義。"""
        pass

    @abstractmethod
    async def run(self, element: Element):
        """
        ツールを実行する非同期メソッド。
        実行結果は self.system.result_queue に Element 型で書き込まなければならない。
        :param element: LLMによって出力されたLPML要素
        """
        pass


class ToolManager:
    """
    指定されたディレクトリから利用可能なツールを発見、ロード、管理する。
    """
    def __init__(self, tool_directories: List[str]):
        self.tool_directories = tool_directories
        self.tool_catalog: Dict[str, Type[BaseTool]] = {}
        self.discover_tools()

    def discover_tools(self):
        """
        ツールディレクトリをスキャンし、モジュールをインポートしてツールクラスを登録する。
        """
        logger.info(f"Discovering tools in: {self.tool_directories}")
        for tool_dir in self.tool_directories:
            if not os.path.isdir(tool_dir):
                logger.warning(f"Tool directory not found: {tool_dir}")
                continue

            for filename in os.listdir(tool_dir):
                if filename.endswith(".py") and not filename.startswith("__"):
                    module_name = f"solipsism.tools.{filename[:-3]}"
                    module_path = os.path.join(tool_dir, filename)
                    
                    try:
                        spec = importlib.util.spec_from_file_location(
                            module_name, module_path
                        )
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

                        for name, obj in inspect.getmembers(module):
                            # ▼▼▼ 修正箇所 ▼▼▼
                            if (inspect.isclass(obj) and
                                    issubclass(obj, BaseTool) and
                                    obj is not BaseTool and
                                    not inspect.isabstract(obj)): # <-- この行を追加
                            # ▲▲▲ 修正箇所 ▲▲▲
                                try:
                                    tool_name = obj.name
                                    if tool_name in self.tool_catalog:
                                        logger.warning(
                                            f"Duplicate tool name '{tool_name}' found. Overwriting."
                                        )
                                    self.tool_catalog[tool_name] = obj
                                    logger.info(
                                        f"Discovered tool '{tool_name}' from {filename}"
                                    )
                                except AttributeError:
                                    logger.error(
                                        f"Tool class '{name}' in {filename} does not have a 'name' class attribute."
                                    )

                    except Exception as e:
                        logger.error(
                            f"Failed to load tools from {module_path}: {e}", exc_info=True
                        )

    def get_tool_class(self, name: str) -> Type[BaseTool] | None:
        """
        カタログからツールクラスを名前で取得する。
        """
        return self.tool_catalog.get(name)

    def get_all_tool_classes(self) -> Dict[str, Type[BaseTool]]:
        """
        発見したすべてのツールクラスのカタログを返す。
        """
        return self.tool_catalog
