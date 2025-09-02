from abc import ABC, abstractmethod
from asyncio import Queue
from .lpml import Element
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .system import System


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
