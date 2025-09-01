from abc import ABC, abstractmethod
from asyncio import Queue
from .lpml import Element


class BaseTool(ABC):
    """すべてのツールのための抽象基底クラス"""

    @property
    @abstractmethod
    def name(self) -> str:
        """ツールの名前。LPMLのタグ名に対応します。"""
        pass

    @abstractmethod
    async def run(self, element: Element, result_queue: Queue):
        """
        ツールを実行する非同期メソッド。
        実行結果はElement型でなければならない。
        :param element: LLMによって出力されたLPML要素
        :param result_queue: 実行結果を格納するキュー
        """
        pass
