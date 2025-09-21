import asyncio
import logging
from typing import Dict, List, Optional

from .lpml import Element, deparse, findall, parse
from .tool import BaseTool

logger = logging.getLogger(__name__)


class System:
    """LLMの出力に応じたツールの実行を管理するクラス"""

    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.result_queue: asyncio.Queue[Element] = asyncio.Queue()
        self.context_id: Optional[str] = None  # どのコンテクストに属しているかを保持
        logger.info("System initialized.")

    def add_tool(self, tool: BaseTool):
        """システムにツールを登録し、ツールにシステムインスタンスへの参照を渡す"""
        if tool.name in self.tools:
            logger.warning(
                f"Tool '{tool.name}' is already registered. Overwriting.")

        tool.system = self  # ツールにシステムインスタンスを渡す
        self.tools[tool.name] = tool
        logger.info(f"Tool '{tool.name}' has been added.")

    def get_tool_definitions(self):
        definitions = ""
        for name, tool in self.tools.items():
            definitions += tool.definition + "\n\n"
        return definitions.strip()

    async def process_llm_output(self, lpml_string: str) -> int:
        """
        LLMの出力をパースし、対応するツールを実行する。
        実行をスケジュールしたタスクの数を返す。
        """
        logger.info("Processing LLM output for tool execution...")
        try:
            exclude = ["define_tag", "rule", "send", "code"]
            exclude += [key for key in self.tools.keys() if key != 'create_context']
            tree = parse(lpml_string, exclude=exclude)
        except Exception as e:
            logger.error(f"Failed to parse LPML string: {e}", exc_info=True)
            return 0

        tasks_to_run = []
        for tag_name, tool in self.tools.items():
            tool_elements = findall(tree, tag_name)
            for element in tool_elements:
                logger.info(f"Found tool tag: <{tag_name}>. Scheduling execution.")
                task = asyncio.create_task(tool.run(element))
                tasks_to_run.append(task)

        if tasks_to_run:
            num_tasks = len(tasks_to_run)
            logger.info(f"Scheduled {num_tasks} tool(s) to run in the background.")
            return num_tasks
        else:
            logger.info("No tool tags found in the LLM output.")
            return 0

    async def get_tool_results_as_lpml(self) -> Optional[str]:
        """
        結果キューに溜まったツール実行結果をLPML文字列として取得する。
        キューが空の場合はNoneを返す。
        """
        if self.result_queue.empty():
            return None

        results: List[Element] = []
        while not self.result_queue.empty():
            result = await self.result_queue.get()
            results.append(result)
            self.result_queue.task_done()

        if results:
            logger.info(f"Drained {len(results)} tool result(s) from the queue.")
            _results = sum([[x, '\n\n'] for x in results], [])[:-1]
            return deparse(_results)

        return None
