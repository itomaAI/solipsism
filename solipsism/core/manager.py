import asyncio
import inspect
import logging
from typing import Any, Dict, List, Optional, Type

from ..tools.manager_tools import CreateContextTool, SendTool
from .context import Context
from .llm import GeminiLLM
from .system import System
from .tool import BaseTool

logger = logging.getLogger(__name__)


class Manager:
    """
    コンテクストのツリー構造を管理し、それらの間の通信を仲介するクラス。
    """

    def __init__(self, tool_catalog: Dict[str, Type[BaseTool]]):
        self.contexts: Dict[str, Any] = {}
        self.tool_catalog = tool_catalog
        logger.info("Manager initialized.")

    def add_context(self, context: Any):
        """コンテクストをマネージャに登録する。"""
        if context.id in self.contexts:
            logger.warning(
                f"Context with ID {context.id} already exists in Manager."
            )
            return
        self.contexts[context.id] = context
        logger.info(f"Context '{context.id}' added to Manager.")

    def get_context(self, context_id: str) -> Optional[Any]:
        """IDを指定して登録済みのコンテクストを取得する。"""
        return self.contexts.get(context_id)

    async def create_new_context(
        self, parent_id: str, task: str, tool_names: List[str],
        custom_id: str = None, llm_config: dict = None, prompt_path: str = None
    ) -> Context:
        """
        新しいLLMコンテクストを動的に生成し、親子関係を構築して起動する。
        """
        logger.info(f"Context creation requested by '{parent_id}'...")

        llm_config = llm_config or {}
        new_llm = GeminiLLM(
            model=llm_config.get("model"),
            temperature=float(llm_config.get("temperature", 0.7))
        )
        new_system = System()

        for tool_name in tool_names:
            if tool_name in self.tool_catalog:
                tool_class = self.tool_catalog[tool_name]
                try:
                    sig = inspect.signature(tool_class.__init__)
                    if 'manager' in sig.parameters:
                        instance = tool_class(manager=self)
                    else:
                        instance = tool_class("./")
                    new_system.add_tool(instance)
                except Exception as e:
                    logger.error(
                        f"Failed to instantiate tool '{tool_name}': {e}",
                        exc_info=True
                    )
            else:
                logger.warning(
                    f"Tool '{tool_name}' not found in catalog. Skipping."
                )

        # These tools are essential and always added with correct dependencies.
        new_system.add_tool(SendTool(self))
        new_system.add_tool(CreateContextTool(self))

        final_prompt_path = (
            prompt_path or "./solipsism/prompts/root_prompt.lpml"
        )

        new_context = Context(
            llm=new_llm,
            system=new_system,
            base_prompt_path=final_prompt_path,
            parent_id=parent_id
        )
        if custom_id:
            if self.get_context(custom_id):
                raise ValueError(f"Context ID '{custom_id}' already exists.")
            new_context.id = custom_id

        parent_context = self.get_context(parent_id)
        if parent_context:
            parent_context.child_ids.append(new_context.id)

        self.add_context(new_context)

        initial_task = (
            f"You were created by context '{parent_id}'. Your parent is your "
            f"sole point of contact.\n"
            f"Your task is: {task}"
        )

        asyncio.create_task(
            new_context.start(initial_task=initial_task, max_turns=100)
        )

        logger.info(
            f"New context '{new_context.id}' has been created and started."
        )
        return new_context

    async def route_message(
        self, from_id: str, to_id: str, element: dict
    ) -> bool:
        """
        親子関係を検証し、許可された通信のみをルーティングする。
        """
        from_context = self.get_context(from_id)
        to_context = self.get_context(to_id)

        if not from_context or not to_context:
            logger.error(
                f"Routing failed: Context not found "
                f"(from: {from_id}, to: {to_id})."
            )
            return False

        is_to_parent = (to_id == from_context.parent_id)
        is_to_child = (to_id in from_context.child_ids)

        if not (is_to_parent or is_to_child):
            logger.warning(
                "Routing DENIED: No parent-child relationship between "
                f"'{from_id}' and '{to_id}'."
            )
            return False

        await to_context.system.result_queue.put(element)
        logger.info(f"Element routed from '{from_id}' to '{to_id}'.")
        return True
