import asyncio
import logging
import sys

from solipsism.core.context import Context
from solipsism.core.system import System
from solipsism.core.llm import GeminiLLM
from solipsism.core.manager import Manager
# UserContextをインポート
from solipsism.interface.user_context import UserContext
from solipsism.interface.chat_interface import ChatInterface
from solipsism.tools.manager_tools import SendTool, CreateContextTool
from solipsism.tools.file_io import (
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
    CreateDirectoryTool,
    MoveItemTool,
    DeleteItemTool
)

async def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s.%(funcName)s - %(levelname)s - %(message)s',
        stream=sys.stdout,
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting Solipsism application...")

    TOOL_CATALOG = {
        "list_files": ListFilesTool, "read_file": ReadFileTool, "write_file": WriteFileTool,
        "create_directory": CreateDirectoryTool, "move_item": MoveItemTool, "delete_item": DeleteItemTool,
    }

    manager = Manager(tool_catalog=TOOL_CATALOG)

    # 3. UserContextをツリーの最上位ルートとしてセットアップ
    user_context = UserContext(context_id="user", parent_id=None)
    user_context.add_tool(SendTool(manager))
    manager.add_context(user_context)

    # 4. 最初のLLMContextをUserContextの子としてセットアップ
    llm = GeminiLLM()
    llm_system = System()

    for tool_class in TOOL_CATALOG.values():
        llm_system.add_tool(tool_class("./"))
    
    llm_system.add_tool(SendTool(manager))
    llm_system.add_tool(CreateContextTool(manager))

    try:
        # ★★★ 親IDとして 'user' を指定 ★★★
        llm_context = Context(
            llm=llm,
            system=llm_system,
            base_prompt_path="./solipsism/prompts/root_prompt.lpml",
            parent_id=user_context.id
        )
        # ★★★ 親であるUserContextに、子のIDを登録 ★★★
        user_context.child_ids.append(llm_context.id)
        
        logger.info(f"LLM Context '{llm_context.id}' created as a child of '{user_context.id}'.")
        manager.add_context(llm_context)
        
    except Exception as e:
        logger.critical(f"Failed to initialize root Context: {e}", exc_info=True)
        return

    # 5. ChatInterfaceをセットアップ
    chat_interface = ChatInterface(manager, llm_contexts=[llm_context])
    # ChatInterfaceは内部でuser_contextを参照する
    chat_interface.user_context = user_context
        
    # 6. 各タスクを並行実行
    initial_task = (
        f"You are an AI assistant. Your context ID is '{llm_context.id}'. "
        f"Your parent context is the user, with ID '{user_context.id}'. "
        "Start by introducing yourself to your parent (the user)."
    )
    
    llm_task = llm_context.start(initial_task=initial_task, max_turns=100)
    chat_task = chat_interface.start()

    await asyncio.gather(llm_task, chat_task)

    logger.info("Solipsism application has finished its run.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, EOFError):
        print("\nApplication interrupted by user. Exiting...")
    except Exception as e:
        logging.critical(f"An unhandled error occurred: {e}", exc_info=True)
