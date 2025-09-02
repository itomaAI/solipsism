import asyncio
import logging
import sys
import inspect

from solipsism.core.context import Context
from solipsism.core.system import System
from solipsism.core.llm import GeminiLLM
from solipsism.core.manager import Manager
from solipsism.core.tool import ToolManager
from solipsism.interface.user_context import UserContext
from solipsism.interface.chat_interface import ChatInterface


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s.%(funcName)s - %(levelname)s - %(message)s',
        stream=sys.stdout,
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting Solipsism application...")

    # 1. 指定されたディレクトリから利用可能なすべてのツールを発見する
    tool_manager = ToolManager(
        tool_directories=["./solipsism/tools", "./workspace/tools"]
    )
    TOOL_CATALOG = tool_manager.get_all_tool_classes()

    # 2. 発見したツールカタログでManagerを初期化する
    manager = Manager(tool_catalog=TOOL_CATALOG)

    # 3. ツールのインスタンス化のための引数を準備する（簡易的な依存性注入）
    tool_init_args = {
        "manager": manager,
        "root_path": "./",
        "tool_manager": tool_manager
    }

    # 4. UserContextをコンテクストツリーのルートとしてセットアップする
    user_context = UserContext(context_id="user", parent_id=None)
    SendTool = TOOL_CATALOG.get("send")
    if SendTool:
        user_context.add_tool(SendTool(manager=tool_init_args["manager"]))
    else:
        logger.critical("SendTool not found in catalog. User cannot send messages.")
        return
    manager.add_context(user_context)

    # 5. 最初のLLMContextをUserContextの子としてセットアップする
    llm = GeminiLLM()
    llm_system = System()

    # 初期コンテクストに基本的なツール群を付与する
    base_tools = [
        "list_files", "read_file", "write_file", "create_directory",
        "move_item", "delete_item", "send", "create_context",
        "register_tool", "list_available_tools"
    ]

    for tool_name in base_tools:
        tool_class = TOOL_CATALOG.get(tool_name)
        if tool_class:
            try:
                # DI引数を使用してツールをインスタンス化する
                sig = inspect.signature(tool_class.__init__)
                params = {}
                for param in sig.parameters.values():
                    if param.name == 'self':
                        continue
                    if param.name in tool_init_args:
                        params[param.name] = tool_init_args[param.name]
                
                instance = tool_class(**params)
                llm_system.add_tool(instance)
            except Exception as e:
                logger.error(
                    f"Failed to instantiate base tool '{tool_name}' for root context: {e}"
                )
        else:
            logger.warning(
                f"Base tool '{tool_name}' not found in catalog. Skipping."
            )

    try:
        # 親IDとして 'user' を指定
        llm_context = Context(
            llm=llm,
            system=llm_system,
            base_prompt_path="./solipsism/prompts/root_prompt.lpml",
            parent_id=user_context.id
        )
        # 親であるUserContextに、子のIDを登録
        user_context.child_ids.append(llm_context.id)
        
        logger.info(
            f"LLM Context '{llm_context.id}' created as a child of '{user_context.id}'."
        )
        manager.add_context(llm_context)
  
    except Exception as e:
        logger.critical(f"Failed to initialize root Context: {e}", exc_info=True)
        return

    # 6. ChatInterfaceをセットアップ
    chat_interface = ChatInterface(manager, llm_contexts=[llm_context])
    chat_interface.user_context = user_context
        
    # 7. 各タスクを並行実行
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