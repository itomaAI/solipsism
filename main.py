import asyncio
import logging
import sys

from solipsism.core.context import Context
from solipsism.core.system import System
from solipsism.core.llm import GeminiLLM
from solipsism.tools.file_io import (
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
    CreateDirectoryTool,
    MoveItemTool,
    DeleteItemTool
)

async def main():
    """
    Solipsismアプリケーションのメインエントリポイント。
    コンポーネントを初期化し、メインの対話コンテクストを開始します。
    """
    # 1. アプリケーション全体のロギングを設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s.%(funcName)s - %(levelname)s - %(message)s',
        stream=sys.stdout,
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting Solipsism application...")

    # 2. 主要なコンポーネントをインスタンス化
    llm = GeminiLLM()
    system = System()

    # 3. 利用可能なすべてのツールをインスタンス化し、Systemに登録
    #    各ツールには、セキュリティサンドボックスとして機能するルートパスを指定します。
    logger.info("Initializing and registering tools...")
    tools_to_register = [
        ListFilesTool(root_path="./"),
        ReadFileTool(root_path="./"),
        WriteFileTool(root_path="./"),
        CreateDirectoryTool(root_path="./"),
        MoveItemTool(root_path="./"),
        DeleteItemTool(root_path="./"),
    ]
    for tool in tools_to_register:
        system.add_tool(tool)

    # 4. メインのContextを作成
    #    LLM、System、そしてエージェントの振る舞いを定義する
    #    ベースプロンプトのパスを渡します。
    try:
        context = Context(
            llm=llm,
            system=system,
            base_prompt_path="./solipsism/prompts/root_prompt.lpml",
        )
    except Exception as e:
        logger.critical(f"Failed to initialize Context: {e}", exc_info=True)
        return

    # 5. Contextに対話を開始させるための最初のタスクを与える
    #    ベースプロンプトのルールに基づき、LLMはこのタスクを起点に行動を開始します。
    initial_task = (
        "You are an AI assistant in a new context. "
        "Your goal is to understand the project and fulfill your role. "
        "Please begin your workflow."
    )
    await context.start(
        initial_task=initial_task,
        max_turns=100
    )

    logger.info("Solipsism application has finished its run.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Application interrupted by user. Exiting...")
    except Exception as e:
        logging.critical(f"An unhandled error occurred: {e}", exc_info=True)
