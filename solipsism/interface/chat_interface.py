import sys
import asyncio
import logging
from typing import TYPE_CHECKING, List

from .user_context import UserContext
from ..core.lpml import deparse

if TYPE_CHECKING:
    from ..core.manager import Manager
    from ..core.context import Context

logger = logging.getLogger(__name__)


class ChatInterface:
    """
    ユーザーとの対話を快適に行うための高機能インターフェース。
    内部でUserContextを管理し、コマンドシステムや対話相手の固定機能を提供する。
    """
    def __init__(self, manager: 'Manager', llm_contexts: List['Context']):
        self.manager = manager
        self.llm_contexts = llm_contexts
        
        # 内部でUserContextを生成・管理する
        self.user_context = UserContext(context_id="user")
        # ChatInterface自身はツールを直接持たず、user_contextに委譲する
        
        self.target_context_id = None
        self.state = "IDLE"
        logger.info("ChatInterface initialized.")

    def add_tool_to_user_context(self, tool):
        """内部のUserContextにツールを追加する。"""
        self.user_context.add_tool(tool)

    def _display_help(self):
        print("\n--- Chat Commands ---")
        print("/target <context_id>  - Set the LLM context to chat with.")
        print("/list                   - List all available LLM contexts.")
        print("/help                   - Show this help message.")
        print("/exit                   - Exit the chat interface.")
        print("---------------------\n")

    async def _listen_for_input(self):
        """ユーザーからの標準入力を監視し、コマンド処理やメッセージ送信を行う。"""
        loop = asyncio.get_running_loop()
        send_tool = self.user_context.system.tools.get("send")
        if not send_tool:
            logger.error("SendTool not found in UserContext. Cannot send messages.")
            return

        while self.state == "RUNNING":
            try:
                user_input = await loop.run_in_executor(None, sys.stdin.readline)
                user_input = user_input.strip()

                if not user_input:
                    print(">>> ", end='', flush=True)
                    continue

                # コマンド処理
                if user_input.startswith('/'):
                    parts = user_input.split()
                    command = parts[0]
                    
                    if command == '/target':
                        if len(parts) > 1:
                            self.target_context_id = parts[1]
                            print(f"Target set to '{self.target_context_id}'.")
                        else:
                            print("Usage: /target <context_id>")
                    elif command == '/list':
                        print("\n--- Available LLM Contexts ---")
                        for ctx in self.llm_contexts:
                            print(f"- {ctx.id}")
                        print("----------------------------\n")
                    elif command == '/help':
                        self._display_help()
                    elif command == '/exit':
                        self.state = "TERMINATED"
                        break
                    else:
                        print(f"Unknown command: {command}")

                # メッセージ送信処理
                else:
                    if not self.target_context_id:
                        print("Error: No target context set. Use /target <context_id> first.")
                        continue
                    
                    send_element = {
                        'tag': 'send', 'attributes': {'to': self.target_context_id},
                        'content': f"\n{user_input}\n"
                    }
                    await send_tool.run(send_element)
                
                print(">>> ", end='', flush=True)

            except (EOFError, KeyboardInterrupt):
                self.state = "TERMINATED"
                break
        
        logger.info("User input listener stopped.")

    async def _listen_for_messages(self):
        """UserContextのキューを監視し、LLMからのログやメッセージを表示する。"""
        queue = self.user_context.system.result_queue
        while self.state == "RUNNING":
            try:
                element = await queue.get()
                tag = element.get("tag")
                attrs = element.get("attributes", {})
                content = deparse(element.get("content", "")).strip()

                # カーソル行をクリアし、メッセージを表示してからプロンプトを再表示
                print("\r" + " " * 80 + "\r", end='') # Clear the line

                if tag == "send":
                    from_id = attrs.get("from", "unknown")
                    print(f"[Log from: {from_id}]\n{content}\n")
                else: # outputタグなど
                    print(f"[System Message]\n{deparse([element])}\n")
                
                print(">>> ", end='', flush=True)
                queue.task_done()
            except asyncio.CancelledError:
                break
        
        logger.info("Message listener stopped.")


    async def start(self):
        """チャットインターフェースのメインループを開始する。"""
        if self.state != "IDLE": return
        self.state = "RUNNING"
        
        print("Chat Interface started. Type /help for commands.")
        if len(self.llm_contexts) == 1:
            self.target_context_id = self.llm_contexts[0].id
            print(f"Auto-targeting the only available context: '{self.target_context_id}'")
        print(">>> ", end='', flush=True)

        input_task = asyncio.create_task(self._listen_for_input())
        message_task = asyncio.create_task(self._listen_for_messages())
        
        done, pending = await asyncio.wait(
            [input_task, message_task], return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        
        self.state = "TERMINATED"
        logger.info("ChatInterface has terminated.")
