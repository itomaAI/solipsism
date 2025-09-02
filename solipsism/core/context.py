import asyncio
import logging
import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum, auto
from typing import List, Optional
import re
from .llm import BaseLLM
from .system import System
from .lpml import Element, LPMLTree, deparse, parse, findall

logger = logging.getLogger(__name__)

ConversationHistory = List[Element]


class ContextState(Enum):
    """コンテクストの状態を管理する列挙型"""
    IDLE = auto()
    RUNNING = auto()
    WAITING = auto()
    TERMINATED = auto()


def _generate_id():
    unique_id = uuid.uuid4()
    hash_object = hashlib.sha224(str(unique_id).encode())
    return hash_object.hexdigest()[:8]


class Context:
    """
    会話履歴と状態を管理し、LLMとSystem間の対話ループを統括するクラス。
    ツリー構造におけるノードとしての役割も持つ。
    """

    def __init__(self, llm: BaseLLM, system: System, base_prompt_path: str, parent_id: Optional[str] = None):
        self.id = _generate_id()
        self.parent_id = parent_id  # 親コンテクストのID
        self.child_ids: List[str] = [] # 子コンテクストのIDリスト
        
        self.llm = llm
        self.system = system
        self.system.context_id = self.id
        self.conversation_history: ConversationHistory = []
        self.state = ContextState.IDLE
        self.turn_count = 0

        try:
            with open(base_prompt_path, 'r', encoding='utf-8') as f:
                self.base_prompt = f.read()
            logger.info(f"Base prompt loaded from {base_prompt_path}")
        except FileNotFoundError:
            logger.error(f"Base prompt file not found at: {base_prompt_path}")
            self.base_prompt = "<error>Base prompt not found.</error>"

    @property
    def prompt(self):
        return self.base_prompt + "\n\n" + self.system.get_tool_definitions()

    def _get_timestamp(self) -> str:
        """ISO 8601形式のUTCタイムスタンプを返す。"""
        return datetime.now(timezone.utc).isoformat()

    def _add_to_history(self, tag: str, content: str):
        """タイムスタンプとターン数を付加して履歴に要素を追加する。"""
        element = {
            "tag": tag,
            "attributes": {
                "turn": str(self.turn_count),
                "timestamp": self._get_timestamp()
            },
            "content": "\n" + content + "\n"
        }
        self.conversation_history.append(element)

    def _sanitize_llm_response(self, lpml_string: str) -> str:
        """
        LLMの応答をサニタイズするメインメソッド。
        """
        # assistant tag を除去
        cleaned = re.sub(r'<assistant[^>]+>', '', lpml_string)
        cleaned = re.sub(r'</assistant>', '', cleaned)
        return cleaned

    def _build_full_prompt(self) -> str:
        """
        ベースプロンプトと会話履歴から、LLMに渡す最終的なプロンプト文字列を構築する。
        """
        # ベースプロンプトは常にプロンプトの土台となる
        # 会話履歴はLPML要素のリストなので、deparseで文字列に変換
        _history = sum(
            [[x, '\n\n'] for x in self.conversation_history], [])[:-1]
        history_str = f"<log>\n{deparse(_history)}\n</log>"
        return f"{self.prompt}\n\n{history_str}"

    async def start(self, initial_task: str = None, max_turns: int = 10):
        """対話ループを開始する"""
        if self.state != ContextState.IDLE:
            logger.warning("Context is already running or finished.")
            return

        logger.info(f"Starting context '{self.id}' with initial task: '{initial_task}'")
        self.state = ContextState.RUNNING

        initial_message = f"context id: {self.id}"
        if self.parent_id:
            initial_message += f"\nparent id: {self.parent_id}"
        if initial_task is not None:
            initial_message += f"\nTask: {initial_task}"

        # 最初のタスクを<system>メッセージとして履歴に追加
        self.turn_count = 1
        self._add_to_history("system", initial_message)

        while self.turn_count <= max_turns and self.state != ContextState.TERMINATED:
            logger.info(f"--- Context '{self.id}' | Turn {self.turn_count}/{max_turns} ---")

            # 1. プロンプトを構築し、LLMに応答を生成させる
            prompt_str = self._build_full_prompt()
            llm_response_str = await self.llm.generate(prompt_str)
            
            # 2. LLMの応答をサニタイズし、履歴に追加
            sanitized_response = self._sanitize_llm_response(llm_response_str)
            logger.info(f"Assistant Response:\n{sanitized_response}")
            self._add_to_history("assistant", sanitized_response)

            # 3. 応答をパースして<finish>をチェック
            try:
                response_tree = parse(sanitized_response)
            except Exception as e:
                logger.error(f"Failed to parse LLM response: {e}", exc_info=True)
                response_tree = []

            if findall(response_tree, "finish"):
                logger.info("'<finish>' tag found. Context is terminating.")
                self.state = ContextState.TERMINATED
                continue

            # 4. Systemにツール実行を依頼
            num_tasks = await self.system.process_llm_output(sanitized_response)

            all_results = []
            # 5. ツールが実行されたか、結果キューに既に何かある場合
            if num_tasks > 0 or not self.system.result_queue.empty():
                logger.info("Waiting for tool results...")
                # 少なくとも1つの結果を待つ
                try:
                    # タイムアウトを設けてもよい
                    first_result = await asyncio.wait_for(self.system.result_queue.get(), timeout=30.0)
                    all_results.append(first_result)
                    self.system.result_queue.task_done()
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for tool results.")
            
            # 6. 他にも結果が溜まっていれば、待たずに全て取得
            while not self.system.result_queue.empty():
                result = self.system.result_queue.get_nowait()
                all_results.append(result)
                self.system.result_queue.task_done()

            # 7. ツール結果があれば履歴に追加して次のターンへ
            if all_results:
                logger.info(f"Drained {len(all_results)} result(s) from the queue.")
                _results = sum([[x, '\n\n'] for x in all_results], [])[:-1]
                tool_results_lpml = deparse(_results)

                logger.info(f"System Response (Tool/Message Results):\n{tool_results_lpml}")
                self.turn_count += 1
                self._add_to_history("system", tool_results_lpml)
                continue

            # 8. ツール実行も結果もない場合
            # LLMが待機を意図しているかチェック
            if findall(response_tree, "wait"):
                logger.info("'<wait>' tag found. Context is entering WAITING state until next message.")
                self.state = ContextState.WAITING
                try:
                    # キューにメッセージが来るまで無期限に待機
                    new_message = await self.system.result_queue.get()
                    self.system.result_queue.task_done()
                    logger.info(f"Context '{self.id}' awakened by a new message.")
                    self.state = ContextState.RUNNING
                    
                    # 履歴にメッセージを追加して次のターンへ
                    message_lpml = deparse([new_message])
                    self.turn_count += 1
                    self._add_to_history("system", message_lpml)
                    continue

                except asyncio.CancelledError:
                    logger.warning(f"Wait in context '{self.id}' was cancelled.")
                    self.state = ContextState.TERMINATED
                    continue
            
            # 9. ツール実行もwait/finishタグもない場合 -> 継続を促す
            logger.info("No tools executed and no stop tags found. Prompting assistant to continue.")
            system_message = (
                "The system is waiting for your next action. "
                "Use a tool, or use `<wait>` to pause, or `<finish>` to terminate."
            )
            self.turn_count += 1
            self._add_to_history("system", system_message)
        
        if self.state != ContextState.TERMINATED:
            logger.warning(f"Max turns ({max_turns}) reached. Setting state to TERMINATED.")
            self.state = ContextState.TERMINATED

        logger.info(f"--- Context loop finished for '{self.id}' with state: {self.state.name} ---")
