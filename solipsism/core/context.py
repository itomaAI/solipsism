import asyncio
import hashlib
import logging
import re
import uuid
from datetime import datetime, timezone
from enum import Enum, auto
from typing import List, Optional

from .llm import BaseLLM, History, Message, TextPart
from .lpml import Element, deparse, findall, parse
from .system import System

logger = logging.getLogger(__name__)

ConversationHistory = List[Element]


class ContextState(Enum):
    """コンテクストの状態を管理する列挙型"""
    IDLE = auto()
    RUNNING = auto()
    WAITING = auto()
    TERMINATED = auto()


def _generate_id():
    """SHA224に基づくユニークな8文字のIDを生成する。"""
    unique_id = uuid.uuid4()
    hash_object = hashlib.sha224(str(unique_id).encode())
    return hash_object.hexdigest()[:8]


class Context:
    """
    会話履歴と状態を管理し、LLMとSystem間の対話ループを統括するクラス。
    ツリー構造におけるノードとしての役割も持つ。
    """

    def __init__(self, llm: BaseLLM, system: System, base_prompt_path: str,
                 parent_id: Optional[str] = None):
        self.id = _generate_id()
        self.parent_id = parent_id
        self.child_ids: List[str] = []

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
        """システムプロンプトとツール定義を結合したプロンプトを返す。"""
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
        """LLMの応答から<assistant>タグを除去し、クリーンアップする。"""
        cleaned = re.sub(
            r'<assistant[^>]*>', '', lpml_string, flags=re.IGNORECASE
        )
        cleaned = re.sub(
            r'</assistant>', '', cleaned, flags=re.IGNORECASE
        )
        return cleaned.strip()

    def _build_llm_history(self) -> History:
        """
        内部の会話履歴(LPML)をLLMが要求するHistory形式に変換する。
        """
        # ベースプロンプト+ツール定義を、対話全体の指示として最初のuserメッセージに設定
        history: History = [
            Message(role="user", parts=[TextPart(text=self.prompt)])
        ]

        # 内部のLPML形式の会話履歴を、Messageオブジェクトのリストに変換
        for element in self.conversation_history:
            # LPML要素全体を文字列化してコンテンツとする
            content_str = deparse([element])

            if element["tag"] == "assistant":
                role = "assistant"
            elif element["tag"] == "system":
                # システムからの通知（ツール結果など）はuserロールとして扱う
                role = "user"
            else:
                logger.warning(f"Unsupported tag '{element['tag']}' skipped.")
                continue

            history.append(
                Message(role=role, parts=[TextPart(text=content_str)])
            )

        return history

    async def start(self, initial_task: Optional[str] = None,
                    max_turns: int = 10, turn_sleep: float = 5.0):
        """対話ループを開始する。"""
        if self.state != ContextState.IDLE:
            logger.warning("Context is already running or finished.")
            return

        logger.info(
            f"Starting context '{self.id}' with initial task: '{initial_task}'"
        )
        self.state = ContextState.RUNNING

        initial_message = f"context id: {self.id}"
        if self.parent_id:
            initial_message += f"\nparent id: {self.parent_id}"
        if initial_task is not None:
            initial_message += f"\nTask: {initial_task}"

        self.turn_count = 1
        self._add_to_history("system", initial_message)

        while self.turn_count <= max_turns and self.state != ContextState.TERMINATED:
            turn_info = f"--- Context '{self.id}' | Turn {self.turn_count}/{max_turns} ---"
            logger.info(turn_info)

            if self.turn_count > 1 and turn_sleep > 0:
                logger.info(
                    f"Sleeping for {turn_sleep} second(s) before next turn."
                )
                await asyncio.sleep(turn_sleep)

            # 1. LLMに渡すための対話履歴を構築し、応答を生成させる
            history = self._build_llm_history()
            llm_response_message = await self.llm.generate(history)

            # 2. LLMの応答(Message)からテキストを抽出し、サニタイズして履歴に追加
            if (llm_response_message.parts and
                    isinstance(llm_response_message.parts[0], TextPart)):
                llm_response_str = llm_response_message.parts[0].text
            else:
                logger.error("LLM response is empty or in an unexpected format.")
                llm_response_str = "<error>LLM response is empty or invalid.</error>"

            sanitized_response = self._sanitize_llm_response(llm_response_str)
            logger.info(f"Assistant Response:\n{sanitized_response}")
            self._add_to_history("assistant", sanitized_response)

            # 3. 応答をパースして<finish>をチェック
            try:
                response_tree = parse(sanitized_response)
            except Exception as e:
                logger.error(f"Failed to parse LLM response: {e}", exc_info=True)
                response_tree = []

            if findall(response_tree, "finish") and False:
                logger.info("'<finish>' tag found. Context is terminating.")
                self.state = ContextState.TERMINATED
                continue

            # 4. Systemにツール実行を依頼
            num_tasks = await self.system.process_llm_output(sanitized_response)

            # 5. ツール実行結果を待機・収集
            all_results = []
            if num_tasks > 0 or not self.system.result_queue.empty():
                logger.info("Waiting for tool results...")
                try:
                    first_result = await asyncio.wait_for(
                        self.system.result_queue.get(), timeout=30.0
                    )
                    all_results.append(first_result)
                    self.system.result_queue.task_done()
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for the first tool result.")

                while not self.system.result_queue.empty():
                    result = self.system.result_queue.get_nowait()
                    all_results.append(result)
                    self.system.result_queue.task_done()

            # 6. ツール結果があれば履歴に追加して次のターンへ
            if all_results:
                logger.info(
                    f"Drained {len(all_results)} result(s) from the queue."
                )
                results_elements = sum(
                    [[x, '\n\n'] for x in all_results], []
                )[:-1]
                tool_results_lpml = deparse(results_elements)

                logger.info(
                    "System Response (Tool/Message Results):\n"
                    f"{tool_results_lpml}"
                )
                self.turn_count += 1
                self._add_to_history("system", tool_results_lpml)
                continue

            # 7. <wait>タグをチェック
            if findall(response_tree, "wait"):
                logger.info(
                    "'<wait>' tag found. Context is WAITING for the next message."
                )
                self.state = ContextState.WAITING
                try:
                    new_message = await self.system.result_queue.get()
                    self.system.result_queue.task_done()
                    logger.info(f"Context '{self.id}' awakened by a new message.")
                    self.state = ContextState.RUNNING

                    self.turn_count += 1
                    self._add_to_history("system", deparse([new_message]))
                    continue
                except asyncio.CancelledError:
                    logger.warning(f"Wait in context '{self.id}' was cancelled.")
                    self.state = ContextState.TERMINATED
                    continue

            # 8. ツール実行も待機/終了タグもない場合、継続を促す
            logger.info(
                "No tools executed and no stop tags found. "
                "Prompting assistant to continue."
            )
            system_message = (
                "The system is waiting for your next action. "
                "Use a tool, or use `<wait>` to pause, or `<finish>` to terminate."
            )
            self.turn_count += 1
            self._add_to_history("system", system_message)

        if self.state != ContextState.TERMINATED:
            logger.warning(
                f"Max turns ({max_turns}) reached. Setting state to TERMINATED."
            )
            self.state = ContextState.TERMINATED

        logger.info(
            f"--- Context loop finished for '{self.id}' with state: "
            f"{self.state.name} ---"
        )
