import asyncio
import logging
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

from dotenv import load_dotenv
from google import genai
from google.api_core import exceptions as google_exceptions

# .envファイルから環境変数を読み込む
load_dotenv()
# ロガーの設定
logger = logging.getLogger(__name__)


@dataclass
class FilePart:
    """ファイルを表すデータクラス。

    Attributes:
        path (str): ファイルへのパス。
        mime_type (str, optional): ファイルのMIMEタイプ。Defaults to None.
    """
    path: str
    mime_type: str = None


@dataclass
class TextPart:
    """テキストを表すデータクラス。

    Attributes:
        text (str): テキストコンテンツ。
    """
    text: str


# Messageのpartsとして許容される型のエイリアス
Part = Union[FilePart, TextPart]


@dataclass
class Message:
    """LLMとのやり取りにおけるメッセージを表すデータクラス。

    Attributes:
        role (str): メッセージの送信者（"user" または "assistant"）。
        parts (List[Part]): メッセージのコンテンツ部分。テキストやファイルを含む。
    """
    role: str
    parts: List[Part] = field(default_factory=list)


# 対話履歴を表す型のエイリアス
History = List[Message]


class BaseLLM(ABC):
    """
    すべてのLLM実装のための抽象基底クラス。

    LLMからの応答を生成するための共通インターフェースを定義します。
    """

    @abstractmethod
    async def generate(self, history: History) -> Message:
        """与えられたメッセージ履歴に基づいてLLMからの応答を生成します。

        このメソッドは、具象クラスで必ず実装されなければなりません。

        Args:
            history (History): LLMに送信する対話履歴。

        Returns:
            Message: LLMからの応答メッセージ。
        """
        pass


class GeminiLLM(BaseLLM):
    """
    Geminiモデルを使用して応答を生成するLLMクラス。

    指数バックオフによるリトライ機能、タイムアウト処理、ファイルアップロード機能
    を備えています。

    Gemini-2.5 pro / flash など思考モードを備えているモデルを想定しています。
    """

    # リトライ対象とするAPIエラーのタプル
    RETRYABLE_EXCEPTIONS = (
        google_exceptions.ResourceExhausted,
        google_exceptions.ServiceUnavailable,
        google_exceptions.InternalServerError,
        google_exceptions.DeadlineExceeded,
    )

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.7,
        thinking_budget: int = -1,
        max_retries: int = 3,
        timeout: float = 180.0,
        backoff_factor: float = 2.0,
    ):
        """GeminiLLMのインスタンスを初期化します。

        Args:
            model (str, optional): 使用するモデル名。
                Defaults to "gemini-1.5-flash".
            temperature (float, optional): 生成時のランダム性を制御する温度。
                Defaults to 0.7.
            thinking_budget (int, optional): 思考時間の上限（秒単位）。
                -1は無制限を意味します。 Defaults to -1.
            max_retries (int, optional): リトライの最大回数。 Defaults to 3.
            timeout (float, optional): API呼び出しのタイムアウト秒数。
                Defaults to 180.0.
            backoff_factor (float, optional): リトライ時の待機時間スケーリング係数。
                Defaults to 2.0.

        Raises:
            ValueError: 環境変数 'GEMINI_API_KEY' が設定されていない場合。
        """
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Environment variable 'GEMINI_API_KEY' not set.")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        self.max_retries = max_retries
        self.timeout = timeout
        self.backoff_factor = backoff_factor
        # アップロード済みファイルをパスをキーにキャッシュする
        self.files: Dict[str, Any] = {}

    def _upload_file(self, path: str, mime_type: str = None) -> Any:
        """ファイルをアップロードし、結果をキャッシュします。

        同期I/O処理のため、非同期メソッドからは直接呼び出さないでください。

        Args:
            path (str): アップロードするファイルのパス。
            mime_type (str, optional): ファイルのMIMEタイプ。 Defaults to None.

        Returns:
            Any: アップロードされたファイルオブジェクト。
        """
        if path in self.files:
            return self.files[path]

        file = self.client.files.upload(
            file=path,
            config=genai.types.UploadFileConfig(
                mimeType=mime_type, displayName=path
            ),
        )
        self.files[path] = file
        return file

    def _convert_message(self, message: Message) -> Dict[str, Any]:
        """汎用的なMessageオブジェクトを、Gemini APIが要求する辞書形式に変換します。

        Args:
            message (Message): 変換対象のMessageオブジェクト。

        Returns:
            Dict[str, Any]: Gemini API形式のメッセージ辞書。
        """
        role = "model" if message.role == "assistant" else "user"
        message_gemini = {"role": role, "parts": []}

        for part in message.parts:
            if isinstance(part, TextPart):
                message_gemini["parts"].append(genai.types.Part(text=part.text))
            elif isinstance(part, FilePart):
                # ファイルアップロードは同期的で時間がかかる処理
                file = self._upload_file(part.path, part.mime_type)
                file_part = genai.types.Part.from_uri(
                    file_uri=file.uri,
                    mime_type=file.mime_type
                )
                message_gemini["parts"].append(file_part)

        return message_gemini

    def _convert_history(self, history: History) -> List[Dict[str, Any]]:
        """汎用的なHistoryを、Gemini APIが要求する形式に変換します。

        Args:
            history (History): 変換対象の対話履歴。

        Returns:
            List[Dict[str, Any]]: Gemini API形式の対話履歴リスト。
        """
        return [self._convert_message(msg) for msg in history]

    async def generate(self, history: History) -> Message:
        """
        タイムアウトと指数バックオフによるリトライロジックを備えたLLM応答生成。

        Args:
            history (History): LLMに送信する対話履歴。

        Returns:
            Message: LLMからの応答。エラーが発生した場合はエラー情報を含む。
        """
        gemini_history = None
        try:
            # ファイルアップロードを含む変換処理は同期的で重いため、
            # asyncio.to_threadで別スレッドで実行し、イベントループをブロックしない。
            gemini_history = await asyncio.to_thread(
                self._convert_history, history
            )
        except Exception as e:
            logger.error(
                f"Failed during history conversion: {e}", exc_info=True
            )
            return Message(
                role="assistant",
                parts=[TextPart(
                    text=f"<error>Failed to process input data: {e}</error>"
                )]
            )

        for attempt in range(self.max_retries + 1):
            try:
                api_call = self.client.aio.models.generate_content(
                    model=f"models/{self.model}",
                    contents=gemini_history,
                    config=genai.types.GenerateContentConfig(
                        temperature=self.temperature,
                        thinking_config=genai.types.ThinkingConfig(
                            thinking_budget=self.thinking_budget
                        ),
                    ),
                )

                response = await asyncio.wait_for(api_call, timeout=self.timeout)
                return Message(
                    role="assistant", parts=[TextPart(text=response.text)]
                )

            except asyncio.TimeoutError:
                msg = (
                    f"LLM generation timed out after {self.timeout} seconds. "
                    f"(Attempt {attempt + 1}/{self.max_retries + 1})"
                )
                logger.warning(msg)

            except self.RETRYABLE_EXCEPTIONS as e:
                msg = (
                    f"A retryable API error occurred: {e.__class__.__name__}. "
                    f"(Attempt {attempt + 1}/{self.max_retries + 1})"
                )
                logger.warning(msg)

            except Exception as e:
                logger.error(
                    f"A non-retryable error occurred: {e}", exc_info=True
                )
                return Message(
                    role="assistant",
                    parts=[TextPart(
                        text=f"<error>A non-retryable error occurred: {e}</error>"
                    )]
                )

            if attempt < self.max_retries:
                delay = self.backoff_factor * (2 ** attempt)
                jitter = random.uniform(0, 1)
                sleep_time = delay + jitter
                logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                await asyncio.sleep(sleep_time)

        final_error_msg = (
            "<error>Failed to get response from LLM after multiple "
            "retries.</error>"
        )
        logger.error(
            "Failed to get response from LLM after %d retries.", self.max_retries
        )
        return Message(role="assistant", parts=[TextPart(text=final_error_msg)])
