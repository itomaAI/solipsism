import asyncio
import logging
import os
import random
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from google import genai
from google.api_core import exceptions as google_exceptions

load_dotenv()
logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """
    すべてのLLM実装のための抽象基底クラス。
    LLMからの応答を生成するための共通インターフェースを定義します。
    """

    @abstractmethod
    async def generate(self, prompt_str: str) -> str:
        """
        与えられたプロンプトに基づいてLLMからの応答を生成します。

        このメソッドは、具象クラス（例: MockLLM, GeminiLLM）で必ず実装されなければなりません。

        :param prompt_str: LLMに送信する完全なプロンプト文字列。
        :return: LLMの応答文字列。
        """
        pass


class GeminiLLM(BaseLLM):
    # リトライ対象とするAPIエラーのタプル
    RETRYABLE_EXCEPTIONS = (
        google_exceptions.ResourceExhausted,    # レート制限
        google_exceptions.ServiceUnavailable,   # サーバーが一時的に利用不可
        google_exceptions.InternalServerError,  # 予期せぬサーバーエラー
        google_exceptions.DeadlineExceeded,     # ゲートウェイタイムアウトなど
    )

    def __init__(
        self,
        model=None,
        temperature=0.7,
        thinking_budget=0,
        max_retries: int = 3,
        timeout: float = 180.0,
        backoff_factor: float = 2.0,
    ):
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API"))
        self.model = model or "gemini-2.5-flash"
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        self.files = []
        # リトライとタイムアウトに関する設定
        self.max_retries = max_retries
        self.timeout = timeout
        self.backoff_factor = backoff_factor

    def upload_file(self, path, name=None, mime_type=None):
        file = self.client.files.upload(
            file=path,
            config=genai.types.UploadFileConfig(
                mimeType=mime_type, displayName=path
            ),
        )
        self.files.append(file)
        return file

    async def generate(self, prompt: str) -> str:
        """
        タイムアウトと指数バックオフによるリトライロジックを備えたLLM応答生成。
        """
        for attempt in range(self.max_retries + 1):
            try:
                api_call = self.client.aio.models.generate_content(
                    model=self.model,
                    contents=[prompt] + self.files,
                    config=genai.types.GenerateContentConfig(
                        thinking_config=genai.types.ThinkingConfig(
                            thinking_budget=self.thinking_budget
                        ),
                        temperature=self.temperature,
                    ),
                )

                # API呼び出しにタイムアウトを設定
                response = await asyncio.wait_for(
                    api_call, timeout=self.timeout
                )
                return response.text

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
                # 認証エラー(PermissionDenied)など、リトライしても無駄なエラー
                logger.error(
                    f"A non-retryable error occurred: {e}", exc_info=True
                )
                return f"<error>A non-retryable error occurred: {e}</error>"

            # --- リトライ処理 ---
            if attempt < self.max_retries:
                # 指数バックオフに基づいて待機時間を計算 (1s, 2s, 4s, ...)
                delay = self.backoff_factor * (2**attempt)
                # 複数のインスタンスが同時にリトライするのを防ぐため、ランダムな揺らぎ（ジッター）を追加
                jitter = random.uniform(0, 1)
                sleep_time = delay + jitter

                logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                await asyncio.sleep(sleep_time)

        # すべてのリトライが失敗した場合
        msg = (
            f"Failed to get response from LLM after {self.max_retries} "
            "retries."
        )
        logger.error(msg)
        return (
            "<error>Failed to get response from LLM after multiple "
            "retries.</error>"
        )
