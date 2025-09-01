import os
import re
import asyncio
import logging
from abc import ABC, abstractmethod
from google import genai
from dotenv import load_dotenv


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

    def __init__(self, model=None, temperature=0.7, thinking_budget=0):
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API"))
        self.model = model or "gemini-2.5-flash"
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        self.files = []

    def upload_file(self, path, name=None, mime_type=None):
        file = self.client.files.upload(
            file=path, config=genai.types.UploadFileConfig(
                mimeType=mime_type,
                displayName=path)
        )
        self.files.append(file)
        return file

    async def generate(self, prompt: str) -> str:        
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=[prompt] + self.files,
            config=genai.types.GenerateContentConfig(
                thinking_config=genai.types.ThinkingConfig(
                    thinking_budget=self.thinking_budget),
                temperature=self.temperature
            ),
        )
        return response
