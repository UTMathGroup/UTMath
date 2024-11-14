"""
For models other than those from OpenAI, use LiteLLM if possible.
"""

import os
import sys
from typing import Literal

import litellm
from litellm.utils import Choices, Message, ModelResponse
from openai import BadRequestError
from tenacity import retry, stop_after_attempt, wait_random_exponential

from models import common
from models.common import Model


class GeminiModel(Model):
    """
    Base class for creating Singleton instances of Gemini models.
    """

    _instances = {}

    def __new__(cls):
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls)
            cls._instances[cls]._initialized = False
        return cls._instances[cls]

    def __init__(
        self,
        name: str,
        cost_per_input: float,
        cost_per_output: float,
    ):
        if self._initialized:
            return
        super().__init__(name, cost_per_input, cost_per_output)
        self._initialized = True

    def setup(self) -> None:
        """
        Check API key.
        """
        self.check_api_key()

    def check_api_key(self) -> str:
        key_name = "GEMINI_API_KEY"
        credential_name = "GOOGLE_APPLICATION_CREDENTIALS"

        gemini_key = os.getenv(key_name)
        credential_key = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not (gemini_key or credential_key):
            print(f"Please set the {key_name} or {credential_name} env var")
            sys.exit(1)
        return gemini_key or credential_key

    def extract_resp_content(self, chat_message: Message) -> str:
        """
        Given a chat completion message, extract the content from it.
        """
        content = chat_message.content
        if content is None:
            return ""
        else:
            return content

    # @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
    def call(
        self,
        messages: list[dict],
    ):
        try:
            response = litellm.completion(
                model=self.model_name,
                messages=messages,
            )
            assert isinstance(response, ModelResponse)
            resp_usage = response.usage
            assert resp_usage is not None
            input_tokens = int(resp_usage.prompt_tokens)
            output_tokens = int(resp_usage.completion_tokens)
            cost = self.calc_cost(input_tokens, output_tokens)

            common.thread_cost.process_cost += cost
            common.thread_cost.process_input_tokens += input_tokens
            common.thread_cost.process_output_tokens += output_tokens

            first_resp_choice = response.choices[0]
            assert isinstance(first_resp_choice, Choices)
            resp_msg: Message = first_resp_choice.message
            content = self.extract_resp_content(resp_msg)
            return content, input_tokens, output_tokens

        except BadRequestError as e:
            if e.code == "context_length_exceeded":
                print("Context length exceeded")
            raise e


class GeminiPro(GeminiModel):
    def __init__(self):
        super().__init__("gemini/gemini-1.0-pro-002")
        self.note = "Gemini 1.0 from Google"


class GeminiFlash(GeminiModel):
    def __init__(self):
        super().__init__("gemini/gemini-1.5-flash")
        self.note = "Gemini 1.5 Flash from Google"


class Gemini15Pro(GeminiModel):
    def __init__(self):
        super().__init__(
            "gemini/gemini-1.5-pro-002",
            0.00000125,
            0.00000500,
        )
        self.note = "Gemini 1.5 from Google"


if __name__ == '__main__':
    llm = Gemini15Pro()
    llm.setup()
    msgs = [{'role': 'user', 'content': 'hello?'}]
    print(llm.call(msgs))
    print(llm.get_overall_exec_stats()) 