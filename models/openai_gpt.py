import requests
import json
import sys
import os
import re
from openai import OpenAI
from openai.types.chat import ChatCompletion
from tenacity import retry, stop_after_attempt, wait_random_exponential
from loguru import logger

from models import common


price = {
    'gpt-4o-mini-2024-07-18':   {'input': 0.00000015,   'output': 0.0000006}, 
    'gpt-4o-2024-08-06':        {'input': 0.0000025,    'output': 0.000010}, 
    'gpt-3.5-turbo-0125':       {'input': 0.0000015,    'output': 0.000002}, 
    # 'gpt-4o-mini-2024-07-18': {'input': 0.00000015, 'output': 0.0000006}, 
}
class OpenAIModel(common.Model):
    def __init__(
        self,
        model_name: str
    ):
        self.model_name: str = model_name
        self.client = None
        self.setup()

        # cost stats - zero for local models
        self.cost_per_input: float = float(price[model_name]['input'])
        self.cost_per_output: float = float(price[model_name]['output'])

    def check_api_key(self) -> str:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            print("Please set the OPENAI_API_KEY env var")
            sys.exit(1)
        return key

    def setup(self) -> None:
        """
        Check API key, and initialize OpenAI client.
        """
        if self.client is None:
            logger.info(f'Setting model {self.model_name}...')
            key = self.check_api_key()
            self.client = OpenAI(api_key=key)

#    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
    def call(self, messages: list[dict], **kwargs):
        print(messages)
        print(messages[0]['content'])
        temperature = kwargs.get('temperature', common.MODEL_TEMP)
        top_p = kwargs.get('top_p', common.MODEL_TOPP)
        response: ChatCompletion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,  # type: ignore
            temperature=temperature,
            top_p=top_p,
            stream=False,
        )
        # content = data['choices'][0]['message']['content']
        content = response.choices[0].message.content
        logger.info(f'API Call: {messages}\n\tResponse: {content}')
        usage_stats = response.usage
        input_tokens = int(usage_stats.prompt_tokens)
        output_tokens = int(usage_stats.completion_tokens)
        cost = self.calc_cost(input_tokens, output_tokens)
        common.thread_cost.process_cost += cost
        common.thread_cost.process_input_tokens += input_tokens
        common.thread_cost.process_output_tokens += output_tokens

        return content,input_tokens,output_tokens


if __name__ == '__main__':

    llm = OpenAIModel(model_name='gpt-3.5-turbo-0125')
    msgs = [{'role': 'user', 'content': '''Please reason through the following sequence problem and implement your reasoning using code. You need to follow these requirements:

The code must use the Python language.
Use the function signature def solution(x: int), and make sure the code part is in markdown format.
To ensure the code is runnable, please import any necessary libraries.
Provide the reasoning process first.
Use the solution with the lowest time complexity.


Problem:
The definition of a(n) is the sequence of 3-smooth numbers, which are numbers of the form \(2^i \cdot 3^j\) where \(i\) and \(j\) are non-negative integers (i.e., \(i, j \geq 0\)).

Examples:
solution(1) == 1
solution(2) == 2
solution(3) == 3'''}]
    response,input_tokens, output_tokens= llm.call(msgs)
    msgs.append({'role': 'assistant', 'content': response})
    msgs.append({'role': 'user', 'content': 'How dare you?'})
    print(llm.call(msgs))
    print(llm.get_overall_exec_stats()) # 获取当前花了多少钱