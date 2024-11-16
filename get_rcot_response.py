import json
import sys
import os
from utils.construct_aseq_prompt import make_rcot_prompt, make_pot_prompt
from models.openai_gpt import OpenAIModel
from models.gemini import Gemini15Pro
import tqdm
import fire 
from utmath_eval.data import stream_jsonl
from concurrent.futures import ThreadPoolExecutor, as_completed


def RCoT(sequence, first_llm, second_llm, save_path):
    prompt_reasoning = make_rcot_prompt(sequence, turn=1)
    prompt_coding = make_rcot_prompt(sequence, turn=2)

    msgs = [{'role': 'user', 'content': prompt_reasoning}]
    content, input_tokens_first, output_tokens_first, cost1 = first_llm.call(msgs)
    msgs.append({'role': 'assistant', 'content': content})
    msgs.append({'role': 'user', 'content': prompt_coding})
    content, input_tokens_second, output_tokens_second, cost2 = second_llm.call(msgs)
    msgs.append({'role': 'assistant', 'content': content})
    
    temp_dictionary = {
        'task_id': sequence['task_id'],
        'model': (first_llm.model_name, second_llm.model_name), 
        'input_tokens_first': input_tokens_first, 
        'output_tokens_first': output_tokens_first, 
        'input_tokens_second': input_tokens_second, 
        'output_tokens_second': output_tokens_second, 
        'cost': cost1 + cost2,
        'messages': msgs, 
    }
    with open(save_path, 'a',encoding='utf-8') as save_file:
        save_line = json.dumps(temp_dictionary, ensure_ascii=False)
        save_file.write(save_line + '\n')

    return cost1 + cost2

def PoT(sequence, llm, save_path):
    prompt = make_pot_prompt(sequence)
    msgs = [{'role': 'user', 'content': prompt}]
    content, input_tokens, output_tokens, cost = llm.call(msgs)
    msgs.append({'role': 'assistant', 'content': content})
    
    temp_dictionary = {
        'task_id': sequence['task_id'],
        'model': llm.model_name,
        'input_tokens': input_tokens, 
        'output_tokens': output_tokens, 
        'cost': cost,
        'messages': msgs, 
    }
    with open(save_path, 'a',encoding='utf-8') as save_file:
        save_line = json.dumps(temp_dictionary)
        save_file.write(save_line + '\n')

    return cost


def entry_point(
    problem_path: str,
    model_name: str,
    save_path: str = None,
    method: str = 'RCoT',
    max_workers: int = 100,
):
    # llm = OpenAIModel(model_name)
    llm = Gemini15Pro()
    llm.setup()

    if save_path is None:
        save_path = os.path.join(os.path.dirname(problem_path), f"utmath_response_{model_name}_{method}.jsonl")

    print('Saving to', save_path)
    task_id_done = set()
    if os.path.exists(save_path):
        for item in stream_jsonl(save_path):
            task_id_done.add(item['task_id'])
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for sample in tqdm.tqdm(stream_jsonl(problem_path)):
            if sample['task_id'] in task_id_done:
                continue
            if method == 'RCoT':
                futures.append(executor.submit(RCoT, sample, llm, llm, save_path))
            elif method == 'PoT':
                futures.append(executor.submit(PoT, sample, llm, save_path))
        
        all_cost = 0
        for future in tqdm.tqdm(as_completed(futures)):
            try:
                all_cost += future.result()
            except Exception as e:
                print(f'Exception raised in sample {sample["task_id"]}, {e}')
    print(f"The total usage is: \n{llm.get_overall_exec_stats()}")
    print(f"Total cost is:", all_cost)

def main():
    fire.Fire(entry_point)
    
if __name__ == '__main__':
    sys.exit(main())