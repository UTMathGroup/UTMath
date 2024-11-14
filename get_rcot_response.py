import json
import sys
import os
from utils.construct_aseq_prompt import make_aseq_prompt
from models.openai_gpt import OpenAIModel
import tqdm
import fire 
from utmath_eval.data import stream_jsonl

problem_path = r'data/UTMath_problem.jsonl'
save_path = r'data/sample_example/gpt-4o_test.jsonl'


def RCoT(sequence, first_llm, second_llm, save_path):
    prompt_reasoning = make_aseq_prompt(sequence, turn=1)
    prompt_coding = make_aseq_prompt(sequence, turn=2)

    msgs = [{'role': 'user', 'content': prompt_reasoning}]
    content, input_tokens_first, output_tokens_first = first_llm.call(msgs)
    msgs.append({'role': 'assistant', 'content': content})
    msgs.append({'role': 'user', 'content': prompt_coding})
    content, input_tokens_second, output_tokens_second = second_llm.call(msgs)
    msgs.append({'role': 'assistant', 'content': content})
    
    temp_dictionary = {
        'task_id': sequence['task_id'],
        'model': (first_llm.model_name, second_llm.model_name), 
        'input_tokens_first': input_tokens_first, 
        'output_tokens_first': output_tokens_first, 
        'input_tokens_second': input_tokens_second, 
        'output_tokens_second': output_tokens_second, 
        'messages': msgs, 
    }
    with open(save_path, 'a',encoding='utf-8') as save_file:
        save_line = json.dumps(temp_dictionary, ensure_ascii=False)
        save_file.write(save_line + '\n')

    return msgs

def entry_point(
    problem_path: str,
    save_path: str,
    model_name: str,
):
    llm = OpenAIModel(model_name)
    task_id_done = set()
    if os.path.exists(save_path):
        for item in stream_jsonl(save_path):
            task_id_done.add(item['task_id'])
    
    for sample in tqdm.tqdm(stream_jsonl(problem_path)):
        if sample['task_id'] in task_id_done:
            continue
        try:
            RCoT(sample, first_llm=llm, second_llm=llm, save_path=save_path)
        except Exception as e:
            print(f'Exception raised in sample {sample["task_id"]}, {e}')

def main():
    fire.Fire(entry_point)
    
if __name__ == '__main__':
    sys.exit(main())