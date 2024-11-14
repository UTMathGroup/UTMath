import re
import fire
import tqdm
import sys
import os

from utmath_eval.evaluation import evaluate_functional_correctness
from utmath_eval.data import stream_jsonl


def extract_code(response, keyword):
    """keyword = "def solution" """
    pattern = re.compile(r'```python(.*?)```', re.DOTALL)

    matches = pattern.findall(response)

    for match in matches:
        if keyword in match:
            return match

    return ""


def entry_point(
    sample_file: str,
    k: str = "1,10,100",
    n_workers: int = 12,
    timeout: float = 50.0,
    with_extra_data: bool = True,
    problem_file: str = None,
    out_file: str = None,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    if out_file is None:
        out_file = sample_file.replace('.jsonl', '_utmath_eval.jsonl')
        if not with_extra_data:
            out_file = out_file.replace('.jsonl', '_noExtraData.jsonl')
    if os.path.exists(out_file):
        print(out_file, 'already exists.')
        return

    k = list(map(int, k.split(",")))
    samples = []
 
    for sample in tqdm.tqdm(stream_jsonl(sample_file)):
        messages = sample['messages']
        codes = extract_code(messages[-1]['content'], keyword="def solution")
        if codes == "":
            print(f"{sample['task_id']} has no code snippet.")
        samples.append({
            'task_id': sample['task_id'],
            'completion': codes,
        })
    print(problem_file)
    results = evaluate_functional_correctness(samples, k, n_workers, timeout, with_extra_data, problem_file, out_file)
    print(sample_file, results)


def main():
    fire.Fire(entry_point)

if __name__ == '__main__':
    sys.exit(main())

