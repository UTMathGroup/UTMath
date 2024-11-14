# Dataset
In `data/oeis_problem.jsonl`, you'll find all 1053 problems from the UTMath benchmark, covering 9 mathematical domains. Each problem includes over 68 test cases.

# RCoT Inference
We have preconfigured the environment to use OpenAI's API to call GPT-4o and apply the RCoT method for reasoning. After setting up your API key in the environment, you can enter the following command:
```python
python get_rcot_response.py --problem_path=data/utmath_problem.jsonl --save_path={your_save_file_path} --model_name={your_llm_name}
```
For example, after setting up the OpenAI API, you can use the following Python code to call GPT-4o and perform reasoning using the RCoT method.
```python
python get_rcot_response.py --problem_path=data/utmath_problem.jsonl --save_path=data/sample_exapmle/gpt-4o_test.jsonl --model_name=gpt-4o-2024-08-06
```

# Evaluation

You can use this sample as a reference for evaluating on UTMath. Please use the following code:
```python
python utmath_eval/utmath_evaluator.py  --problem_file=data/utmath_problem.jsonl --sample_file={your_sample_file_path}
```

For example, you can directly use our response sample:
The file `data/sample_example/gpt-4o_sample.jsonl` contains responses generated using the RCoT method with GPT-4o on the UTMath benchmark. This sample includes responses to all 1053 problems.
```python
python utmath_eval/utmath_evaluator.py  --problem_file=data/utmath_problem.jsonl --sample_file=data/sample_example/gpt-4o_sample.jsonl

# --with_extra_data=True represents testing both easy and hard cases
# --with_extra_data=None represents testing only easy cases
```

# Acknowledgement

A large part of the codes are borrowed from [HumanEval](https://github.com/openai/human-eval)