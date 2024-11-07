In data/oeis_problem.jsonl, you'll find all 1053 problems from the UTMath benchmark, covering 9 mathematical domains. Each problem includes over 68 test cases.

The file data/sample_example/gpt-4o_sample.jsonl contains responses generated using the RCoT method with GPT-4o on the UTMath benchmark. This sample includes responses to all 1053 problems.

You can use this sample as a reference for evaluating on UTMath. Please use the following code:
```python
python eval/oesi_evaluator.py  --problem_file=data/oeis_problem.jsonl --sample_file={your_sample_file_path} --with_extra_data=True

# For example, you can directly use our response sample:
python eval/oesi_evaluator.py  --problem_file=data/oeis_problem.jsonl --sample_file=data/sample_example/gpt-4o_sample.jsonl  --with_extra_data=True

# with_extra_data=True represents testing both easy and hard cases
# with_extra_data=None represents testing only easy cases
```

We have preconfigured the environment to use OpenAI's API to call GPT-4o and apply the RCoT method for reasoning. After setting up your API key in the environment, you can enter the following command:
```python
python get_oeis_response.py
```
