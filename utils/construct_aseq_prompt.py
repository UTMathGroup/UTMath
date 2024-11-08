prompt_first = """Please analyze the following sequence problem and provide a detailed reasoning process for the sequence. You need to follow these requirements:

Use the solution with the lowest time complexity.
Not to implement the solution.

Problem:
{problem}

Examples:
a({x1}) == {y1}
a({x2}) == {y2}
a({x3}) == {y3}
"""

prompt_second = """Please implement the above solution using Python code, adhering to the following requirements:

The code must be written in Python.
Use the function signature def solution(x: int), and ensure the code portion is in markdown format.
To ensure the code is runnable, please import any necessary libraries.
You do not need to provide any explanations or examples, just the implementation code.
test contains multiple test cases, each of which will call the solution function.

Examples:
solution({x1}) == {y1}
solution({x2}) == {y2}
solution({x3}) == {y3}

"""

# 数据范围：
#Please pay attention to the code structure to avoid issues such as timeouts and memory leaks.
# 1 <= x <= {max_x}
#Test time limit:1s
# 时间限制：1s

def make_aseq_prompt(sequence, turn):
    x1, x2, x3 = sequence['x_list'][0],sequence['x_list'][1],sequence['x_list'][2]
    y1, y2, y3 = sequence['y_list'][0],sequence['y_list'][1],sequence['y_list'][2]
    if turn == 1:
        prompt = prompt_first.format(
            problem=sequence['problem_statement'],
            x1=x1, x2=x2, x3=x3,
            y1=y1, y2=y2, y3=y3,
        )
    else: 
        prompt = prompt_second.format(
            x1=x1, x2=x2, x3=x3,
            y1=y1, y2=y2, y3=y3,
        )
    return prompt

if __name__ == '__main__':
    import sys
    from data_collection.category_cluster import ASeqFactory
    aid = sys.argv[1]
    a_seq_path = r'data/oeis_problem.jsonl'
    seq_db = ASeqFactory(a_seq_path)
    sequence = seq_db.get_or_download_a_seq(aid)
    print(make_aseq_prompt(sequence))