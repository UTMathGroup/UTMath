pot_prompt = """Please reason through the following sequence problem and implement your reasoning using code. You need to follow these requirements:

The code must use the Python language.
Use the function signature def solution(x: int), and make sure the code part is in markdown format.
To ensure the code is runnable, please import any necessary libraries.
Provide the reasoning process first.
Use the solution with the lowest time complexity.


Problem:
{problem}

Examples:
solution({x1}) == {y1}
solution({x2}) == {y2}
solution({x3}) == {y3}"""

rcot_prompt_first = """Please analyze the following sequence problem and provide a detailed reasoning process for the sequence. You need to follow these requirements:

Use the solution with the lowest time complexity.
Not to implement the solution.

Problem:
{problem}

Examples:
a({x1}) == {y1}
a({x2}) == {y2}
a({x3}) == {y3}
"""

rcot_prompt_second = """Please implement the above solution using Python code, adhering to the following requirements:

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


def make_rcot_prompt(sequence, turn=None):
    x1, x2, x3 = sequence['x_list'][0],sequence['x_list'][1],sequence['x_list'][2]
    y1, y2, y3 = sequence['y_list'][0],sequence['y_list'][1],sequence['y_list'][2]
    if turn == 1:
        prompt = rcot_prompt_first.format(
            problem=sequence['problem_statement'],
            x1=x1, x2=x2, x3=x3,
            y1=y1, y2=y2, y3=y3,
        )
    elif turn == 2: 
        prompt = rcot_prompt_second.format(
            x1=x1, x2=x2, x3=x3,
            y1=y1, y2=y2, y3=y3,
        )
    return prompt


def make_pot_prompt(sequence):
    x1, x2, x3 = sequence['x_list'][0],sequence['x_list'][1],sequence['x_list'][2]
    y1, y2, y3 = sequence['y_list'][0],sequence['y_list'][1],sequence['y_list'][2]
    prompt = pot_prompt.format(
        problem=sequence['problem_statement'],
        x1=x1, x2=x2, x3=x3,
        y1=y1, y2=y2, y3=y3,
    )
    return prompt