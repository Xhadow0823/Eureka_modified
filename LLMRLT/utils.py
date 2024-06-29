# TODO: rewrite all functions here !!!
import os
import time
import re
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_current_file_dir():
    return os.path.dirname(os.path.realpath(__file__))

def get_timestamp():
    return time.strftime("%y-%m-%d-%H-%M-%S", time.localtime())

def make_sure_dir_exist(dir_or_file_path: str):
    dir_path = os.path.dirname(dir_or_file_path)
    os.makedirs(dir_path, exist_ok=True)

def file_to_string(filename: str) -> str:
    with open(filename, 'r') as file:
        return file.read()
    
def extract_function_code(raw: str, remove_imports=False):
    code_string = None
    patterns = [
        r'```python(.*?)```',
        r'```(.*?)```',
        r'"""(.*?)"""',
        r'""(.*?)""',
        r'"(.*?)"',
    ]
    for pattern in patterns:
        code_string = re.search(pattern, raw, re.DOTALL)
        if code_string is not None:
            code_string = code_string.group(1).strip()
            break
    if remove_imports:
        code_string = remove_import_lines(code_string)

    return code_string

def remove_import_lines(code_str: str):
    start_idx = 0
    lines = code_str.splitlines()
    for line in lines:
        if "import" in line or line.strip() == '':
            start_idx += 1
        else:
            break
    return '\n'.join(lines[start_idx:])

# TODO: make this better and move to Code module
def get_function_signature(code_string):  # TODO: this may raise exception if generated code not correct
    'from eureka'
    import ast
    # Parse the code string into an AST
    module = ast.parse(code_string)
    # Find the function definitions
    function_defs = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    # If there are no function definitions, return None
    if not function_defs:
        return None
    # For simplicity, we'll just return the signature of the first function definition
    function_def = function_defs[0]

    input_lst = []
    # Construct the function signature (within object class)
    signature = function_def.name + '(self.' + ', self.'.join(arg.arg for arg in function_def.args.args) + ')'
    for arg in function_def.args.args:
        input_lst.append(arg.arg)
    return signature, input_lst

def task_name_to_env_name(task_name):
    'ApplePieBun -> apple_pie_bun'
    # 使用正則表達式尋找大寫字母並在前面加上_
    snake_case_str = re.sub(r'([a-z])([A-Z])', r'\1_\2', task_name)
    # 將所有字母轉為小寫
    return snake_case_str.lower()

def tensorboard_log_to_dd(tensorboard_log_dir: str) -> defaultdict:
    tag_to_scalars = defaultdict(list)
    
    ea = EventAccumulator(tensorboard_log_dir) # for reading tf events files (from a dir)
    ea.Reload()
    tags = ea.Tags()["scalars"]
    for tag in tags:
        events = ea.Scalars(tag) # event object
        tag_to_scalars[tag] = list(map(lambda e: e.value, events))
    
    return tag_to_scalars

def register_SIGINT_to(func_to_call, then_exit=True):
    'call func_to_call() when catch ctrl+c'
    import signal
    import sys
    def signal_handler(sig, frame):
        print('[Ctrl+C]')
        func_to_call()
        if then_exit:
            sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    # signal.pause()

def pretty_dict_maker(d):
    return str(d).replace("], ", "], \n")


from dataclasses import dataclass
@dataclass
class CliArgs:
    task: str
    max_epochs: int
    max_iters: int
def read_all_cli_args() -> CliArgs:
    'read all cli arguments and return in a CliArgs object (Namespace)'
    import argparse

    parser = argparse.ArgumentParser(prog='LLMRCS', description='LLM based Reward Codesign System', epilog=':)')
    parser.add_argument('-t', '--task', type=str)
    parser.add_argument('-m', '--max_epochs', type=int)
    parser.add_argument('-i', '--max_iters', type=int)
    args = parser.parse_args()
    # print(args)

    return args