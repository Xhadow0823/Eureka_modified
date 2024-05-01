import os
import time
import re

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