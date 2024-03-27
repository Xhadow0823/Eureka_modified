import os
import time

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