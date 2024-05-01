from typing import Dict, List
from utils import file_to_string, extract_function_code, remove_import_lines, get_current_file_dir, make_sure_dir_exist
import json
from Chat import MessageDict

class CodeFile:
    '''this class can be read as string. \ 
    use .save() to save as file'''
    content: str = None
    def __init__(self, content: str):
        self.content = content
    def __str__(self):
        return self.content
    def save(self, filepath: str):
        'save as file'
        make_sure_dir_exist(filepath)
        with open(filepath, "wt") as f:
            f.write(self.content)
    

class Code:
    env_code_str: str = None
    obs_func_code_str: str = None
    reward_func_code_str: str = None

    def load_env(self, env_file_path):
        pass
    def load_reward_function_from_file(self, filepath: str):
        raw_str: str = None
        if ".json" in filepath:
            msg_dict_list: List[MessageDict] = json.loads(file_to_string(filepath))
            for md in msg_dict_list:
                if md["role"] == "assistant":
                    raw_str = md["content"]
                    break
        elif ".log" in filepath or ".txt" in filepath:
            raw_str = file_to_string(filepath)
        else:
            raise Exception("unknown file extension")
        return self.load_reward_function(raw_str)
    def load_reward_function(self, raw_str: str):
        self.reward_func_code_str = extract_function_code(raw_str)
        return self.reward_func_code_str
    
    def gen_reward_function(self):
        return CodeFile(self.reward_func_code_str)
    def gen_env_code(self):
        # replace obs func and rew func with preproc'ed string
        pass
    pass

if __name__ == "__main__":
    C = Code()   
    C.load_reward_function_from_file("LLMRLT/logs/FrankaLift/24-04-26-13-37-24.json")
    reward_function = C.gen_reward_function()
    print(reward_function)
    reward_function.save("./LLMRLT/codes/test/reward_function.py")

    print("Done")
    pass