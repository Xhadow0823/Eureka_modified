from typing import Dict, List
from utils import file_to_string, extract_function_code, remove_import_lines, get_current_file_dir, make_sure_dir_exist, remove_import_lines, get_function_signature
import json
from Chat import MessageDict
import re

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

    def load_env_from_file(self, filepath: str):
        self.env_code_str = file_to_string(filepath)
        return self.env_code_str
    
    def load_obs_function_from_file(self, filepath: str):
        raw_obs_func_code_str = file_to_string(filepath)
        lines = raw_obs_func_code_str.splitlines()
        idx = 0
        for line in lines:
            if line.strip().startswith("def compute_observations("):
                break
            idx += 1
        self.obs_func_code_str = '\n'.join(lines[idx:])

        return self.obs_func_code_str

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
        processed_env_code = self.env_code_str
        processed_env_code = re.sub(r"^.*#[ ]*@LLMRLT:[ ]*(DEF_OBS_HERE).*$", 
                                    self.obs_func_code_str, processed_env_code, flags=re.RegexFlag.M)
        processed_env_code = re.sub(r"^.*#[ ]*@LLMRLT:[ ]*(DEF_REWARD_HERE).*$", 
                                    remove_import_lines(self.reward_func_code_str), processed_env_code, flags=re.RegexFlag.M)
        # TODO: think about the indent problem...
        reward_signature = get_function_signature(self.reward_func_code_str)[0]
        reward_signature = f"self.rew_buf[:], self.reward_components = {reward_signature}"
        processed_env_code = re.sub(r"#[ ]*@LLMRLT:[ ]*(CALL_REWARD_HERE).*$",    # NOTE: replace with the care of indent
                                    reward_signature, processed_env_code, flags=re.RegexFlag.M)
        
        return CodeFile(processed_env_code)
    
    def _get_reward_signature(self):
        # TODO: move get_function_signature() from utils to here!!
        pass

    def _find_all_tag(self):
        'deprecated'
        res = re.findall(r"#[ ]*@LLMRLT:[ ]*([a-zA-Z_]+)$", self.env_code_str, flags=re.RegexFlag.M)
        return res

    pass

if __name__ == "__main__":
    task = "FrankaLift2"
    C = Code()

    C.load_env_from_file(f"LLMRLT/tasks/{task}/env.py")
    C.load_obs_function_from_file(f"LLMRLT/tasks/{task}/obs.py")
    C.load_reward_function_from_file("LLMRLT/logs/FrankaLift/24-04-26-13-37-24.json")

    code = C.gen_env_code()
    print(code)
    code.save(f"./LLMRLT/codes/test/{task}.py")

    exit()

    C.load_reward_function_from_file("LLMRLT/logs/FrankaLift/24-04-26-13-37-24.json")
    reward_function = C.gen_reward_function()
    print(reward_function)
    reward_function.save("./LLMRLT/codes/test/reward_function.py")
    print("Done")
    pass