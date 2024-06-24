from Code import Code
import subprocess
import shutil
from utils import task_name_to_env_name, tensorboard_log_to_dd
from typing import Literal, Dict, List
from pynput.keyboard import Listener
import time
import re

# Alg.
# 1. search Network Directory: /home/miat/Eureka_modified/policy-2024-06-13_12-09-41/runs/FrankaLift2-2024-06-13_12-09-41/nn
# 2. search Tensorboard Directory: /home/miat/Eureka_modified/policy-2024-06-13_12-09-41/runs/FrankaLift2-2024-06-13_12-09-41/summaries
# 3. search fps step: 190740 fps step and policy inference: 145039 fps total: 122382 epoch: 1/100 frames: 0
# 4. detect Traceback: ...


class EvalMonitor:
    state: Literal[
        "prepare",
        "training",
        "finish",
        "error"
    ] = "prepare"
    training_log_to_trace = None
    last_line_idx = 0
    'this is only available in prepare and train state'

    tensorboard_log_dir = None
    'the path of folder where the tensorboard log is'
    training_progress_str = None
    error_msg = None

    tags_to_eval_data_dict = None
    'Dict[tag_name, List[float]] from tensorboard log'
    eval_summary_str = None
    'summary string from tensorboard log'
    highest_achievable_st = -1
    last_st = -1

    def __init__(self, log_file):
        self.training_log_to_trace = log_file

    def find_tensorboard_dir(self):
        i = 0
        with open(self.training_log_to_trace, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines[self.last_line_idx:]):
                if line.startswith("Tensorboard Directory: "):
                    self.tensorboard_log_dir = line.split("Tensorboard Directory: ")[1].strip()
                    break
        self.last_line_idx += (i)
    def find_training_progress(self):
        i = 0
        with open(self.training_log_to_trace, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines[self.last_line_idx:]):
                if line.startswith("fps step: "):
                    self.training_progress_str = re.search(r"epoch: (\d+/\d+)", line, re.DOTALL).group(1)
                    self.state = "training"
        self.last_line_idx += (i)
    def find_error_msg(self):
        has_found = False
        msg = []
        with open(self.training_log_to_trace, "r") as f:
            lines = f.readlines()
            for line in lines:
                if has_found:
                    msg.append(line)
                    continue
                if line.startswith("Traceback "):
                    has_found = True
        self.error_msg = ''.join(msg)
                    
    def set_error_flag(self):
        self.state = "error"
    def set_finish_flag(self):
        self.state = "finish"

    def get_tags_to_eval_data_dict(self) -> Dict[str, List[float]]:
        if self.tags_to_eval_data_dict == None:
            self.tags_to_eval_data_dict = extract_tags_to_dict(self.tensorboard_log_dir)
        return self.tags_to_eval_data_dict
    def get_eval_summary(self) -> str:
        if self.eval_summary_str == None:
            if self.tags_to_eval_data_dict == None:
                self.get_tags_to_eval_data_dict()
            self.eval_summary_str = ""
            for tag_name, datalist in self.tags_to_eval_data_dict.items():
                min_data, max_data = min(datalist), max(datalist)
                line_for_a_tag = f"{tag_name}: {datalist}, min: {min_data}, max: {max_data}\n"
                self.eval_summary_str += line_for_a_tag
                # collect detailed BSR informations
                if tag_name == "r/BSR":
                    import math
                    self.highest_achievable_st = math.ceil(max_data)
                elif tag_name.startswith("r/state"):
                    self.last_st = max(self.last_st, int(tag_name.replace("r/state", '')))
        return self.eval_summary_str
    def gen_BSR_analysis_str(self) -> str:
        'return sub-task n/m, ...'
        analysis = f"In the evaluation, the highest achievable sub-task index is sub-task {self.highest_achievable_st}, while the last sub-task in the entire task is sub-task {self.last_st}. "
        analysis += f"Please consider redesigning the reward functions for sub-tasks {self.highest_achievable_st} through {self.last_st}. "
        return analysis
    
def Evaluation(**kargs) -> EvalMonitor:
    '''
    input task, env_name, raw_reward_code and max_epochs \n
    output a EvalMonitor object as evaluation result
    '''
    task = kargs["task"]
    env_name = kargs["env_name"]
    raw_reward_code_str = kargs["raw_reward_code"]
    max_epochs = kargs["max_epochs"]

    C = Code()
    C.load_env_from_file(f"LLMRLT/tasks/{task}/env.py")
    C.load_obs_function_from_file(f"LLMRLT/tasks/{task}/obs.py")
    # C.load_reward_function_from_file("LLMRLT/logs/FrankaLift/24-04-26-13-37-24.json")
    C.load_reward_function(raw_reward_code_str)
    env_code_filepath = f"./LLMRLT/codes/test/{env_name}.py"
    C.gen_env_code().save(env_code_filepath)  # TODO: gen env code will raise if reward function not found

    ISAAC_ROOT_DIR = "./isaacgymenvs/isaacgymenvs/"
    eval_log = f"./LLMRLT/codes/test/{task}.log"
    
    shutil.copy(env_code_filepath, ISAAC_ROOT_DIR+f"/tasks/{env_name}.py")
    
    em = None
    with open(eval_log, 'w') as f:
        process = subprocess.Popen(['python', '-u', f'{ISAAC_ROOT_DIR}/train.py',  
                                    'hydra/output=subprocess',
                                    f'task={task}',
                                    f'headless={True}', 'force_render=False',
                                    f'max_iterations={max_epochs}'],
                                    stdout=f, stderr=f)
        em = EvalMonitor(eval_log)
        while em.tensorboard_log_dir == None:
            em.find_tensorboard_dir()
            time.sleep(1)
            if process.poll():
                em.set_error_flag()
                break
        print(f"tensorboard log dir: {em.tensorboard_log_dir}")
        while em.training_progress_str==None or eval(em.training_progress_str) < 1.0:
            em.find_training_progress()
            # print(f"line progress: {em.last_line_idx}")
            print(f"\rtrain progress: {em.training_progress_str}", end='')
            time.sleep(1)
            if process.poll():
                em.set_error_flag()
                break
        print(f"\nfinish")
        
        if em.state == "error":
            em.find_error_msg()
            print(em.error_msg)
        else:
            em.set_finish_flag()
    process.communicate()
    return em  # EvalMonitor

def extract_tags_to_dict(tensorboard_log_dir: str) -> Dict[str, List[float]]:
    'input a tensorboard log dir path, output a summary dict about some tags'
    tag_to_scalars = tensorboard_log_to_dd(tensorboard_log_dir)
    max_epochs = len(tag_to_scalars["info/epochs"])
    epoch_step = max(max_epochs // 10, 1)
    tags_to_summary = ["rewards/iter", ]  # TODO: add more ??
    reward_component_tags = list(filter(lambda k: k.startswith("r/"), tag_to_scalars.keys()))
    tags_to_summary.extend( reward_component_tags )
    summary_dict = {}
    for tag_name in tags_to_summary:
        summary_dict[tag_name] = tag_to_scalars[tag_name][::epoch_step]
    return summary_dict

def summary_maker(tensorboard_log_dir: str) -> str:
    'input a tensorboard log dir path, output a formated summary string for policy feedback'
    summary_dict = extract_tags_to_dict(tensorboard_log_dir)
    summary = ""
    for tag_name, datalist in summary_dict.items():
        line_for_a_tag = f"{tag_name}: {datalist}, min: {min(datalist)}, max: {max(datalist)}\n"
        summary += line_for_a_tag

    return summary

if __name__ == "__main__":
    print()
    import math
    d = extract_tags_to_dict("policy-2024-06-24_16-19-44/runs/FrankaCubeStackRRR-2024-06-24_16-19-45/summaries")
    latest_state = max([int(name.replace("r/state", '')) for name in d.keys() if name.startswith("r/state")])
    max_st = math.ceil(max(d["r/BSR"]))
    print(f"the max sub-task index in the evaluation is sub-task {max_st}, the latest sub-task in this task is {latest_state}")
    print(f"so, re-design th")
    "In the evaluation, the highest achievable sub-task index is sub-task 4, while the last sub-task in the entire task is sub-task 6. Please consider redesigning the reward functions for sub-tasks 4 through 6."
    # s = summary_maker("policy-2024-06-24_16-19-44/runs/FrankaCubeStackRRR-2024-06-24_16-19-45/summaries")
    # print(s)
    exit()

    print( task_name_to_env_name("FrankaLift2") )

    task = "FrankaLift2"
    env_name = "franka_lift2"

    C = Code()
    C.load_env_from_file(f"LLMRLT/tasks/{task}/env.py")
    C.load_obs_function_from_file(f"LLMRLT/tasks/{task}/obs.py")
    C.load_reward_function_from_file("LLMRLT/logs/FrankaLift/24-04-26-13-37-24.json")
    env_code_filepath = f"./LLMRLT/codes/test/{env_name}.py"
    C.gen_env_code().save(env_code_filepath)
    

    ISAAC_ROOT_DIR = "./isaacgymenvs/isaacgymenvs/"
    eval_log = f"./LLMRLT/codes/test/{task}.log"
    
    shutil.copy(env_code_filepath, ISAAC_ROOT_DIR+f"/tasks/{env_name}.py")  # TODO: naming problem...

    with open(eval_log, 'w') as f:
        process = subprocess.Popen(['python', '-u', f'{ISAAC_ROOT_DIR}/train.py',  
                                    'hydra/output=subprocess',
                                    f'task={task}',
                                    f'headless={True}', 'force_render=False',
                                    f'max_iterations={10}'],
                                    stdout=f, stderr=f)
        em = EvalMonitor(eval_log)
        while em.tensorboard_log_dir == None:
            em.find_tensorboard_dir()
            time.sleep(1)
            if process.poll():
                em.set_error_flag()
                break
        print(f"tensorboard log dir: {em.tensorboard_log_dir}")
        while em.training_progress_str==None or eval(em.training_progress_str) < 1.0:
            em.find_training_progress()
            # print(f"line progress: {em.last_line_idx}")
            print(f"\rtrain progress: {em.training_progress_str}", end='')
            time.sleep(1)
            if process.poll():
                em.set_error_flag()
                break
        print(f"\nfinish")
        
        if em.state == "error":
            em.find_error_msg()
            print(em.error_msg)
    
    print( process.communicate() )

    pass