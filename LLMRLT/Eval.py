from Code import Code
import subprocess
import shutil
from utils import task_name_to_env_name, tensorboard_log_to_dd
from typing import Literal, Dict, List
from pynput.keyboard import Listener
import time
import re

# Alg
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
    training_progress_str = None
    error_msg = None

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

class Evaluation:
    em = None
    def __init__(self, **kargs):
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

        self.em = em  # temp
    def get_result(self) -> EvalMonitor:
        'return self.em as evaluation result'
        return self.em

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
    s = summary_maker("policy-2024-06-13_15-48-20/runs/FrankaLift2-2024-06-13_15-48-21/summaries")
    print(s)
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