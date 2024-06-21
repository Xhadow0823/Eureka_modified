# Guide to use

## Install Task for IsaacGymEnv in LLMRCT
1. open `isaacgymenvs/isaacgymenvs/tasks/__init__.py`
2. add this import code `from {env_file} import {task_class}` (e.g. `from .franka_cube_stack_rrr import FrankaCubeStackRRR`)
3. add `"{task_name}": task_name,` in the `isaacgym_task_map` dict (e.g. `"FrankaCubeStackRRR": FrankaCubeStackRRR,`)
4. add config files in the `isaacgymenvs/isaacgymenvs/cfg/task/` and `isaacgymenvs/isaacgymenvs/cfg/train/`, 

## Task Information Preparation
1. prepare your task code (full)
2. prepare your obs function code
3. prepare your task description
4. prepare your sub-task transition function (optional)
5. mark `# @LLMRLT: CALL_REWARD_HERE` in task code
6. mark `# @LLMRLT: DEF_OBS_HERE` in task code
7. mark `# @LLMRLT: DEF_REWARD_HERE` in task code
8. mark `# @LLMRLT: DEF_ST_HERE` in task code (optional) <-(not implement yet)


## LLMRCT, from Input to Run
1. create a folder called {TaskName}/ in LLMRCT/tasks/
2. move task description as `dect.txt` into the task folder
3. move task code as `env.py` into the task folder
4. move task obs function as `obs.py` into the task folder
5. set variables `task` and `env_name` in the `LLMRCT/main.py`
6. run

## Note
+ Sub-task function cannot be called self.FSM()