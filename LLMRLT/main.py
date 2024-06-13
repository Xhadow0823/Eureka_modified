from Prompt import Prompt
from Chat import Chat
from Logger import Logger
from Eval import Evaluation

task = 'FrankaLift2'
env_name = "franka_lift2"

prompt = Prompt(task_name=task, prompt_config={
    "code_output_tip": "prompts/FSM/FSM_code_output_tip.txt",
    "reward_signature": "prompts/FSM/FSM_reward_signature.txt"
})
chat = Chat()
chat_logger = Logger(task_name=task)
logger = chat_logger.getLogger()

logger.info("DEMO START")

logger.info(f"SYSTEM PROMPT: {prompt.initial_system}")
chat.set_system_content(prompt.initial_system)

logger.info(f"USER PROMPT: {prompt.initial_user}")
resp = chat.chat(prompt.initial_user)
logger.info(f"LLM RESPONSE: {resp}")

logger.info(f"TOTAL TOKEN: {chat.get_total_usage()}")

logger.info(f"EVALUATION START")
res = Evaluation(task=task, env_name=env_name, raw_reward_code=resp).em
if res.state == "error":
    logger.info(f"EVALUATION ERROR")
    logger.info(f"ERROR MSG: {res.error_msg}")
else:
    logger.info(f"TENSORBOARD LOG DIR: {res.tensorboard_log_dir}")
logger.info(f"EVALUATION FINISH")


logger.info("DEMO END")
chat_logger.save_conversation( chat.get_conversation() )  # save as json