from Prompt import Prompt
from Chat import Chat
from Logger import Logger
from Eval import Evaluation, summary_maker

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

next_prompt = prompt.initial_user # just for the begining

while True:
    logger.info(f"USER PROMPT: {next_prompt}")
    resp = chat.chat(next_prompt)
    logger.info(f"LLM RESPONSE: {resp}")

    logger.info(f"TOTAL TOKEN: {chat.get_total_usage()}")

    logger.info(f"EVALUATION START")
    eval_result = Evaluation(task=task, env_name=env_name, raw_reward_code=resp).em
    eval_summary = {}

    if eval_result.state == "error":
        logger.info(f"EVALUATION ERROR")
        logger.info(f"ERROR MSG: {eval_result.error_msg}")
        next_prompt = prompt.gen_prompt_after_error(eval_result.error_msg)
    else:
        logger.info(f"TENSORBOARD LOG DIR: {eval_result.tensorboard_log_dir}")
        eval_summary = summary_maker(eval_result.tensorboard_log_dir)
        next_prompt = prompt.gen_prompt_after_train(str(eval_summary))
    logger.info(f"EVALUATION FINISH")
    raw = input("optimize ?")
    if not raw.lower().startswith("y"):
        break

logger.info("DEMO END")
chat_logger.save_conversation( chat.get_conversation() )  # save as json