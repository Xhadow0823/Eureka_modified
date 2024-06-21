from Prompt import Prompt
from Chat import Chat
from Logger import Logger
from Eval import Evaluation, summary_maker
from Feedback import wait_any_key_for, select
from utils import task_name_to_env_name, register_SIGINT_to, pretty_dict_maker

task = 'FrankaCubeStackRRR'
env_name = task_name_to_env_name(task)
max_epochs = 800

prompt = Prompt(task_name=task, prompt_config={
    "code_output_tip": "prompts/FSM/FSM_code_output_tip.txt",
    "reward_signature": "prompts/FSM/FSM_reward_signature.txt"
})
chat = Chat()
chat_logger = Logger(task_name=task)
logger = chat_logger.getLogger()

logger.info(f"TASK INFO: {env_name}")

logger.info("DEMO START")

logger.info(f"SYSTEM PROMPT: {prompt.initial_system}")
chat.set_system_content(prompt.initial_system)

next_prompt = prompt.initial_user # just for the begining

def caht_SIGINT_handler():
    global logger, chat_logger
    logger.info("DEMO END BY INTERUPT")
    conversation_json = chat_logger.save_conversation(chat.get_conversation())
    logger.info(f"CONVERSATION LOG JSON: {conversation_json}")
register_SIGINT_to(caht_SIGINT_handler)

while True:
    human_feedback = None

    logger.info(f"USER PROMPT: {next_prompt}")
    resp = chat.chat(next_prompt)
    logger.info(f"LLM RESPONSE: {resp}")

    logger.info(f"TOTAL TOKEN: {chat.get_total_usage()}")

    is_any_key_pressed = wait_any_key_for(3, "select action (or system will do evaluation)")
    if is_any_key_pressed:
        selected = select([
            "do evaluation",
            "input human feedback and re-generate again",
            "re-generate again",
        ])
        if selected == 1:
            human_feedback = input("Input your feedback: ")
            logger.info(f"USER SELECT: HUMAN FEEDBACK AND RE-GENERATE")
            # update the user prompt with human feedback
            next_prompt = prompt.gen_prompt_for_regen(next_prompt, human_feedback)
            continue
        elif selected == 2:
            logger.info(f"USER SELECT: RE-GENERATE AGAIN")
            continue

    logger.info(f"EVALUATION START")
    logger.info(f"EVALUATION INFO: max:epochs={max_epochs}")
    eval_result = Evaluation(task=task, env_name=env_name, raw_reward_code=resp, max_epochs=max_epochs).get_result()
    eval_summary = {}
    human_feedback = None

    if eval_result.state == "error":
        logger.info(f"EVALUATION ERROR")
        logger.info(f"ERROR MSG: {eval_result.error_msg}")
    else:
        logger.info(f"TENSORBOARD LOG DIR: {eval_result.tensorboard_log_dir}")
        eval_summary = summary_maker(eval_result.tensorboard_log_dir)
        logger.info(f"EVALUATION SUMMARY: {pretty_dict_maker(eval_summary)}")
    logger.info(f"EVALUATION FINISH")

    is_any_key_pressed = wait_any_key_for(5, "select action (or system will auto improve)")
    if is_any_key_pressed:
        selected = select([
            "auto improve",
            "input human feedback",
            "enough, quit"
        ])
        if selected == 0:
            pass
        elif selected == 1:
            logger.info(f"USER SELECT: HUMAN FEEDBACK TO IMPROVE")
            human_feedback = input("Input your feedback")
            pass
        elif selected == 2:
            logger.info(f"USER SELECT: QUIT MANUALLY")
            print("OK, system exiting...")
            break

    if eval_result.state == "error":
        next_prompt = prompt.gen_prompt_after_error(eval_result.error_msg, human_feedback)
    else:
        next_prompt = prompt.gen_prompt_after_train(str(eval_summary), human_feedback)
    # END OF MAIN WHILE LOOP

logger.info("DEMO END")
conversation_json = chat_logger.save_conversation( chat.get_conversation() )  # save as json
logger.info(f"CONVERSATION LOG JSON: {conversation_json}")