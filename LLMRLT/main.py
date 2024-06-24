from Prompt import Prompt
from Chat import Chat
from Logger import Logger
from Eval import Evaluation, summary_maker
from Feedback import wait_any_key_for, select
from utils import task_name_to_env_name, register_SIGINT_to, pretty_dict_maker


# 注意！ 現在的 PROMPT 超爛，記得改回來再繼續測試並設計新 PROMPT！！

task = 'FrankaCubeStackRRR'
env_name = task_name_to_env_name(task)
max_epochs = 300

prompt = Prompt(task_name=task, prompt_config={
    "code_output_tip":  "prompts/FSM/FSM_code_output_tip.txt",
    "reward_signature": "prompts/FSM/FSM_reward_signature.txt",
    "code_feedback":    "prompts/FSM/R3D_code_feedback.txt",
    "policy_feedback":  "prompts/FSM/R3D_policy_feedback.txt",
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

    is_any_key_pressed = wait_any_key_for(5, "select action (or system will do evaluation)")
    if is_any_key_pressed:
        selected = select([
            "do evaluation",   # (0)
            "input human feedback and re-generate again",  # (1)
            "re-generate again",  # (2)
            "set max_epochs",  # (3)
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
        elif selected == 3:
            try:
                temp = int(input(f"input max_epoch (now is {max_epochs}): "))
                max_epochs = temp
                print(f"max_epochs is {max_epochs}")
            except:
                print("input error")

    logger.info(f"EVALUATION START")
    logger.info(f"EVALUATION INFO: max_epochs={max_epochs}")
    eval_result = Evaluation(task=task, env_name=env_name, raw_reward_code=resp, max_epochs=max_epochs)
    eval_summary = ""
    human_feedback = None

    if eval_result.state == "error":
        logger.info(f"EVALUATION ERROR")
        logger.info(f"ERROR MSG: {eval_result.error_msg}")
    else:
        logger.info(f"TENSORBOARD LOG DIR: {eval_result.tensorboard_log_dir}")
        eval_summary = eval_result.get_eval_summary()
        logger.info(f"EVALUATION SUMMARY: \n{eval_summary}")
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
        BSR_analysis = eval_result.gen_BSR_analysis_str()
        next_prompt = prompt.gen_prompt_after_train(eval_summary, BSR_analysis=BSR_analysis, human_feedback=human_feedback)
    # END OF MAIN WHILE LOOP

logger.info("DEMO END")
conversation_json = chat_logger.save_conversation( chat.get_conversation() )  # save as json
logger.info(f"CONVERSATION LOG JSON: {conversation_json}")