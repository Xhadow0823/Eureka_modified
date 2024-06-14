from Prompt import Prompt
from Chat import Chat
from Logger import Logger
from Eval import Evaluation, summary_maker
from Feedback import wait_any_key_for, select

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
            human_feedback = input("Input your feedback")
            raise Exception("not implement this yet")
            logger.info(f"USER SELECT: HUMAN FEEDBACK AND RE-GENERATE")
            continue
        elif selected == 2:
            # raise Exception("not implement this yet")
            logger.info(f"USER SELECT: RE-GENERATE AGAIN")
            continue

    logger.info(f"EVALUATION START")
    eval_result = Evaluation(task=task, env_name=env_name, raw_reward_code=resp, max_epochs=100).em
    eval_summary = {}
    human_feedback = None

    if eval_result.state == "error":
        logger.info(f"EVALUATION ERROR")
        logger.info(f"ERROR MSG: {eval_result.error_msg}")
    else:
        logger.info(f"TENSORBOARD LOG DIR: {eval_result.tensorboard_log_dir}")
        eval_summary = summary_maker(eval_result.tensorboard_log_dir)
        logger.info(f"EVALUATION SUMMARY: {eval_summary}")
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
chat_logger.save_conversation( chat.get_conversation() )  # save as json