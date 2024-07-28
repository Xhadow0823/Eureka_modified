from Prompt import Prompt
from Chat import Chat
from Logger import Logger
from Eval import Evaluation, EvoEvaluation
from Feedback import wait_any_key_for, select
from utils import task_name_to_env_name, register_SIGINT_to, read_all_cli_args, get_timestamp

args = read_all_cli_args()

task = 'FrankaCabinetRRR'  if args.task is None else args.task
env_name = task_name_to_env_name(task)
max_epochs = 300           if args.max_epochs is None else args.max_epochs
max_iters = 99             if args.max_iters is None else args.max_iters
max_samples = 1            if args.max_samples is None else args.max_samples
seed = None                if args.seed is None else int(args.seed)
# print(args)  # for debug

prompt = Prompt(task_name=task, prompt_config={
    "code_output_tip":  "prompts/R3D/R3D_code_output_tip.txt",
    "reward_signature": "prompts/R3D/R3D_reward_signature.txt",
    "code_feedback":    "prompts/R3D/R3D_code_feedback_2.txt",
    "policy_feedback":  "prompts/R3D/R3D_policy_feedback.txt",
})
chat = Chat(seed=seed)
chat_logger = Logger(task_name=task)
logger = chat_logger.getLogger()

task_info_for_logger = {
    "task":        task,
    "env_name":    env_name,
    "llm":         Chat.GPT_MODEL,
    "seed":        seed,
    "max_iters":   max_iters,
    "max_samples": max_samples,
}
logger.info(f"TASK INFO: {task_info_for_logger}")

logger.info(f"DEMO START @ {get_timestamp()}")

logger.info(f"SYSTEM PROMPT: {prompt.initial_system}")
chat.set_system_content(prompt.initial_system)

next_prompt = prompt.initial_user # just for the begining

def caht_SIGINT_handler():
    global logger, chat_logger
    logger.info("DEMO END BY INTERUPT")
    conversation_json = chat_logger.save_conversation(chat.get_conversation())
    logger.info(f"CONVERSATION LOG JSON: {conversation_json}")
register_SIGINT_to(caht_SIGINT_handler)

for iter_idx in range(max_iters):
    human_feedback = None

    logger.info(f"USER PROMPT: {next_prompt}")
    # resp = chat.chat(next_prompt)
    resp_list = chat.chat_evo(next_prompt, max_samples)
    # logger.info(f"LLM RESPONSE: {resp}")
    logger.info(f"LLM RESPONSE: ...OK")

    logger.info(f"TOTAL TOKEN: {chat.get_total_usage()}")

    is_any_key_pressed = wait_any_key_for(5, "select action (or system will do evaluation)", args.prevent_auto)
    if is_any_key_pressed:
        selected = select([
            "do evaluation",   # (0)
            "input human feedback and re-generate again",  # (1)
            "re-generate again",  # (2)
            "set max_epochs and do evaluation",  # (3)
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
                temp = int(input(f"input max_epoch (current is {max_epochs}): "))
                max_epochs = temp
                logger.info(f"USER SELECT: SET max_epochs={max_epochs}")
                print(f"max_epochs is {max_epochs} now")
            except:
                print("input error")
            pass

    logger.info(f"EVALUATION START")
    logger.info(f"EVALUATION INFO: \n\tmax_epochs={max_epochs}\n\tmax_samples={max_samples}")
    # eval_result = Evaluation(task=task, env_name=env_name, raw_reward_code=resp, max_epochs=max_epochs)
    eval_result = EvoEvaluation(task=task, env_name=env_name, raw_reward_code_list=resp_list, max_epochs=max_epochs, max_samples=max_samples, iter_idx=iter_idx)
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
    logger.info(f"BEST RESULT: sample{eval_result.sample_idx}")
    chat.log_message_pair_evo(next_prompt, resp_list[eval_result.sample_idx])

    is_any_key_pressed = wait_any_key_for(5, "select action (or system will auto improve)", args.prevent_auto)
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
        next_prompt = prompt.gen_prompt_after_train(eval_summary, format_dict=eval_result.get_code_feedback_info(), human_feedback=human_feedback)
    # END OF MAIN WHILE LOOP

logger.info(f"DEMO END @ {get_timestamp()}")
conversation_json = chat_logger.save_conversation( chat.get_conversation() )  # save as json
logger.info(f"CONVERSATION LOG JSON: {conversation_json}")