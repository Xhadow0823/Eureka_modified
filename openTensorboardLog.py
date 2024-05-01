from eureka.utils.file_utils import load_tensorboard_logs


path = "eureka/outputs/eureka/2024-03-27_21-11-19/policy-2024-03-27_23-26-58/runs/FrankaCubeStackGPT-2024-03-27_23-26-58/summaries"
tensorboard_logs = load_tensorboard_logs(path)
content = ""
epoch_freq = 1
for metric in tensorboard_logs:  # the tensorboard_logs is a default_dict
    if "/" not in metric:
        metric_cur = ['{:.2f}'.format(x) for x in tensorboard_logs[metric][::epoch_freq]]
        metric_cur = "[SKIP]"
        metric_cur_max = max(tensorboard_logs[metric])
        metric_cur_mean = sum(tensorboard_logs[metric]) / len(tensorboard_logs[metric])
        if "consecutive_successes" == metric:
            # successes.append(metric_cur_max)
            breakpoint()
        metric_cur_min = min(tensorboard_logs[metric])
        if metric != "gt_reward" and metric != "gpt_reward":
            if metric != "consecutive_successes":
                metric_name = metric 
            else:
                metric_name = "task_score"
            content += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"                    
        else:  # NOTE: 這應該是縮排對錯了吧...，這個寫法會重複出現 ground-truth score
            # Provide ground-truth score when success rate not applicable
            if "consecutive_successes" not in tensorboard_logs:
                content += f"ground-truth score: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"     

print( content )