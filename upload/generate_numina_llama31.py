import argparse
import json
import pdb
import torch
import jsonlines
from tqdm import tqdm
import util
from vllm import LLM, SamplingParams
from datasets import load_dataset
import sys
import time
import re
MAX_INT = sys.maxsize
INVALID_ANS = "[invalid]"

invalid_outputs = []

def remove_left_right(s):
    left = "\\left{"
    idx = s.find(left)
    if idx >= 0:
        text = s[idx+len(left):]
        right_idx = text.find("\\right")
        return text[:right_idx]
    else:
        return s
    
def check_and_remove_box(s):
    if "$\\boxed{" in s:
        start_idx = s.find("$\\boxed{")
        end_idx = s.find("}$")
        return s[start_idx+len("$\\boxed{"):end_idx]
    elif "$" in s:
        start_idx = s.find("$")
        s = s[start_idx+len("$"):]
        end_idx = s.find("$")
        return s[:end_idx]
    else:
        return s
    
def remove_boxed(s):
    if not s:
        return None
    left = "\\boxed{"
    idx = s.find(left)
    if idx >= 0:
        s = s[idx+len(left):]
        right_idx = s.rfind("}")
        return s[:right_idx]
    else:
        return s
    
def remove_dollar(s):
    if not s:
        return None
    left = "$"
    if s[:len(left)] == left:
        s = s[1:]
        idx = s.find("$")
        return s[:idx]
    else:
        return s
    
def remove_final_dollar(s):
    if not s:
        return None
    left = "$"
    idx = s.find(left)
    if idx >= 0 and s.count("$") == 2:
        s = s[len(left)+idx:]
        right_idx = s.rfind("$")
        return s[:right_idx]
    else:
        return s

def remove_text_box(s):
    if not s:
        return None
    left = "\\text{"
    idx = s.find(left)
    if idx == 0:
        return s[idx+len(left):-1]
    else:
        return s
    
def remove_square_box(s):
    if not s:
        return None
    left = "\\["
    idx = s.find(left)
    if idx >= 0:
        right_idx = s.find("\\]")
        return s[idx+len(left):right_idx]
    else:
        return s
 
def remove_circle_box(s):
    if not s:
        return None
    if len(s) != 3:
        return s 
    left = "("
    idx = s.find(left)
    if idx >= 0:
        return s[len(left):-1]
    else:
        return s

def remove_mbox(s):
    if not s:
        return None
    left = "\\mbox{ "
    idx = s.find(left)
    if idx >= 0:
        return s[:idx].strip()
    else:
        return s    
    
def remove_text_box(s):
    if not s:
        return None
    left = "\\text{"
    idx = s.find(left)
    if idx >= 0:
        return s[len(left):-1]
    else:
        return s
    
def check_math_answer(content,ground_truth):
    #print(content)
    # split_ans = content.split('The answer is')
    split_ans = re.split(r"The answer is|the answer is",content)
    if len(split_ans) > 1:
        # ans = split_ans[-1]
        # if len(ans) == 0:
        #     return False
        # if ans[0] == ":":
        #     ans = ans[1:]
        # ans = ans.strip()
        # #extract_ans_temp = ans.split('.\n')[0]
        # #extract_ans_temp = ans.split('ки')[0]
        # extract_ans_temp = ans
        # #extract_ans_temp = extract_ans_temp.strip()
        # if len(extract_ans_temp)>0 and extract_ans_temp[-1] == '.':
        #     extract_ans = extract_ans_temp[0:-1]
        # else:
        #     extract_ans = extract_ans_temp
        # extract_ans = extract_ans.strip()
        # #print(extract_ans)
        # extract_ans = remove_dollar(extract_ans)
        # extract_ans = remove_dollar(extract_ans)
        # extract_ans = remove_boxed(extract_ans)
        # #print(extract_ans)
        # extract_ans = remove_text_box(extract_ans)
        # extract_ans = remove_square_box(extract_ans)
        # extract_ans = remove_circle_box(extract_ans)
        # extract_ans = remove_final_dollar(extract_ans)
        # if not extract_ans:
        #     return False
        # extract_ans = extract_ans.strip()
        #print(extract_ans)
        extract_ans = remove_boxed(content)
        gt_ans = remove_boxed(util.last_boxed_only_string(ground_truth))
        if util.is_equiv(extract_ans, gt_ans):
            return True
        else:
            return False
    else:
        extract_ans = remove_boxed(content)
        gt_ans = remove_boxed(util.last_boxed_only_string(ground_truth))
        if util.is_equiv(extract_ans, gt_ans):
            return True
        else:
            return False


def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data

def check_math(pred,gt):
    if check_math_answer(pred,gt):
        return 1
    else:
        return 0
    
def test_hendrycks_math(model, data_path, start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=1):
    hendrycks_math_ins = []
    hendrycks_math_answers = []
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
    print('prompt =====', problem_prompt)
    math_prompt = []
    math_gt = []
    count = 0
    dataset_math = load_dataset(args.data_file,split='train')
    for ref in dataset_math:
        math_prompt.append(ref['problem'])
        math_gt.append(ref['solution'])
    # with open("../hallucination_test/data/test/MATH_test_500.jsonl", "r+", encoding="utf8") as f:
    #     for idx, item in enumerate(jsonlines.Reader(f)):
    #         # #temp_instr = problem_prompt.format(instruction=item["instruction"])
    #         # temp_instr = item['problem'] + prompt_template
    #         # #temp_instr = item['instruction']
    #         # hendrycks_math_ins.append(temp_instr)
    #         # solution = item['solution']
    #         # #solution = item['output']
    #         # temp_ans = remove_boxed(util.last_boxed_only_string(solution))
    #         # hendrycks_math_answers.append(temp_ans)
    #         math_prompt.append(item['problem'])
    #         math_gt.append(item['solution'])

    math_prompt = math_prompt[start:end]
    math_gt = math_gt[start:end]
    batch_math = batch_data(math_prompt,batch_size=batch_size)

    stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]
    sampling_params = SamplingParams(n=16,temperature=1.0, top_p=1, max_tokens=2048, seed=42)
    print('sampling =====', sampling_params)

    llm = LLM(model=model,tensor_parallel_size=tensor_parallel_size, enforce_eager=True, dtype = "float16", gpu_memory_utilization=0.85,swap_space=128)
            
    res_completions = []
    save_data = []
    for idx, (prompt) in enumerate(tqdm(batch_math)):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]
        
        prompt = [f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{i}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" for i in prompt]
        # messages_list = []
        # for i in prompt:
        #     messages = [
        #         {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        #         {"role": "user", "content": i}
        #     ]
        #     messages_list.append(messages)
        #     prompt = messages_list
        
        # tokenizer = llm.get_tokenizer()
        # format_prompt = []
        # for i in prompt[:]:
        #     conversations = tokenizer.apply_chat_template(
        #         [i],
        #         tokenize=False,
        #         add_generation_prompt=True, 
        #     )
        #     format_prompt.append(conversations)
        format_prompt = prompt
        completions = llm.generate(format_prompt, sampling_params)
        for i,output in enumerate(completions):
            tmp = {"prompt":math_prompt[count]}
            label_list = []
            prompt_temp = output.prompt
            generated_text = [output.outputs[i].text for i in range(len(output.outputs))]
            #processed_text = process_deepseek(generated_text)
            processed_text = generated_text
            tmp['answers'] = processed_text
            for j in processed_text:
                if check_math(j,math_gt[count]):
                    label_list.append(1)
                else:
                    label_list.append(0)
            #tmp['label'] = label_list
            tmp['gt'] = remove_boxed(util.last_boxed_only_string(math_gt[count]))
            save_data.append(tmp)
            count += 1
            
        with open(args.output_dir,'w',encoding='utf-8') as f:
            json.dump(save_data,f,indent=4,ensure_ascii=False)

    # results = []
    # for idx, (prompt, completion, prompt_answer) in enumerate(zip(hendrycks_math_ins, res_completions, hendrycks_math_answers)):
    #     res = process_results(prompt, completion, prompt_answer)
    #     results.append(res)

    # acc = sum(results) / len(results)
    # #print('len invalid outputs ====', len(invalid_outputs), ', valid_outputs===', invalid_outputs)
    # #print('start===', start, ', end====',end)
    # print('length====', len(results), ', acc====', acc)
    # output_dir = args.output_dir.replace("/","_")
    # with open(f"eval_result/{output_dir}",'w+') as f:
    #     json.dump({"math":acc},f)
    #     json.dump(invalid_outputs, f, indent=4)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='meta-llama/Llama-3.1-8B-Instruct')  # model path
    parser.add_argument("--data_file", type=str, default='AI-MO/NuminaMath-CoT')  # data path
    parser.add_argument("--start", type=int, default=0) #start index
    parser.add_argument("--end", type=int, default=MAX_INT)  # end index
    parser.add_argument("--batch_size", type=int, default=5000)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=1)  # tensor_parallel_size
    parser.add_argument("--output_dir", type=str, default="numina_math/llama31_numina_n16_forth50k_temp1_seed42.json")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print("---------------")
    print("begin to evaluate the MATH dataset.")
    print("---------------")
    test_hendrycks_math(model=args.model, data_path=args.data_file, start=args.start, end=args.end, batch_size=args.batch_size, tensor_parallel_size=args.tensor_parallel_size)
