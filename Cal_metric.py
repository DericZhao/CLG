from transformers import AutoTokenizer, AutoConfig, AddedToken
from transformers import AutoModelForCausalLM
import torch
from loguru import logger
import copy
from tqdm import tqdm
import json
import os
import platform
import csv
import sys
sys.path.append("../../")
from component.template import template_dict
import re



def load_files(file_path):
    cands = []
    refs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = f.readlines()
        for line in reader:
            data = json.loads(line)
            pattern = r"answer is (.+)"
            match = re.search(pattern, data[0], re.IGNORECASE)
            if match:
                cands.append(match.group(1))
            else:
                cands.append(data[0])
            refs.append(data[1])
    return cands, refs


def build_prompt(tokenizer, template, query, history, system=None):
    template_name = template.template_name
    system_format = template.system_format
    user_format = template.user_format
    assistant_format = template.assistant_format
    system = system if system is not None else template.system

    history.append({"role": 'user', 'message': query})
    input_ids = []

    # setting system information
    if system_format is not None:
        # system信息不为空
        if system is not None:
            system_text = system_format.format(content=system)
            input_ids = tokenizer.encode(system_text, add_special_tokens=False)
    # concat conversation
    for item in history:
        role, message = item['role'], item['message']
        if role == 'user':
            message = user_format.format(content=message, stop_token=tokenizer.eos_token)
        else:
            message = assistant_format.format(content=message, stop_token=tokenizer.eos_token)
        tokens = tokenizer.encode(message, add_special_tokens=False)
        input_ids += tokens

    input_ids = torch.tensor([input_ids], dtype=torch.long)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    return input_ids, attention_mask


def load_tokenizer(model_name_or_path):
    # config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    # 加载tokenzier
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False
        # llama不支持fast
        # use_fast=False if config.model_type == 'llama' else True
    )

    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
    return tokenizer


def main(candidates, targets):
    if template.stop_word is None:
        template.stop_word = tokenizer.eos_token
    stop_token_id = tokenizer.encode(template.stop_word, add_special_tokens=False)
    assert len(stop_token_id) == 1
    stop_token_id = stop_token_id[0]

    correct = 0
    wrong = 0
    test_text = []
    history = []
    for candidate, target in tqdm(zip(candidates, targets), total=len(targets)):
        query = (f"The student answer is {candidate} \n"
                 f"The reference answer is {target} \n" )

        input_ids, attention_mask = build_prompt(tokenizer, template, query, copy.deepcopy(history), system=None)
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)

        outputs = model.generate(
            input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True,
            top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty, attention_mask=attention_mask,
            eos_token_id=stop_token_id, pad_token_id=tokenizer.eos_token_id
        )
        outputs = outputs.tolist()[0][len(input_ids[0]):]
        response = tokenizer.decode(outputs)
        response = response.strip().replace(template.stop_word, "").strip().lower()
        test_text.append(response)

        # print(f'cand={candidate}, target={target}, response={response}')

        if response.strip() == 'yes':
            correct += 1
        else:
            wrong += 1
            print(f'cand={candidate}, target={target}, response={response}')

    return test_text, format((correct/len(targets)) * 100, '.4f'), f'{correct}/{len(targets)}'


def aaa1(candi, targets):
    correct = 0
    wrong = 0
    for cand, tar in zip(candi, targets):
        if cand == tar:
            correct += 1
        else:
            wrong += 1

    print(correct)
    print(wrong)


if __name__ == '__main__':
    model_name = 'QwenMath'
    file_type = ['0-shot-cot']
    file_list = ['MATH_AL', 'MATH_CP', 'MATH_NT', 'MATH_PR', 'CARP']

    sys = platform.system()
    if sys == "Windows":
        model_name_or_path = 'D:\LLMs\Llama-3.1-8B-Instruct'
    else:
        model_name_or_path = '/home/LLM/Llama-3.1-8B-Instruct'

    template_name = 'examiner'

    template = template_dict[template_name]
    # 是否使用4bit进行推理，能够节省很多显存，但效果可能会有一定的下降
    load_in_4bit = False
    # 生成超参配置
    max_new_tokens = 50
    top_p = 0.1
    temperature = 0.2
    repetition_penalty = 1.0

    # 加载模型
    logger.info(f'Loading model from: {model_name_or_path}')

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        load_in_4bit=load_in_4bit,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map='auto',
        quantization_config=None
    ).eval()

    tokenizer = load_tokenizer(model_name_or_path)

    target_file = f'./{model_name}/results.csv'

    # with open(target_file, 'a+', encoding='utf-8', newline='') as f:
    #     fp_write = csv.writer(f)
    #     fp_write.writerow(['name', 'Acc', 'Correct/Total'])
    #     for i, type in enumerate(file_type):
    #         for j, file in enumerate(file_list):
    #             file_path = f'./{model_name}/results/{file_type[i]}/{file_list[j]}.json'
    #             print(file_path)
    #             cands, refs = load_files(file_path)
    #             # aaa1(cands1, refs)
    #             test_text_forward, Acc_forward, correct = main(cands, refs)
    #
    #             fp_write.writerow([
    #                 file_type[i] + ' ' + file_list[j],
    #                 Acc_forward,
    #                 correct
    #             ])
    #             print(correct)






