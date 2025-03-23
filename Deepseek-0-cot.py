from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import json
import random
from component.template import template_dict
from loguru import logger
import copy

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

    return input_ids


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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
    return tokenizer


def load_llm():
    model_name_or_path = "D:\LLMs\Deepseek-math-7b-Instruct"

    adapter_name_or_path = None

    # 是否使用4bit进行推理，能够节省很多显存，但效果可能会有一定的下降
    load_in_4bit = False
    # 生成超参配置
    max_new_tokens = 500
    top_p = 0.9
    temperature = 0.35
    repetition_penalty = 1.0

    # 加载模型
    logger.info(f'Loading model from: {model_name_or_path}')
    logger.info(f'adapter_name_or_path: {adapter_name_or_path}')
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        load_in_4bit=load_in_4bit,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map='auto',
        quantization_config=None
    ).eval()

    tokenizer = load_tokenizer(model_name_or_path if adapter_name_or_path is None else adapter_name_or_path)

    # if template.stop_word is None:
    #     template.stop_word = tokenizer.eos_token
    # stop_token_id = tokenizer.encode(template.stop_word, add_special_tokens=False)
    # assert len(stop_token_id) == 1
    # stop_token_id = stop_token_id[0]

    return tokenizer, model


def load_MATH():
    file_list = ['algebra', 'counting_and_probability', 'number_theory', 'prealgebra']
    data_list = []
    prompts_list = []
    for file in file_list:
        source_file = f"../data/MATH/{file}/test.json"
        datas = []
        prompts = []
        with open(source_file, "r", encoding="utf-8") as f:
            reader = f.readlines()
            for line in tqdm(reader):
                line = json.loads(line)
                datas.append([line['content'], line['steps'], line['answer']])
                prompts.append(f"Question: {line['content']} {line['steps']} \nAnswer: {line['answer']}\n\n")
        data_list.append(datas)
        prompts_list.append(prompts)

    return data_list, prompts_list


def load_CARP():
    source_file = "../data/carp/test.json"
    data_list = []
    prompts_list = []
    with open(source_file, "r", encoding="utf-8") as f:
        reader = f.readlines()
        for line in tqdm(reader):
            line = json.loads(line)
            data_list.append([line['content'], line['steps'], line['answer']])
            prompts_list.append(f"Question: {line['content']} {line['steps']} \nAnswer: {line['answer']}")

    return data_list, prompts_list


if __name__ == "__main__":
    template_name = 'deepseek0'
    template = template_dict[template_name]

    tokenizer, model = load_llm()

    data_MATH, prompts_MATH = load_MATH()
    data_CARP, prompts_CARP = load_CARP()

    file_list = ['MATH_AL', 'MATH_CP', 'MATH_NT', 'MATH_PR', 'CARP']
    data_list = [data_MATH[0], data_MATH[1], data_MATH[2], data_MATH[3], data_CARP]

    answers_list = []
    for i, data_set in enumerate(data_list):
        data_set_answer = []
        target_file = f'../DeepseekMath/results/0-shot-cot/{file_list[i]}.json'
        for line in tqdm(data_set):
            history = []
            query = line[0].strip()
            input_ids = build_prompt(tokenizer, template, query, copy.deepcopy(history), system=None).to(model.device)
            outputs = model.generate(
                input_ids=input_ids, max_new_tokens=512, do_sample=True,
                top_p=0.9, temperature=0.6
                # eos_token_id=stop_token_id
            )
            outputs = outputs.tolist()[0][len(input_ids[0]):]
            response = tokenizer.decode(outputs)
            response = response.strip().replace(template.stop_word, "").strip()

            print(response)
            data_set_answer.append([response, line[2]])

        with open(target_file, mode="w", encoding="utf-8") as f:
            for line in data_set_answer:
                json.dump(line, f, ensure_ascii=False)
                f.write('\n')

