from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import json


def load_llm():
    model_id = "D:\LLMs\Llama-3.1-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    return tokenizer, model


def load_MATH():
    file_list = ['algebra', 'counting_and_probability', 'number_theory', 'prealgebra']
    data_list = []
    for file in file_list:
        source_file = f"../data/MATH/{file}/test.json"
        datas = []
        with open(source_file, "r", encoding="utf-8") as f:
            reader = f.readlines()
            for line in tqdm(reader):
                line = json.loads(line)
                datas.append([line['content'], line['answer']])
        data_list.append(datas)

    return data_list


def load_CARP():
    source_file = "../data/carp/test.json"
    data_list = []
    with open(source_file, "r", encoding="utf-8") as f:
        reader = f.readlines()
        for line in tqdm(reader):
            line = json.loads(line)
            data_list.append([line['content'], line['answer']])

    return data_list


if __name__ == "__main__":
    sys_prompt = ('You are a math expert.'
              'I am going to give you a math Problem. '
              'At the end of the Solution, when you give your final write it in the form "Final Answer: The final answer is $answer$. '
              "Your thought process should be simpler. Let's think step by setp.")

    tokenizer, model = load_llm()

    data_MATH = load_MATH()
    data_CARP = load_CARP()

    file_list = ['MATH_AL', 'MATH_CP', 'MATH_NT', 'MATH_PR', 'CARP']
    data_list = [data_MATH[0], data_MATH[1], data_MATH[2], data_MATH[3], data_CARP]

    answers_list = []
    for i, data_set in enumerate(data_list):
        data_set_answer = []
        target_file = f'../Llama3.1/results/0-shot-cot/{file_list[i]}.json'

        for line in tqdm(data_set):
            messages = [
                {"role": "system", "content": f"{sys_prompt}"},
                {"role": "user", "content": f"{line[0]}\nThe answer is "},
            ]

            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)

            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = model.generate(
                input_ids,
                max_new_tokens=512,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            response = outputs[0][input_ids.shape[-1]:]
            response = tokenizer.decode(response, skip_special_tokens=True)
            print(response)

            data_set_answer.append([response, line[1]])

        with open(target_file, mode="w", encoding="utf-8") as f:
            for line in data_set_answer:
                json.dump(line, f, ensure_ascii=False)
                f.write('\n')

