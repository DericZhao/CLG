from tqdm import tqdm
import json


def load_MATH():
    file_list = ['algebra', 'counting_and_probability', 'number_theory', 'prealgebra']
    data_list = []
    for file in file_list:
        source_file = f"data/MATH/{file}/test.json"
        datas = []
        with open(source_file, "r", encoding="utf-8") as f:
            reader = f.readlines()
            for line in tqdm(reader):
                line = json.loads(line)
                datas.append([line['content'], line['answer']])
        data_list.append(datas)

    return data_list


def load_CARP():
    source_file = "data/carp/test.json"
    data_list = []
    with open(source_file, "r", encoding="utf-8") as f:
        reader = f.readlines()
        for line in tqdm(reader):
            line = json.loads(line)
            data_list.append([line['content'], line['answer']])

    return data_list

def load_GAOKAO():
    source_file = "data/gaokao-mathcloze/gaokao-mathcloze_test.json"
    data_list = []
    with open(source_file, "r", encoding="utf-8") as f:
        reader = f.readlines()
        for line in tqdm(reader):
            line = json.loads(line)
            data_list.append([line['content'], line['answer']])

    return data_list

def load_SAT():
    source_file = "data/sat-math/sat-math_test.json"
    data_list = []
    with open(source_file, "r", encoding="utf-8") as f:
        reader = f.readlines()
        for line in tqdm(reader):
            line = json.loads(line)
            data_list.append([line['content'], line['answer']])

    return data_list


if __name__ == '__main__':
    model = 'qwq-plus'
    prompt = ('You are a math expert. '
              'I am going to give you a math Problem. '
              'At the end of the Solution, when you give your final write it in the form "Final Answer: The final answer is $answer$. '
              "Your thought process should be simpler. Let's think step by setp.")

    carp = load_CARP()
    al, cp, nt, pr = data_MATH = load_MATH()
    sat = load_SAT()
    gaokao = load_GAOKAO()

    prompts = []
    num = 4167
    for line in sat:
        test_dict = {
            "custom_id": f"request-{num}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": f"{model}",
                "messages": [{
                    "role": "user",
                    "content": f"{prompt}\n{line[0]}"
                }]
            }
        }
        num += 1
        prompts.append(test_dict)

    target_file = f'./{model}.jsonl'
    with open(target_file, "w", encoding="utf-8") as f:
        for entry in prompts:
            # 序列化为 JSON 字符串并写入文件
            json_line = json.dumps(entry, ensure_ascii=False)
            f.write(json_line + "\n")  # JSONL 需要换行分隔
        print(f"写入文件完成...")