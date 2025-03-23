from tqdm import tqdm
import json


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

def load_GAOKAO():
    source_file = "../data/gaokao-mathcloze/test.json"
    data_list = []
    prompts_list = []
    with open(source_file, "r", encoding="utf-8") as f:
        reader = f.readlines()
        for line in tqdm(reader):
            line = json.loads(line)
            data_list.append([line['content'], line['answer']])
            prompts_list.append(f"Question: {line['content']} \nAnswer: {line['answer']}")

    return data_list, prompts_list

def load_SAT():
    source_file = "../data/sat-math/test.json"
    data_list = []
    prompts_list = []
    with open(source_file, "r", encoding="utf-8") as f:
        reader = f.readlines()
        for line in tqdm(reader):
            line = json.loads(line)
            data_list.append([line['content'], line['answer']])
            prompts_list.append(f"Question: {line['content']} \nAnswer: {line['answer']}")

    return data_list, prompts_list

if __name__ == "__main__":
    sys_prompt = (
        'You are a math expert.\n'
        'I am going to give you a math Problem.\n'
        'At the end of the Solution, when you give your final write it in the form "The answer is $answer$.\n'
        "Let's think step by step.\n\n")


    data_MATH = load_MATH()
    data_GAOKAO, prompts_GAOKAO = load_GAOKAO()
    data_SAT, prompts_SAT = load_SAT()

    file_list = ['MATH_AL', 'MATH_CP', 'MATH_NT', 'MATH_PR', 'GAOKAO', 'SAT']
    data_list = [data_MATH[0], data_MATH[1], data_MATH[2], data_MATH[3], data_GAOKAO, data_SAT]

    for i, data_set in enumerate(data_list):
        data_set_answer = []
        target_file = f'./0-shot-{file_list[i]}.json'

        for line in tqdm(data_set):
            messages = [
                {"role": "system", "content": f"{sys_prompt}"},
                {"role": "user", "content": f"{line[0]}"},
            ]

            # 在这里插入接口代码

            # 调用api接口结束之后返回值用data_set_answer接收
            data_set_answer.append()

        with open(target_file, mode="w", encoding="utf-8") as f:
            for line in data_set_answer:
                json.dump(line, f, ensure_ascii=False)
                f.write('\n')

