# 改写成对话的形式，截断器

# 问题抽取 + 一阶段推理 + 二阶段推理
# 全局COT + 一阶段推理 + 二阶段推理
# 一阶段推理 + 二阶段推理 + 关键步骤COT
# 全局COT + 关键步骤推理 + 最终答案
# 关键步骤推理 + 全局COT

# 二阶段推理 his=[问题抽取, 一阶段推理]
# 全局COT his=[问题抽取]
# 关键步骤COT his=[一阶段推理 + 二阶段推理]
# 最终结果 his=[全局COT, 关键步骤COT]
# 问题抽取
import json
import re
import random

source_file = f'./data/carp_en/train_gpt4o_final3.json'
target_file = f'./Finetune_sft/SFT_data.json'

system = 'You are a math teacher.\n'
prompts1 = []
prompts2 = []
prompts3 = []
prompts4 = []
prompts5 = []
prompts6 = []

with open(source_file, 'r', encoding='utf-8') as f:
    reader = f.readlines()
    for line in reader:
        data = json.loads(line)
        # 二阶段推理
        third_step = {}
        third_step['instruction'] = 'In the second stage of reasoning, what can be inferred?'
        third_step['input'] = (f'Math problem: {data["question"]}\n'
                               f'{data["math_expressions"]}'
                               f'{data["conditions"]}'
                               f'Intermediate steps: {" ".join(data["final"][0][:-1])}\n'
                               f'Problem Target: {data["final"][0][-1]}')
        third_step['output'] = '\n'.join(data['final'][2]) + " The final answer is : " + data['final'][1][0]
        third_step['system'] = system

        third_step['history'] = []

        # 问题抽取
        expression = {}
        expression['instruction'] = 'Please extract the math expressions and given conditions from the math problem.'
        expression['input'] = f'Problem: {data["question"]}'
        expression['output'] = data['math_expressions'] + data['conditions']
        expression['system'] = system
        prompts5.append(expression)
        third_step['history'].append([expression['instruction'] + '\n' + expression['input'], expression['output']])

        # 计算目标抽取
        target ={}
        target['instruction'] = 'Please extract the calculation object from the math problem.'
        target['input'] = f'Problem: {data["question"]}'
        target['output'] = data['question_target']
        target['system'] = system
        prompts6.append(target)
        third_step['history'].append([target['instruction'] + '\n' + target['input'], target['output']])

        # 一阶段推理
        single_step = {}
        single_step['instruction'] = 'In the first stage of reasoning, what can be inferred?'
        single_step['input'] = (f'Math problem: {data["question"]}\n'
                                f'{data["math_expressions"]}'
                                f'{data["conditions"]}')
        single_step['output'] = '\n'.join(data['intermediate'][2])
        single_step['system'] = system
        third_step['history'].append([single_step['instruction'] + '\n' + single_step['input'], single_step['output']])

        prompts1.append(third_step)


        # 全局cot
        cot = {}
        cot['instruction'] = 'Please summarize the overall thought process for solving the problem.'
        cot['input'] = (f'Problem: {data["question"]}\n'
                        f'{data["math_expressions"]}')
        cot['output'] = data['steps']
        cot['system'] = system
        cot['history'] = []
        cot['history'].append([expression['instruction'] + '\n' + expression['input'], expression['output']])

        prompts2.append(cot)

        # 关键步骤推理
        key_steps = {}
        key_steps['instruction'] = 'Please provide the key steps to solve the following math problem.'
        key_steps['input'] = (f'Problem: {data["question"]} \n'
                              f'{data["question_target"]}\n')
        second_step = []
        for inter in data['intermediate'][2]:
            second_step.append(inter.split('<SO>')[1])

        final_step = []
        for final1 in data['final'][2]:
            final_step.append(final1.split("<SO>")[1])
        key_steps['output'] = f"First: {' '.join(second_step)} Finally: {' '.join(final_step)}"
        key_steps['system'] = system
        key_steps['history'] = []

        key_steps['history'].append([single_step['instruction'] + '\n' + single_step['input'], single_step['output']])
        key_steps['history'].append([third_step['instruction'] + '\n' + third_step['input'], third_step['output']])

        prompts3.append(key_steps)

        # 最终答案
        final_answer = {}
        final_answer['instruction'] = 'Based on the key steps above and the overall thought process the final answer is:'
        final_answer['input'] = (f"{data['steps']}\n"
                                 f"First: {' '.join(second_step)}\n"
                                 f"Finally: {' '.join(final_step)}\n"
                                 )
        final_answer['output'] = data['answer']
        final_answer['system'] = system
        final_answer['history'] = []
        final_answer['history'].append([cot['instruction'] + '\n' + cot['input'], cot['output']])
        final_answer['history'].append([key_steps['instruction'] + '\n' + key_steps['input'], key_steps['output']])

        prompts4.append(final_answer)

combined = prompts1 + prompts2 + prompts3 + prompts4 + prompts5 + prompts6
random.shuffle(combined)
with open(target_file, "w", encoding="utf-8") as f:
    json.dump(combined, f, ensure_ascii=False, indent=4)
    print(f"写入文件完成...")