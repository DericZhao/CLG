from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import json
import random


examples = {
    'math': [
        (
            "The sum of two numbers is 6. The difference of their squares is 12. What is the positive difference of the two numbers?",
            """Call the two numbers $x$ and $y$.\nWe are given that $x+y = 6$ and $x^2 - y^2 = 12$.\nBecause $x^2 - y^2$ factors into $(x+y)(x-y)$, we can substitute in for $x+y$, giving $6(x-y) = 12$, or $x-y = 2$.\nThe answer is 2"""
        ),
        (
            "Which integer is closest to the cube root of 100?",
            """Either 4 or 5 is closest to $\\sqrt[3]{100}$, since $4^3=64$ and $5^3=125$. Since $4.5^3=91.125<100$, $\\sqrt[3]{100}$ is closer to 5 than to 4.\nThe answer is 5"""
        ),
        (
            "What is the value of $(x - y)(x + y)$ if $x = 10$ and $y = 15$?",
            """$(x-y)(x+y)=(10-15)(10+15) = (-5)(25) = -125$.\nThe answer is -125"""
        ),
        (
            "If $g(x) = 3x + 7$ and $f(x) = 5x - 9$, what is the value of $f(g(8))$?",
            """$g(8)=3(8)+7=24+7=31$. Thus, $f(g(8))=f(31)=5(31)-9=155-9=146$.\nThe answer is 146"""
        ),
        (
            "What is the greatest possible positive integer value of $x$ if $\displaystyle\frac{x^4}{x^2} < 10$?",
            """On the left-hand side, $x^2$ cancels, reducing the inequality to $x^2<10$. Since  $3^2=9<10$ while $4^2=16>10$, the greatest possible value of $x$ is 3$.\nThe answer is 3"""
        ),
        (
            "A scale drawing of a park shows that one inch represents 800 feet. A line segment in the drawing that is 4.75 inches long represents how many feet?",
            """Each inch of the 4.75-inch line segment represents 800 feet, so the whole line segment represents $4.75\\times800=\\frac{19}{4}\cdot800=19\cdot200=3800$ feet.\nThe answer is 3800"""
        ),
        (
            "In Mr. Abraham's class, $10$ of the $15$ students received an $A$ on the latest exam. If the same ratio of students received an $A$ on Mrs. Berkeley's latest exam, and if Mrs. Berkeley has $24$ students total, how many students in Mrs. Berkeley's class received an $A$?",
            """If $10$ of $15$ students received an $A$, then the ratio of students receiving an $A$ to students not receiving an $A$ is $\\frac{10}{15}$, or $\\frac{2}{3}$. Let $x$ be the number of students in Mrs. Berkeley's class who received an $A$. Since the ratio is consistent across the two classes, $\\frac{2}{3} = \\frac{x}{24}$. Cross-multiplying yields $x = \\frac{24\cdot 2}{3}$, so, by simplification, we can see that 16 of Mrs. Berkeley's students must have received an $A$.\nThe answer is 16"""
        ),
        (
            "Find the value of the first term in the geometric sequence $a,b,c,32,64$.",
            """The common ratio is $\\frac{64}{32} = 2$. Therefore, the first term is $\\frac{32}{2^3} = \\frac{32}{8} = 4$. \nThe answer is 4"""
        )
    ],
    'svamp': [
        (
            'children were riding on the bus. At the bus stop 82 children got on the bus while some got off the bus. Then there were 30 children altogether on the bus. How many more children got on the bus than those that got off?',
            'Let\'s assume there are x students getting on the bus and y students getting off the bus, than 28 + x - y = 30. Therefore, x - y = 30 - 28 = 2. The answer is 2.'),
        (
            'Mary is baking a cake. The recipe calls for 11 cups of flour and 7 cups of sugar. She already put in some cups of flour. If she still needs 2 more cups of flour than sugar, How many cups of flour did she put in?',
            'Let\'s assume there are x cups of clour already added, so we know 11  - x is the remaining cups of flour. Since the reminaing cups of flour is 2 more than sugar. Then we have 11 - x - 2 = 7. Therefore, x = 11 - 2 - 7 = 2. The answer is 2.',
        ),
        (
            'Frank put 11 pieces of candy in each bag. If he had 22 pieces of candy. How many bags would he have?',
            'Let\'s assume there are x bags. Then we know 11 * x = 22. Therefore, x = 22 / 11 = 2. The answer is 2.'
        ),
        (
            'A farmer had 90 tomatoes in his garden. If he picked 154 of them yesterday and 50 today. How many tomatoes did he pick in all?',
            'The number of tomatoes picked is x = 154 + 50. Therefore, x = 204. The answer is 204.'
        ),
        (
            'The grasshopper, the frog and the mouse had a jumping contest. The grasshopper jumped 19 inches. The frog jumped 10 inches farther than the grasshopper and the mouse jumped 20 inches farther than the frog. How much farther did the mouse jump than the grasshopper?',
            'frog jumps 19 + 10 = 29 inches. The mouse jumps 29 + 20 = 49 inches. Thefore, the mouse jumps 49 - 19 = 30 inches farther than grasshopper. The answer is 30.',
        ),
        (
            'Allan brought 3 balloons and Jake brought 5 balloons to the park. Allan then bought 2 more balloons at the park. How many balloons did Allan and Jake have in the park?',
            'Allan has 3 + 2 = 5 ballons. Jake as 5 ballons. Therefore, the total ballons is 5 + 5 = 10. The answer is 10.',
        ),
        (
            'Jake has 7 fewer peaches than Steven and 9 more peaches than Jill. Steven has 16 peaches. How many peaches does Jake have?',
            'Let\'s assume Jake has x peaches. x + 7 = 16. Therefore, x = 16 - 7 = 9. The answer is 9.'
        ),
        (
            'Katie had 57 new games and 39 old games. Her friends had 34 new games. How many more games does Katie have than her friends?',
            'Katie has a total of 57 + 39 = 96 games. Therefore, Katie has 96 - 34 = 62 games than her friend. The answer is 62.'
        )
    ],
    'sat': [
        (
            "If $\frac{x-1}{3}=k$ and $k=3$, what is the value of $x$ ? \nAnswer Choices: (A) 2 (B) 4 (C) 9 (D) 10",
            "If k = 3, then x - 1 = 3 * 3, therfore, x - 1 = 9 and x = 10. The answer is D",
        ),
        (
            "For $i=\sqrt{-1}$, what is the sum $(7+3 i)+(-8+9 i)$ ? \nAnswer Choices: (A) $-1+12 i$ (B) $-1-6 i$ (C) $15+12 i$ (D) $15-6 i$ 3",
            "For (7+3 i)+(-8+9 i), the real part is 7 + (-8) = -1, the imageinary part is 3 i + 9 i = 12 i. The answer is A",
        ),
        (
            "On Saturday afternoon, Armand sent $m$ text messages each hour for 5 hours, and Tyrone sent $p$ text messages each hour for 4 hours. Which of the following represents the total number of messages sent by Armand and Tyrone on Saturday afternoon?\nAnswer Choices: (A) $9 m p$ (B) $20 m p$ (C) $5 m+4 p$ (D) $4 m+5 p$",
            "Armand texts m messages each hour for 5 hours, which leads to 5m messages. Tyrone texts p messages each hour for 4 hours, which leds to 4p messages. The total is 5m + 4p. The answer is C.",
        ),
        (
            "$$\begin{array}{r}3 x+4 y=-23 \\2 y-x=-19\end{array}$$What is the solution $(x, y)$ to the system of equations above?\nAnswer Choices: (A) $(-5,-2)$ (B) $(3,-8)$ (C) $(4,-6)$ (D) $(9,-6)$",
            "By solving this equation, we found that x = 3 and y = -8. The answer is B.",
        )
    ],
    'gaokao': [
        (
            "问题 1.   设数列 $\\left\\{a_{n}\\right\\}$ 的前 $n$ 项和为 $S_{n}$, 且 $a_{1}=-1, a_{n+1}=S_{n+1} S_{n}$, 则 $S_{n}=(\\quad)$.",
            """问题 1的解析:   让我们写出这个数列的前n项和：
$S_n = a_1 + a_2 + ... + a_n$
$S_n = -1 + (S_2 S_1) + (S_3 S_2) + ... + (S_{n+1} S_n)$
$S_n = -1 + (S_2 S_1) + (S_3 S_2) + ... + (S_n S_{n-1}) + (S_{n+1} S_n)$
$S_n = -1 + S_n (S_{n+1} - S_1)$
$S_n - S_n S_{n+1} = -1 - S_n$
$S_n (1 - S_{n+1}) = -1 - S_n$
$S_n = -\\frac{1}{1 - S_{n+1}}$
因为这个数列后面的所有项都是0，我们可以看到对于所有$n\\geq 1$，$S_{n+1} = 0$。因此，我们有：
$S_n = -\\frac{1}{1 - S_{n+1}} = -\\frac{1}{1 - 0} = -1$
这个数列前n项和的公式是$S_n = -\\frac{1}{n}$。
答案是 $-\\frac{1}{n}$"""
        ),
        (
            "问题 2.   若 $\\left(x+\\frac{1}{x}\\right)^{n}$ 的展开式中第 3 项与第 7 项的二项式系数相等, 则该展 开式中 $\\frac{1}{x^{2}}$ 的系数为 $(\\quad)$.",
            """问题 2的解析:   由题意可得, $c_{n}^{2}=c_{n}^{6}$
$\\therefore n=8$
展开式的通项 T_{r+1}=C_8^r x^{8-r}\\left(\\frac{1}{x}\\right)^r=C_8^r x^{8-2 r}$
令 $8-2 r=-2$ 可得 $r=5$
此时系数为 $c_{8}^{5}=56$
答案是 56"""
        ),
        (
            "问题 3.   函数 $\\mathrm{f}(\\mathrm{x})=\\sin (\\mathrm{x}+2 \\phi)-2 \\sin \\phi \\cos (\\mathrm{x}+\\phi)$ 的最大值为 $(\\quad)$.",
            """问题 3的解析:   函数 $f(x)=\\sin (x+2 \\phi)-2 \\sin \\phi \\cos (x+\\phi)=\\sin [(x+\\phi)+\\phi]-$ $2 \\sin \\phi \\cos (x+\\phi)$
$=\\sin (x+\\phi) \\cos \\phi+\\cos (x+\\phi) \\sin \\phi-2 \\sin \\phi \\cos (x+\\phi)=\\sin (x+\\phi) \\cos \\phi-\\cos$ $(x+\\phi) \\sin \\phi$ $=\\sin [(x+\\phi)-\\phi]=\\sin x$
故函数 $f(x)$ 的最大值为 1
答案是 1"""
        ),
        (
            "问题 4.   已知向量 $\\vec{a}=(3,1), \\vec{b}=(1,0), \\vec{c}=\\vec{a}+k \\vec{b}$. 若 $\\vec{a} \\perp \\vec{c}$, 则 $k=(\\quad)$",
            """问题 4的解析:   \\because \\vec{a}=(3,1), \\vec{b}=(1,0), \\therefore \\vec{c}=\\vec{a}+k \\vec{b}=(3+k, 1)$ ，
$\\because \\vec{a} \\perp \\vec{c}, \\therefore \\vec{a} \\square \\vec{c}=3(3+k)+1 \\times 1=0$, 解得 $k=-\\frac{10}{3}$
答案是 $-\\frac{10}{3}$"""
        ),
        (
            "问题 5.   设向量 $\\vec{a}, \\vec{b}$ 不平行, 向量 $\\lambda \\vec{a}+\\vec{b}$ 与 $\\vec{a}+2 \\vec{b}$ 平行, 则实数 $\\lambda=(\\quad)$.",
            """问题 5的解析:   $\\because$ 向量 $\\vec{a}, \\vec{b}$ 不平行, 向量 $\\lambda \\vec{a}+\\vec{b}$ 与 $\\vec{a}+2 \\vec{b}$ 平行,
$\\therefore \\lambda \\vec{a}+\\vec{b}=t(\\vec{a}+2 \\vec{b})=t \\vec{a}+2 t \\vec{b}$
$\\therefore\\left\\{\\begin{array}{c}\\lambda=\\mathrm{t} \\\\ 1=2 \\mathrm{t},\\end{array}\\right.$ 解得实数 $\\lambda=\\frac{1}{2}$.
答案是 $\\frac{1}{2}$"""
        )
    ]
}


def load_llm():
    model_id = "D:\LLMs\Llama-3-8B-Instruct"

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
    example = []
    for question, answer in examples['math'][0:5]:
        example.append(f'Problem: {question}\nAnswer: {answer}\n\n')

    tokenizer, model = load_llm()

    data_MATH, prompts_MATH = load_MATH()
    data_CARP, prompts_CARP = load_CARP()

    # file_list = ['MATH_CP', 'MATH_NT', 'MATH_PR', 'CARP']
    # data_list = [data_MATH[1], data_MATH[2], data_MATH[3], data_CARP]

    file_list = ['MATH_AL']
    data_list = [data_MATH[0]]

    # prompts_list = [prompts_MATH[0], prompts_MATH[1], prompts_MATH[2], prompts_MATH[3], prompts_CARP]

    answers_list = []
    for i, data_set in enumerate(data_list):
        data_set_answer = []
        target_file = f'../Llama3/results/5-shot-cot/{file_list[i]}.json'
        # prompt = random.Random(0).sample(prompts_list[i], 5)
        for line in tqdm(data_set):
            sys_prompt = ('You are a math expert. You need to learn examples and answer question.'
                          # 'When you respond, you only need to answer the last question. '
                          'Write it in the form: The answer is $answer$.'
                          'Think step by step.'
                          )

            messages = [
                {"role": "system", "content": f"{sys_prompt}"},
                {"role": "user", "content": f"{' '.join(example)}Problem: {line[0]} \nAnswer: "},
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
            data_set_answer.append([response, line[2]])

        with open(target_file, mode="w", encoding="utf-8") as f:
            for line in data_set_answer:
                json.dump(line, f, ensure_ascii=False)
                f.write('\n')

