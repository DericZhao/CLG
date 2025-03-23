from dataclasses import dataclass
from typing import Dict


@dataclass
class Template:
    template_name:str
    system_format: str
    user_format: str
    assistant_format: str
    system: str
    stop_word: str
    # stop_token_id: int


template_dict: Dict[str, Template] = dict()


def register_template(template_name, system_format, user_format, assistant_format, system, stop_word=None):
    template_dict[template_name] = Template(
        template_name=template_name,
        system_format=system_format,
        user_format=user_format,
        assistant_format=assistant_format,
        system=system,
        stop_word=stop_word,
        # stop_token_id=stop_token_id
    )


register_template(
    template_name='llama2',
    system_format='<<SYS>>\n{content}\n<</SYS>>\n\n',
    user_format='[INST]{content}[/INST]',
    assistant_format='{content} </s>',
    system=
        # "You are a helpful, respectful and honest assistant. "
        # "You are a helpful, respectful and honest assistant. "
        # "Always answer as helpfully as possible, while being safe. "
        # "You are a math teacher. ",
        "You are a math teacher. Please provide the correct answer based on the question. \n\n"
        "You just need to tell me the final answer to the question. \n\n"
        "You don't need to talk about the middle thought process. \n\n",
        # "racist, sexist, toxic, dangerous, or illegal content. "
        # "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        # "If a question does not make any sense, or is not factually coherent, "
        # "explain why instead of answering something not correct. "
        # "If you don't know the answer to a question, please don't share false information.",
    stop_word='</s>'
)

register_template(
    template_name='llama3',
    system_format='<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>',
    user_format='<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n',
    assistant_format='{content}<|eot_id|>',
    system="You are a math teacher. "
           "Please provide the correct answer to the question. ",
    stop_word='<|eot_id|>'
)

register_template(
    template_name='examiner',
    system_format='<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>',
    user_format='<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n',
    assistant_format='{content}<|eot_id|>',
    system="You are a math teacher."
           "Please determine whether the student answer is correct based on the math problem and the reference answer."
           "You just need to say 'yes' or 'no'. ",
    stop_word='<|eot_id|>'
)

register_template(
    template_name='incontext',
    system_format='<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>',
    user_format='<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n',
    assistant_format='{content}<|eot_id|>',
    system="You are a helpful assistant for solving math problems. \n"
           "Please generate Given conditions by imitating the given examples. "
           "You only need to state the Given conditions. ",
    stop_word='<|eot_id|>'
)


register_template(
    template_name='deepseek',
    system_format=None,
    user_format='User: {content}\n\nAssistant: ',
    assistant_format='{content}<｜end▁of▁sentence｜>',
    system=None,
    stop_word='<｜end▁of▁sentence｜>'
)
