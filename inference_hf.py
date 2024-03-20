#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Annotated, Union
import json

import typer
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

app = typer.Typer(pretty_exceptions_show_locals=False)


def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def load_model_and_tokenizer(model_dir: Union[str, Path]) -> tuple[ModelType, TokenizerType]:
    model_dir = _resolve_path(model_dir)
    if (model_dir / 'adapter_config.json').exists():
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, device_map='auto'
        )
        tokenizer_dir = model.peft_config['default'].base_model_name_or_path
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, device_map='auto'
        )
        tokenizer_dir = model_dir
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=True
    )
    return model, tokenizer


def write_to_txt(str_list, txt_path):
    with open(txt_path, 'w', encoding='utf-8') as file:
        # 将字符串数组逐行写入文件，并在每行末尾添加换行符
        for line in str_list:
            file.write(line + '\n')


@app.command()
def main(
        model_dir: Annotated[str, typer.Argument(help='')],
        test_file: Annotated[str, typer.Option(help='')],
        output_dir: Annotated[str, typer.Option(help='')],
):
    model, tokenizer = load_model_and_tokenizer(model_dir)
    print('Load model complete.')
    prompt_list = []
    summary_list = []
    with open(test_file, 'r', encoding='utf-8') as file:
        # 逐行读取数据
        for line in file:
            # 对每一行进行处理
            # 例如，打印每一行内容
            conversation = json.loads(line.strip())
            prompt_list.append(conversation['conversations'][0]['content'])
            summary_list.append(conversation['conversations'][1]['content'])
    assert len(prompt_list) == len(summary_list)
    pred_list = []
    for i in range(len(prompt_list)):
        prompt = prompt_list[i]
        response, _ = model.chat(tokenizer, prompt)
        print('Prompt {} is done. {}'.format(i, response))
        pred_list.append(response)
    print(len(pred_list))
    write_to_txt(pred_list, output_dir + '/pred.txt')
    write_to_txt(summary_list, output_dir + '/target.txt')
    print('Inference done.')


if __name__ == '__main__':
    app()
