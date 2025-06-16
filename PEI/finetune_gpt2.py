import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
from typing import List, Dict

import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

def load_and_prepare_dataset(json_path: str, tokenizer: GPT2TokenizerFast, 
                             max_length: int = 1024) -> Dataset:
    """
    1) 从 JSON 文件读取数据
    2) 拼接成“Instruction+Input+Output”格式
    3) 调用 tokenizer
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        records: List[Dict] = json.load(f)

    prompts = []
    for rec in records:
        instr = rec.get("instruction", "").strip()
        inp   = rec.get("input", "").strip()
        outp  = rec.get("output", "").strip()
        # 自定义拼接格式，你可以按需改
        text = "### Instruction:\n" + instr + "\n"
        if inp:
            text += "### Input:\n" + inp + "\n"
        text += "### Response:\n" + outp
        prompts.append(text)

    # 构造一个 HuggingFace Dataset
    ds = Dataset.from_dict({"text": prompts})

    # tokenize
    def tokenize_fn(batch):
        return tokenizer(batch["text"],
                         truncation=True,
                         max_length=max_length,
                         padding="max_length")

    ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    return ds

def main():
    # 路径设置
    model_dir = "/path/to/your/model"           # 本地 GPT-2 目录
    base_data_dir = "/path/to/your/data"
    data_paths = ["identity.json"]  # 训练数据 , "alpaca_en_demo.json", "alpaca_zh_demo.json"
    for i in range(len(data_paths)):
        data_paths[i] = base_data_dir + data_paths[i]

    # 1. 加载 tokenizer 和模型
    tokenizer = GPT2TokenizerFast.from_pretrained(model_dir,
                                                  pad_token="[PAD]")
    # GPT-2 自带的 pad_token_id 可能不存在，显式指定
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    model.resize_token_embeddings(len(tokenizer))

    # 2. 加载并合并所有子数据集
    datasets_list = []
    for path in data_paths:
        ds = load_and_prepare_dataset(path, tokenizer, max_length=1024)
        datasets_list.append(ds)

    # 如果只有一个文件就直接用第一个，否则合并
    if len(datasets_list) == 1:
        full_dataset = datasets_list[0]
    else:
        full_dataset = concatenate_datasets(datasets_list)

    # 3. DataCollator（对因果语言模型无需 mlm）
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # 4. TrainingArguments
    training_args = TrainingArguments(
        output_dir="./gpt2-finetuned-ctx20480",
        overwrite_output_dir=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        num_train_epochs=30,
        logging_steps=20,
        save_steps=200,
        fp16=torch.cuda.is_available(),
        # ↓ 删除 evaluation_strategy，改用老接口
        do_train=True,
        do_eval=False,      # 如果想 eval 就改成 True
        eval_steps=200,     # 每多少步做一次 eval
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=full_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 6. 开始训练
    trainer.train()
    trainer.save_model("./gpt2-finetuned-ctx20480")

if __name__ == "__main__":
    main()
