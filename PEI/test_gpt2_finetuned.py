#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

def main():
    # 1. 设置路径
    model_dir = "./gpt2-finetuned-ctx20480"    # 你的微调输出目录

    # 2. 加载 tokenizer 和模型
    #    注意：如果你在微调时指定了 pad_token="[PAD]"，这里也要传入
    tokenizer = GPT2TokenizerFast.from_pretrained(model_dir, pad_token="[PAD]")
    model = GPT2LMHeadModel.from_pretrained(model_dir)

    # 3. 移动到 GPU/CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 4. 构造输入
    while True:
        # 1. 从终端读取用户输入
        prompt = input("请输入您的提问（输入 exit 或 quit 退出）：").strip()
        if prompt.lower() in ["exit", "quit"]:
            print("已退出。")
            break

        # 2. 构造输入
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # 3. 生成
        #   这里示范采样生成，可根据需求改成 greedy、beam search 等
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_length=100,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

        # 4. 解码并打印
        generated = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        # 去掉 prompt，只保留模型回复
        reply = generated[len(prompt):].strip()

        print(f"Prompt: {prompt}")
        print(f"Response: {reply}\n")

if __name__ == "__main__":
    main()
