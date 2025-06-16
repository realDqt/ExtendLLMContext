import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1. load
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # 2. 新增 pad_token（一定要和 pad_token_id / model.config.pad_token_id 对上）
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    # 3. 读文本，构造 input_ids
    with open("long_text.txt", encoding="utf-8") as f:
        text = f.read()
    tokenizer.model_max_length = 20480
    tokens = tokenizer(text, add_special_tokens=False,
                       truncation=False,
                       return_tensors="pt")
    input_ids = tokens.input_ids
    input_ids = input_ids.to(device)

    # 4. 构造 attention_mask，全 1（没有 padding）
    attention_mask = torch.ones_like(input_ids, device=device)

    # 5. 前向一次，检验能跑通
    with torch.no_grad():
        out = model(input_ids, attention_mask=attention_mask)
    print("logits shape:", out.logits.shape)  # [1,20480,vocab]

    # 6. 续写 50 token
    generated = model.generate(
        input_ids,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=50,     
        do_sample=False,
    )
    print("generated shape:", generated.shape)
    print(tokenizer.decode(generated[0, -50:], skip_special_tokens=True))

if __name__ == "__main__":
    main()
