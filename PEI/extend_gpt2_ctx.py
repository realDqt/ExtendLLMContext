import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import torch.nn.functional as F

def extend_gpt2_pos_embeddings(model, new_max_pos: int):
    old_wpe = model.transformer.wpe
    device  = old_wpe.weight.device   # 原 wpe 所在的 device，可能是 cuda:0
    old_max_pos, dim = old_wpe.weight.shape

    # 1. 先插值 wpe.weight
    w = old_wpe.weight.data.transpose(0,1).unsqueeze(0)      # [1, dim, old_max_pos]
    w_interp = F.interpolate(w, size=new_max_pos, mode='linear', align_corners=False)
    new_wpe_weight = w_interp.squeeze(0).transpose(0,1).contiguous()  # [new_max_pos, dim]

    # 2. 在同一个 device 上创建新的 embedding
    new_wpe = torch.nn.Embedding(new_max_pos, dim).to(device)
    new_wpe.weight.data.copy_(new_wpe_weight)

    # 3. 替换并更新 config
    model.transformer.wpe = new_wpe
    model.config.n_positions = new_max_pos
    model.config.n_ctx       = new_max_pos
    model.config.max_position_embeddings = new_max_pos

    print(f"[extend] pos embeddings {old_max_pos} → {new_max_pos} on {device}")

if __name__ == "__main__":
    local_dir   = "gpt2"
    new_max_pos = 20480
    device      = "cuda" if torch.cuda.is_available() else "cpu"

    model = GPT2LMHeadModel.from_pretrained(local_dir).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(local_dir)
    tokenizer.model_max_length = new_max_pos

    extend_gpt2_pos_embeddings(model, new_max_pos=new_max_pos)

    # 测试
    dummy = torch.arange(new_max_pos, device=device).unsqueeze(0)
    with torch.no_grad():
        out = model(dummy)
    print("logits.shape:", out.logits.shape)  

    save_dir = f"{local_dir}-ctx{new_max_pos}"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("✅ 完成扩展并保存到", save_dir)
