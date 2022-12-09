import numpy as np
import torch
from tqdm import tqdm

def get_device(model):
    return next(model.parameters()).device

def get_frequency(*, model, dataset):
    print(model.cfg)
    counts = np.zeros(50257) # todo: get this from the model
    for text in tqdm(dataset):
        tokens = model.to_tokens(text).cpu().numpy()
        for token in tokens:
            counts[token] += 1
    freqs = counts / counts.sum()
    return freqs

def get_tokens(*, model, text):
    tokens = model.tokenizer.encode(text)
    tokens = [50256] + tokens + [50256]
    tokens = tokens[:model.cfg.n_ctx]
    device = get_device(model)
    tokens = torch.tensor(tokens, device=device).unsqueeze(0)
    return tokens

def get_str_tokens(*, model, text):
    tokens = get_tokens(model=model, text=text)
    str_tokens = model.tokenizer.batch_decode(tokens)
    return str_tokens