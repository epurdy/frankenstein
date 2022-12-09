import datasets
import easy_transformer
import numpy as np
import torch
from easy_transformer import EasyTransformer
from tqdm import tqdm


def get_projector(subspace):
#    return subspace @ torch.pinverse(subspace.T @ subspace) @ subspace.T
    return subspace.T @ torch.pinverse(subspace @ subspace.T) @ subspace


def get_model(*, name, device):
    # device = 'cuda'
    # device = 'mps'
    # device = 'cpu'
    # @param ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'facebook/opt-125m', 'facebook/opt-1.3b', 
    # 'facebook/opt-2.7b', 'facebook/opt-6.7b', 'facebook/opt-13b', 'facebook/opt-30b', 'facebook/opt-66b', 
    # 'EleutherAI/gpt-neo-125M', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-2.7B']
    model = EasyTransformer.from_pretrained(name).to(device)
    return model

def get_induction_dataset(device, seed=0):
    torch.random.manual_seed(seed)
    random_tokens = torch.multinomial(torch.ones(10_000), 1000, replacement=True).to(device)
    random_tokens = random_tokens.reshape(10, 100)
    induction = torch.cat([random_tokens] * 3, dim=1).to(device)
    induction = torch.unbind(induction, dim=0)
    return induction

def get_openwebtext_dataset(seed=0):
    data = datasets.load_dataset('stas/openwebtext-10k', split='train')
    openwebtext = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
    openwebtext = [x['text'][0] for x in openwebtext]
    return openwebtext


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
    str_tokens = model.tokenizer.batch_decode(tokens[0])
    return str_tokens

def detect_outliers(vals):
    median = np.median(vals)
    diff = np.abs(vals - median)
    mad = np.median(diff)
    modified_z_score = 0.6745 * diff / mad
    retval = modified_z_score > 3.5
    return retval