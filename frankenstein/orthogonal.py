import torch

from frankenstein.utils import get_tokens, get_str_tokens

def compute_orthogonal_decomposition2(model, text):
    """Compute the orthogonal decomposition of a model's attention heads.
    This is a reimplementation to check for bugs in the other implementation."""
    tokens = get_tokens(model=model, text=text)
    str_tokens = get_str_tokens(model=model, text=text)
    tokens = model.tokenizer.encode(text)
    tokens = [50256] + tokens + [50256]
    tokens = tokens[:model.cfg.n_ctx]
    str_tokens = model.tokenizer.batch_decode(tokens)
    tokens = torch.tensor(tokens, device = device).unsqueeze(0)
    logits, cache = model.run_with_cache(tokens)
    terms = []
    terms.append(cache['hook_embed'].squeeze(0))
    terms.append(cache['hook_pos_embed'].squeeze(0))
    for layer in range(model.cfg.n_layers):
        attn_term = cache[f'blocks.{layer}.hook_attn_out'].squeeze(0)
        terms.append(attn_term)
        mlp_term = cache[f'blocks.{layer}.hook_mlp_out'].squeeze(0)
        terms.append(mlp_term)

    for i in range(len(tokens[0])):
        print(f'token {i}: {str_tokens[i]}')
        for j in range(len(terms)):
            print(f'term {j}: {terms[j][i].norm().item()}')
        print()