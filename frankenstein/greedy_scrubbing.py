import numpy as np
from functools import partial
import torch.nn.functional as F
import torch
from tqdm import tqdm
from multiprocessing import Pool

from frankenstein.utils import get_model, get_tokens, get_str_tokens
import frankenstein.ioi as ioi

def patch_head(result, hook, head, cache, idx_to_change):
    result = result.clone()
    result[:, idx_to_change, head, :] = cache[hook.name][:, :, head, :][:, idx_to_change, :]
    return result


MEGACACHE = {}

def evaluate_subset_index(*, dataset, good_idxs, model, subset, seed):    
    np.random.seed(seed)
    model.set_use_attn_result(True)
    model.reset_hooks()

    base_idx = np.random.choice(good_idxs)

    random_idx = {}
    for layer, head, index in subset:
        idx = base_idx
        while idx == base_idx:
            idx = np.random.choice(good_idxs)
        random_idx[layer, head, index] = idx

    caches = {}
    for idx in set(random_idx.values()):
        if idx in MEGACACHE:
            caches[idx] = MEGACACHE[idx]
            continue
        model.reset_hooks()
        logits, cache = model.run_with_cache(dataset.toks[idx].long())
        caches[idx] = cache
        MEGACACHE[idx] = cache

    with torch.no_grad():
        hooks = []
        for (layer, head, index) in subset:
            hooks.append((f'blocks.{layer}.attn.hook_result', partial(patch_head, head=head,
                          cache=caches[random_idx[layer, head, index]], idx_to_change=index)))
        model.reset_hooks()                        
        logits = model.run_with_hooks(dataset.toks[base_idx, :1 + dataset.sem_tok_idx['end'][base_idx]].long(), 
                                      fwd_hooks=hooks)
        io = dataset.toks[base_idx, dataset.sem_tok_idx['IO'][base_idx]].long()
        s = dataset.toks[base_idx, dataset.sem_tok_idx['S'][base_idx]].long()
        # negative -> model is doing well
        # positive -> model is doing poorly
        #loss_diff = F.cross_entropy(logits[0, -1, :], io) - F.cross_entropy(logits[0, -1, :], s)
        loss_diff = logits[0, -1, io] - logits[0, -1, s]

        return loss_diff.item()

NUM_TRIALS = 3
def evaluate_subset_index_wrapper(kwargs):
    total = 0
    seed = kwargs['seed']
    kwargs.pop('seed')
    for trial in range(NUM_TRIALS):
        total += evaluate_subset_index(seed=seed + trial, **kwargs)
    return total / NUM_TRIALS

def greedy_scrubbing_index(*, dataset, good_idxs, model):
    to_delete = []
    layer_head_indices = [(layer, head, index) 
        for layer in range(model.cfg.n_layers) 
        for head in range(model.cfg.n_heads)
        for index in range(1 + dataset.sem_tok_idx['end'][0])]
    base_loss_diff = evaluate_subset_index(dataset=dataset, good_idxs=good_idxs, model=model, subset=[], seed=0)
    seed = 0
    while len(layer_head_indices) > 0:
        best_loss_diff = None
        best_layer_head_index = None
        args = [dict(dataset=dataset, good_idxs=good_idxs, model=model,
                     subset=to_delete + [layer_head_index], seed=0)
                for layer_head_index in layer_head_indices]
        with Pool(8) as p:
            results = list(tqdm(p.imap(evaluate_subset_index_wrapper, args), total=len(args)))
        for layer_head_index, loss_diff in zip(layer_head_indices, results):
            loss_diff = np.abs(loss_diff - base_loss_diff)
            if best_loss_diff is None or loss_diff < best_loss_diff:
                best_loss_diff = loss_diff
                best_layer_head_index = layer_head_index
        to_delete.append(best_layer_head_index)
        layer_head_indices.remove(best_layer_head_index)
        print(f'best loss diff: {best_loss_diff} for {best_layer_head_index}')

    return to_delete


def do_the_thing():
    model = get_model(name='gpt2', device='cpu')
    dataset = ioi.IOIDataset(prompt_type='mixed', N=10_000)
    good_idxs = [i for i in range(len(dataset)) 
        if all(dataset.sem_tok_idx[k][i] == dataset.sem_tok_idx[k][0] for k in dataset.sem_tok_idx)]

    deletion_order_index = greedy_scrubbing_index(dataset=dataset, good_idxs=good_idxs, model=model)
    print(deletion_order_index)


if __name__ == '__main__':
    do_the_thing()