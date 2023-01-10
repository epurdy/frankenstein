import os
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from functools import partial
from collections import defaultdict
from frankenstein.utils import get_tokens

def remember_hook(result, hook, cache):
    cache[hook.name] = result.clone()

def remember_head_hook(result, hook, cache, head):
    cache[hook.name] = result[:, :, head, :].clone()

def p2p_ablate_mlp_hook(result, hook, cache, key1, key2, ln, mean):
    counterfactual_rs = cache[key2] + mean - cache[key1]
    # this next step is probably unnecessary, not sure
    counterfactual_centered_rs = counterfactual_rs - counterfactual_rs.mean(dim=-1, keepdim=True)
    normed_counterfactual_input = ln(counterfactual_centered_rs)
    return normed_counterfactual_input

def p2p_ablate_attn_hook(result, hook, head, cache, key1, key2, ln, mean):
    counterfactual_rs = cache[key2] + mean - cache[key1]
    # this next step is probably unnecessary, not sure
    counterfactual_centered_rs = counterfactual_rs - counterfactual_rs.mean(dim=-1, keepdim=True)
    normed_counterfactual_input = ln(counterfactual_centered_rs)
    result = result.clone()
    result[:, :, head, :] = normed_counterfactual_input
    return result

def harvest_p2p_ablation_scores_mlp_mlp(*, model, layer1, layer2, mean_activations, tokens):
    print(f'Harvesting p2p ablation scores for mlp-mlp {layer1} -> {layer2}')
    microcache = {}
    p2p_ablated_logits = model.run_with_hooks(tokens,
        fwd_hooks=[(f'blocks.{layer1}.hook_mlp_out', partial(remember_hook, cache=microcache)),
                    (f'blocks.{layer2}.hook_resid_mid', partial(remember_hook, cache=microcache)),
                    (f'blocks.{layer2}.mlp.hook_input', 
                        partial(p2p_ablate_mlp_hook, cache=microcache,
                                key1=f'blocks.{layer1}.hook_mlp_out',
                                key2=f'blocks.{layer2}.hook_resid_mid',
                                ln=model.blocks[layer2].ln2,
                                mean=mean_activations[f'blocks.{layer1}.hook_mlp_out']))])
    return p2p_ablated_logits

def harvest_p2p_ablation_scores_attn_attn(*, model, tokens, layer1, head1, layer2, head2, mat2, 
                                          mean_activations):
    print(f'Harvesting p2p ablation scores for attn-attn {layer1}.{head1} -> {layer2}.{head2}')
    microcache = {}
    mat2 = mat2.lower()
    p2p_ablated_logits = model.run_with_hooks(tokens,
        fwd_hooks=[(f'blocks.{layer1}.attn.hook_result', partial(remember_head_hook, cache=microcache, head=head1)),
                    (f'blocks.{layer2}.hook_resid_pre', partial(remember_hook, cache=microcache)),
                    (f'blocks.{layer2}.attn.hook_input_{mat2}', 
                        partial(p2p_ablate_attn_hook, cache=microcache, head=head2,
                                key1=f'blocks.{layer1}.attn.hook_result',
                                key2=f'blocks.{layer2}.hook_resid_pre',
                                ln=model.blocks[layer2].ln1,
                                mean=mean_activations[f'blocks.{layer1}.attn.hook_result'][head1, :]))])
    return p2p_ablated_logits

def harvest_p2p_ablation_scores_mlp_attn(*, model, tokens, layer1, layer2, head2, mat2, mean_activations):
    print(f'Harvesting p2p ablation scores for mlp-attn {layer1} -> {layer2}.{head2}')
    microcache = {}
    mat2 = mat2.lower()
    p2p_ablated_logits = model.run_with_hooks(tokens,
        fwd_hooks=[(f'blocks.{layer1}.hook_mlp_out', partial(remember_hook, cache=microcache)),
                    (f'blocks.{layer2}.hook_resid_pre', partial(remember_hook, cache=microcache)),
                    (f'blocks.{layer2}.attn.hook_input_{mat2}', 
                        partial(p2p_ablate_attn_hook, cache=microcache, head=head2,
                                key1=f'blocks.{layer1}.hook_mlp_out',
                                key2=f'blocks.{layer2}.hook_resid_pre',
                                ln=model.blocks[layer2].ln1,                                
                                mean=mean_activations[f'blocks.{layer1}.hook_mlp_out']))])
    return p2p_ablated_logits


def harvest_p2p_ablation_scores_attn_mlp(*, model, tokens, layer1, head1, layer2, mean_activations):
    print(f'Harvesting p2p ablation scores for attn-mlp {layer1}.{head1} -> {layer2}')
    microcache = {}
    p2p_ablated_logits = model.run_with_hooks(tokens,
        fwd_hooks=[(f'blocks.{layer1}.attn.hook_result', partial(remember_head_hook, cache=microcache, head=head1)),
                    (f'blocks.{layer2}.hook_resid_mid', partial(remember_hook, cache=microcache)),
                    (f'blocks.{layer2}.mlp.hook_input', 
                        partial(p2p_ablate_mlp_hook, cache=microcache,
                                key1=f'blocks.{layer1}.attn.hook_result',
                                key2=f'blocks.{layer2}.hook_resid_mid',
                                ln=model.blocks[layer2].ln2,
                                mean=mean_activations[f'blocks.{layer1}.attn.hook_result'][head1, :]))])
    return p2p_ablated_logits

def harvest_p2p_ablation_scores(*, output_dir, model, dataset, upstream, downstream, mean_activations,
    clean_logit_losses, save=True):
    layer1, component1, head1, mat1 = upstream.split('.')
    layer2, component2, head2, mat2 = downstream.split('.')

    retval = np.zeros((len(dataset), 1024))
    lengths = np.zeros(len(dataset))
    mean_computation_values = []
    for itext, text in enumerate(dataset):
        tokens = get_tokens(model=model, text=text)
        lengths[itext] = len(tokens[0])
        shifted_tokens = torch.cat([tokens[:, 1:], tokens[:, :1]], dim=1)
        model.reset_hooks()

        if component1 == component2 == 'mlp':
            p2p_ablated_logits = harvest_p2p_ablation_scores_mlp_mlp(model=model, tokens=tokens,
                                                layer1=int(layer1), layer2=int(layer2),
                                                mean_activations=mean_activations)
        elif component1 == component2 == 'attn':
            p2p_ablated_logits = harvest_p2p_ablation_scores_attn_attn(model=model, tokens=tokens,
                layer1=int(layer1), head1=int(head1), layer2=int(layer2), head2=int(head2), mat2=mat2,
                mean_activations=mean_activations)
        elif (component1, component2) == ('mlp', 'attn'):
            p2p_ablated_logits = harvest_p2p_ablation_scores_mlp_attn(model=model, tokens=tokens,
                layer1=int(layer1), layer2=int(layer2), head2=int(head2), mat2=mat2,
                mean_activations=mean_activations)
        elif (component1, component2) == ('attn', 'mlp'):
            p2p_ablated_logits = harvest_p2p_ablation_scores_attn_mlp(model=model, tokens=tokens,
                layer1=int(layer1), head1=int(head1), layer2=int(layer2),
                mean_activations=mean_activations)

        clean_loss = clean_logit_losses[itext]
        p2p_ablated_loss = F.cross_entropy(p2p_ablated_logits[0], shifted_tokens[0], reduction='none')
        loss_diff = p2p_ablated_loss - clean_loss
        for i in range(loss_diff.shape[0]):
            retval[itext, i] = loss_diff[i].item()
            mean_computation_values.append(loss_diff[i].item())

    mean = np.mean(mean_computation_values)
    print(f'Mean loss diff: {mean:.3f} (std: {np.std(mean_computation_values):.3f})')

    if save:
        output_path = os.path.join(output_dir, f'{upstream}_{downstream}.{len(dataset)}docs.npz')
        np.savez_compressed(output_path, loss_diffs=retval, lengths=lengths)

    return retval, lengths

def before(*, upstream, downstream):
    layer1, component1, head1, mat1 = upstream.split('.')
    layer2, component2, head2, mat2 = downstream.split('.')
    if component1 == component2 == 'mlp':
        return int(layer1) < int(layer2)
    elif component1 == component2 == 'attn':
        return (int(layer1) < int(layer2)) and (mat1 == 'O') and (mat2 in 'QKV')
    elif (component1, component2) == ('mlp', 'attn'):
        return (int(layer1) < int(layer2)) and (mat2 in 'QKV')
    elif (component1, component2) == ('attn', 'mlp'):
        return (int(layer1) <= int(layer2)) and (mat1 == 'O')

def harvest_mean_activations(*, model, dataset, device=None):
    megacache = defaultdict(list)
    clean_logit_losses = []
    suffixes = ['hook_mlp_out', 'attn.hook_result']
    with torch.no_grad():
        for text in dataset:
            tokens = get_tokens(model=model, text=text)
            shifted_tokens = torch.cat([tokens[:, 1:], tokens[:, :1]], dim=1)
            model.reset_hooks()
            logits, cache = model.run_with_cache(tokens)
            clean_logit_losses.append(F.cross_entropy(input=logits[0], target=shifted_tokens[0], reduction='none'))
            for k in cache:
                if any(k.endswith(s) for s in suffixes):
                    megacache[k].append(cache[k][0].mean(dim=0).cpu())
        mean_activations = {}
        for k in megacache:
            # slightly weird weighting, but whatever
            mean_activations[k] = torch.stack(megacache[k], dim=0).mean(dim=0)
            if device:
                mean_activations[k] = mean_activations[k].to(device)
            #print(k, mean_activations[k].shape)
    return mean_activations, clean_logit_losses

def harvest_all_p2p_ablation_scores(*, model, dataset, output_dir):
    model.set_use_attn_input(True)
    model.set_use_attn_result(True)

    mean_activations, clean_logit_losses = harvest_mean_activations(
        model=model, dataset=dataset)

    components = ['%d.attn.%d' % (layer, head) 
        for layer in range(model.cfg.n_layers) for head in range(model.cfg.n_heads)]
    components.extend(['%d.mlp.xxx' % layer for layer in range(model.cfg.n_layers)])

    for upstream in tqdm(components):
        for downstream in components:
            if not before(upstream=upstream, downstream=downstream):
                continue
            with torch.no_grad():
                harvest_p2p_ablation_scores(model=model, dataset=dataset, 
                    upstream=upstream, downstream=downstream,
                    mean_activations=mean_activations, output_dir=output_dir, 
                    clean_logit_losses=clean_logit_losses)