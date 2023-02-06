import random
from collections import defaultdict

from pysbd import Segmenter
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from frankenstein.ioi import IOIDataset
from frankenstein.utils import get_openwebtext_dataset, get_tokens

def harvest_prefixes(owt):
    prefixes = defaultdict(list)
    seg = Segmenter(language='en', clean=False)

    for i, doc in enumerate(owt):
        sentences = seg.segment(doc)
        for j, sent in enumerate(sentences):
            sent = sent.strip()
            if sent[1:].lower() != sent[1:]:
                continue
            if sent[0].lower() == sent[0]:
                continue
            if sent[-1] != '.':
                continue
            if '@' in sent:
                continue
            if 'http' in sent:
                continue
            if '/' in sent:
                continue
            if len(sent) < 10:
                continue

            first_word = sent.split()[0]
            prefixes[first_word].append(sent)
            print(i, j, len(prefixes), sent)

    return prefixes

def make_whitelisted_prefixes(prefixes):
    whitelisted_prefixes = []
    for k in prefixes:
        if len(prefixes[k]) > 100:
            print(k)
            whitelisted_prefixes.extend(prefixes[k])
    return whitelisted_prefixes

def strip_endoftext(tokens):
    assert tokens[0] == tokens[-1] == 50256
    tokens = tokens[1:-1]
    assert (tokens != 50256).all()
    return tokens


def created_augmented_prompt(*, model, prefix, ioi, i):
    i_ioi = i % len(ioi.toks)
    prefix = prefix.strip()
    prefix_tokens = strip_endoftext(get_tokens(model=model, text=prefix)[0])
    ioi_text = ioi.ioi_prompts[i_ioi]['text'].strip()
    correct_text = prefix + ' ' + ioi_text
    correct_tokens = strip_endoftext(get_tokens(model=model, text=correct_text)[0])
    ioi_tokens = ioi.toks[i_ioi][:2 + ioi.sem_tok_idx['end'][i_ioi]]
    assert (ioi_tokens != 50256).all()
    naive_tokens = torch.cat([prefix_tokens, ioi_tokens], dim=0)
    offset = len(prefix_tokens) + len(correct_tokens) - len(naive_tokens)
    assert correct_tokens[offset + ioi.sem_tok_idx['S'][i_ioi]].item() == correct_tokens[offset + ioi.sem_tok_idx['S2'][i_ioi]].item()
    assert correct_tokens[offset + ioi.sem_tok_idx['IO'][i_ioi]].item() == correct_tokens[offset + 1 + ioi.sem_tok_idx['end'][i_ioi]].item()
    if len(correct_tokens) == len(naive_tokens):
        # first word in ioi prompt is now preceded by a space
        assert (naive_tokens != correct_tokens).sum() == 1
    else:   
        # "Afterwards" is split into two tokens if there is no preceding space!
        assert 'Afterwards' in correct_text
        assert 'Afterwards' in ioi_text
        correct_list = correct_tokens.tolist()
        naive_list = naive_tokens.tolist()
        correct_list_split = len(correct_list) - 1 - correct_list[::-1].index(39063) # " Afterwards"
        assert correct_list[:correct_list_split] == naive_list[:correct_list_split]
        assert correct_list[correct_list_split + 1:] == naive_list[correct_list_split + 2:]
        assert naive_list[correct_list_split: correct_list_split + 2] == [3260, 2017] # "Afterwards"

    return dict(tokens=correct_tokens, offset=offset, i_ioi=i_ioi)

def get_prefix_ioi(model):
    """Get the IOI dataset with prefixes from openwebtext."""
    N = 3000
    ioi = IOIDataset(prompt_type='mixed', N=N, nb_templates=30)

    owt = get_openwebtext_dataset()

    prefixes = harvest_prefixes(owt)

    whitelisted_prefixes = make_whitelisted_prefixes(prefixes)

    random.shuffle(whitelisted_prefixes)

    length_filtered_prefixes = []
    token_counts = []
    for i, prefix in enumerate(whitelisted_prefixes):
        tokens = get_tokens(model=model, text=prefix)[0]
        token_counts.append(len(tokens))
        if len(tokens) <= 50:
            length_filtered_prefixes.append(prefix)

    plt.hist(token_counts, bins=200)
    plt.yscale('log')
    print(len(length_filtered_prefixes))
    print(np.bincount(token_counts))

    return dict(ioi=ioi, prefixes=length_filtered_prefixes)



def check_ioi_performance(*, model, prefixes, ioi):
    hits = 0
    total = 0
    logit_diffs = []
    for i, prefix in enumerate(tqdm(prefixes[:1000])):
        stuff = created_augmented_prompt(model=model, prefix=prefix, ioi=ioi, i=i)
        correct_tokens = stuff['tokens']
        offset = stuff['offset']
        i_ioi = stuff['i_ioi']

        model.reset_hooks()
        logits = model(correct_tokens)[0, -2, :]
        s1pos = ioi.sem_tok_idx['S'][i_ioi].item() + offset
        iopos = ioi.sem_tok_idx['IO'][i_ioi].item() + offset
        sname = correct_tokens[s1pos].item()
        ioname = correct_tokens[iopos].item()

        logit_diff = logits[ioname].item() - logits[sname].item()
        logit_diffs.append(logit_diff)

        if logit_diff > 0:
            hits += 1
        total += 1

        if i % 100 == 99:
            print(i, hits / total, np.mean(logit_diffs))

    print(hits / total, np.mean(logit_diffs))

    plt.figure()
    plt.hist(logit_diffs, bins=100)


def check_ioi_performance2(model, ioi):
    hits = 0
    total = 0
    logit_diffs = []
    for i in tqdm(range(len(ioi.toks))):
        i_ioi = i % len(ioi.toks)
        ioi_text = ioi.ioi_prompts[i_ioi]['text'].strip()
        ioi_tokens = ioi.toks[i_ioi][:2 + ioi.sem_tok_idx['end'][i_ioi]]
        assert (ioi_tokens != 50256).all()
        correct_tokens = ioi_tokens
        model.reset_hooks()
        logits = model(correct_tokens.long())[0, -2, :]
        s1pos = ioi.sem_tok_idx['S'][i_ioi].item()
        iopos = ioi.sem_tok_idx['IO'][i_ioi].item()
        sname = correct_tokens[s1pos].item()
        ioname = correct_tokens[iopos].item()

        logit_diff = logits[ioname].item() - logits[sname].item()
        logit_diffs.append(logit_diff)

        if logit_diff > 0:
            hits += 1
        total += 1

        if i % 100 == 99:
            print(i, hits / total, np.mean(logit_diffs))

    print(hits / total, np.mean(logit_diffs))

    plt.figure()
    plt.hist(logit_diffs, bins=100)


def build_classifier_datasets(*, model, ioi, prefixes, n=1000):
    dataset = defaultdict(list)
    print(ioi.sem_tok_idx.keys())
    for i in tqdm(range(n)):
        augmented_prompt = created_augmented_prompt(model=model, prefix=prefixes[i], ioi=ioi, i=i)
        tokens = augmented_prompt['tokens']
        offset = augmented_prompt['offset']
        i_ioi = augmented_prompt['i_ioi']

        iopos = offset + ioi.sem_tok_idx['IO'][i_ioi].item()
        s1pos = offset + ioi.sem_tok_idx['S'][i_ioi].item()
        s1p1pos = offset + ioi.sem_tok_idx['S+1'][i_ioi].item()
        s2pos = offset + ioi.sem_tok_idx['S2'][i_ioi].item()
        endpos = offset + ioi.sem_tok_idx['end'][i_ioi].item()
        abb = (iopos < s1pos)
        if abb:
            name1pos = iopos
            name2pos = s1pos
        else: # bab
            name1pos = s1pos
            name2pos = iopos
        ioname = tokens[iopos].item()
        sname = tokens[s1pos].item()
        name1 = tokens[name1pos].item()
        name2 = tokens[name2pos].item()

        datapoint = dict(
            augmented_prompt=augmented_prompt,
            iopos=iopos, s1pos=s1pos, s1p1pos=s1p1pos, s2pos=s2pos,
            endpos=endpos, name1pos=name1pos, name2pos=name2pos,
            ioname=ioname, sname=sname,
            name1=name1, name2=name2, abb=abb)

        model.set_use_attn_result(True)
        model.reset_hooks()
        logits, cache = model.run_with_cache(tokens.long())
        for layer in range(model.cfg.n_layers):
            for head in range(model.cfg.n_heads):
                vecend = cache[f'blocks.{layer}.attn.hook_result'][0, endpos, head, :]
                vecio = cache[f'blocks.{layer}.attn.hook_result'][0, iopos, head, :]                
                vecs1 = cache[f'blocks.{layer}.attn.hook_result'][0, s1pos, head, :]
                vecs1p1 = cache[f'blocks.{layer}.attn.hook_result'][0, s1p1pos, head, :]
                vecs2 = cache[f'blocks.{layer}.attn.hook_result'][0, s2pos, head, :]
                vecname1 = cache[f'blocks.{layer}.attn.hook_result'][0, name1pos, head, :]
                vecname2 = cache[f'blocks.{layer}.attn.hook_result'][0, name2pos, head, :]
                dataset[layer, head].append(dict(
                    vecend=vecend, vecio=vecio, vecs1=vecs1, vecs1p1=vecs1p1, vecs2=vecs2,
                    vecname1=vecname1, vecname2=vecname2, **datapoint))
            # mlp
            vecend = cache[f'blocks.{layer}.hook_mlp_out'][0, endpos, :]
            vecio = cache[f'blocks.{layer}.hook_mlp_out'][0, iopos, :]
            vecs1 = cache[f'blocks.{layer}.hook_mlp_out'][0, s1pos, :]
            vecs1p1 = cache[f'blocks.{layer}.hook_mlp_out'][0, s1p1pos, :]
            vecs2 = cache[f'blocks.{layer}.hook_mlp_out'][0, s2pos, :]
            vecname1 = cache[f'blocks.{layer}.hook_mlp_out'][0, name1pos, :]
            vecname2 = cache[f'blocks.{layer}.hook_mlp_out'][0, name2pos, :]
            dataset[layer, 'mlp'].append(dict(
                vecend=vecend, vecio=vecio, vecs1=vecs1, vecs1p1=vecs1p1, vecs2=vecs2,
                vecname1=vecname1, vecname2=vecname2, **datapoint))

            for resid_point in ['pre', 'mid', 'post']:
                if resid_point == 'post' and layer < 11:
                    continue
                vecend = cache[f'blocks.{layer}.hook_resid_{resid_point}'][0, endpos, :]
                vecio = cache[f'blocks.{layer}.hook_resid_{resid_point}'][0, iopos, :]
                vecs1 = cache[f'blocks.{layer}.hook_resid_{resid_point}'][0, s1pos, :]
                vecs1p1 = cache[f'blocks.{layer}.hook_resid_{resid_point}'][0, s1p1pos, :]
                vecs2 = cache[f'blocks.{layer}.hook_resid_{resid_point}'][0, s2pos, :]
                vecname1 = cache[f'blocks.{layer}.hook_resid_{resid_point}'][0, name1pos, :]
                vecname2 = cache[f'blocks.{layer}.hook_resid_{resid_point}'][0, name2pos, :]
                dataset[layer, resid_point].append(dict(
                    vecend=vecend, vecio=vecio, vecs1=vecs1, vecs1p1=vecs1p1, vecs2=vecs2,
                    vecname1=vecname1, vecname2=vecname2,
                    **datapoint))
                
    return dataset
    
