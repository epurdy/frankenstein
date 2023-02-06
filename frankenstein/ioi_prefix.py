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
