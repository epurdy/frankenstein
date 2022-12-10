from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.express as px
import torch
import torch.nn.functional as F
from tqdm import tqdm

from frankenstein.utils import get_projector, get_device, show_layer_head_image, get_tokens, get_str_tokens


def get_attention_head_subspaces(model):
    subspaces = {}
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):            
            subspaces[f'{layer:02d}.{head:02d}.Q'] = model.blocks[layer].attn.W_Q[head].T
            subspaces[f'{layer:02d}.{head:02d}.K'] = model.blocks[layer].attn.W_K[head].T
            subspaces[f'{layer:02d}.{head:02d}.V'] = model.blocks[layer].attn.W_V[head].T
            subspaces[f'{layer:02d}.{head:02d}.O'] = model.blocks[layer].attn.W_O[head]

    print(f'Found {len(subspaces)} subspaces.')
    key = list(subspaces.keys())[0]
    print(f'Each subspace is of shape {subspaces[key].shape}')

    return subspaces

def relation_score(subspace1, subspace2):
    subspace1 = subspace1.unsqueeze(0)
    subspace2 = subspace2.unsqueeze(1)
    return torch.cosine_similarity(subspace1, subspace2, dim=-1).abs().mean()

def relation_score2(subspace1_projector, subspace2):
    #subspace2_projected = subspace1_projector @ subspace2
    subspace2_projected = subspace2 @ subspace1_projector
    return torch.cosine_similarity(subspace2, subspace2_projected, dim=-1).abs().mean()


def get_attention_head_subspace_relations(model):
    n_subspaces = model.cfg.n_layers * model.cfg.n_heads * 4
    im = np.zeros((n_subspaces, n_subspaces))
    retval = {}
    subspaces = get_attention_head_subspaces(model)
    keys = sorted(subspaces.keys())
    projectors = {key: get_projector(subspaces[key]) for key in tqdm(keys)}
    for i, key1 in enumerate(tqdm(keys)):
        for j, key2 in enumerate(keys):
            if i == j:
                continue
            im[i, j] = relation_score2(projectors[key1], subspaces[key2])
            retval[key1, key2] = im[i, j]
    px.imshow(im, labels=dict(x='subspacex', y='subspacey', color='relation score'), x=keys, y=keys, 
                color_continuous_scale='viridis').show()
    return retval

def get_attention_head_subspace_relations_per_head(*, model, head, layer, thresh=0.5):
    retval = {}
    subspaces = get_attention_head_subspaces(model)
    head_keys = [f'{layer:02d}.{head:02d}.{mat}' for mat in ['Q', 'K', 'V', 'O']]
    keys = sorted(subspaces.keys())
    projectors = {key: get_projector(subspaces[key]) for key in tqdm(head_keys)}
    for i, key1 in enumerate(tqdm(head_keys)):
        for j, key2 in enumerate(keys):
            retval[key1, key2] = relation_score2(projectors[key1], subspaces[key2])

    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    for imat, mat in enumerate(['K', 'Q', 'O', 'V']):
        for imat2, mat2 in enumerate(['K', 'Q', 'O', 'V']):
            key = f'{layer:02d}.{head:02d}.{mat}'
            im = np.zeros((12, 12))
            for layer2 in range(model.cfg.n_layers):
                for head2 in range(model.cfg.n_heads):
                    key2 = f'{layer2:02d}.{head2:02d}.{mat2}'
                    score = retval[key, key2]
                    if score > thresh:
                        im[layer2, head2] = retval[key, key2]
                    else:
                        im[layer2, head2] = np.nan
            show_layer_head_image(im=im, ax=axes[imat, imat2])
            axes[imat, imat2].set_title(f'{key} vs XX.XX.{mat2}')




def get_relation_score_baseline():
    n_subspaces = 144 * 4 # 144 attention heads, 4 subspaces per head
    random_subspaces = torch.randn(n_subspaces, 64, 768)
    scores = []
    projectors = {i: get_projector(subspace) for i, subspace in enumerate(random_subspaces)}
    for i, ispace in enumerate(tqdm(random_subspaces)):
        for j, jspace in enumerate(random_subspaces):
            if i == j:
                continue
            scores.append(relation_score2(projectors[i], jspace))
    plt.figure(figsize=(10, 10))
    plt.hist(scores, bins=100)
    return scores

def analyze_scores(scores, *, lower_thresh, upper_thresh):
    scores_by_mat = defaultdict(list)
    gf = nx.Graph()
    all_subspace_keys = set()
    for i, (key, score) in enumerate(sorted(scores.items(), key=lambda x: x[1], reverse=True)):
        key1, key2 = key
        all_subspace_keys.add(key1)
        all_subspace_keys.add(key2)
        score = (score + scores[key2, key1]) / 2
        if key1 <= key2:
            continue
        layer1, head1, mat1 = key1.split('.')
        layer2, head2, mat2 = key2.split('.')
        if layer1 == layer2:
            if head1 == head2:
                sym1 = '*'
            else:
                sym1 = '.'
        else:
            sym1 = ' '
        if layer1 == layer2:
            sym2 = ''.join(sorted([mat1, mat2]))
        elif layer1 < layer2:
            sym2 = mat1 + mat2
        else:
            sym2 = mat2 + mat1
        if 0:
            if sym1 == '*' and (int(layer1), int(head1)) in interesting_heads:
                print(f'{sym2}{sym1}    {key1} {key2} {score:.2f}')
        if score > 2 * 0.303:
            #print(f'{sym2}{sym1}    {key1} {key2} {score:.2f}')
            gf.add_edge(key1, key2, weight=score)
        scores_by_mat[sym2 + sym1].append(score)

    for key, scores2 in sorted(scores_by_mat.items(), key=lambda x: np.mean(x[1]), reverse=True):
        print(f'{key} {np.mean(scores2):.2f} {np.std(scores2):.2f}')

    
    for key, scores2 in sorted(scores_by_mat.items(), key=lambda x: np.mean(x[1]), reverse=True):
        plt.figure(figsize=(3, 3))
        retval = plt.hist(scores2, label=key, range=(0, 1), bins=30, density=False)
        max_count = np.max(retval[0])
        plt.vlines([lower_thresh, upper_thresh], 0, max_count, color='red', label='random')
        plt.title(f'{key} {np.mean(scores2):.2f}+-{np.std(scores2):.2f}')

    nrows = int(np.ceil(np.sqrt(len(all_subspace_keys))))
    fig, axes = plt.subplots(nrows, nrows, figsize=(nrows * 3, nrows * 3))
    all_subspace_keys = sorted(all_subspace_keys)
    thresh = 0.5 # 0.303
    for i, (key, ax) in enumerate(zip(all_subspace_keys, axes.flatten())):
        ax.set_title(key)
        ims = {mat: np.zeros((12, 12)) for mat in 'KOQV'}
        for layer in range(12):
            for head in range(12):
                for mat in 'KOQV':
                    key2 = f'{layer:02d}.{head:02d}.{mat}'
                    if key2 == key:
                        ims[mat][layer, head] = 1
                        continue
                    score = scores[key, key2]
                    if score > thresh:
                        ims[mat][layer, head] = score
                    else:
                        ims[mat][layer, head] = np.nan
        im = np.concatenate([np.concatenate([ims[mat] for mat in 'KQ'], axis=1),
                                np.concatenate([ims[mat] for mat in 'OV'], axis=1)], axis=0)
        ax.imshow(im, vmin=thresh, vmax=1, cmap='turbo', interpolation='nearest')
        ax.hlines([11.5, 23.5], 0, 24, color='black')
        ax.vlines([11.5, 23.5], 0, 24, color='black')
        ax.hlines([0.5 + i for i in range(24)], 0, 24, color='black', linewidth=0.5)
        ax.vlines([0.5 + i for i in range(24)], 0, 24, color='black', linewidth=0.5)
        ax.colorbar = plt.colorbar(ax.images[0], ax=ax)
    plt.tight_layout()

def show_all_head_plots(*, model, layer, head):
    """Show all the plots for a given head."""
    scores = get_head_subspace_scores(model=model, layer=layer, head=head)
    for mat in 'KQOV':
        plt.figure(figsize=(10, 10))
        plt.title(f'{layer:02d}.{head:02d}.{mat}')
        plt.imshow(scores[mat], vmin=0, vmax=1, cmap='turbo', interpolation='nearest')
        plt.colorbar()

def describe_head_subspace_relations(*, model, scores, layer, head, thresh=0.5, omit_uninteresting=True):
    """Describe the relations between the subspaces of a given head and those of all other heads."""
    relevant_subspaces = defaultdict(dict)
    key = f'{layer:02d}.{head:02d}'
    for mat in 'KQOV':
        matkey = f'{key}.{mat}'
        for layer2 in range(model.cfg.n_layers):
            for head2 in range(model.cfg.n_heads):
                key2 = f'{layer2:02d}.{head2:02d}'
                for mat2 in 'KQOV':
                    if omit_uninteresting:
                        if mat == mat2:
                            continue # these are pretty noisy and not very interesting
                        if set([mat, mat2]) in [set(['K', 'V']), set(['Q', 'V']), set(['K', 'Q'])]:
                            continue
                        if layer < layer2 and mat != 'O':
                            continue
                        if layer < layer2 and mat2 == 'O':
                            continue
                        if layer > layer2 and mat2 != 'O':
                            continue
                        if layer > layer2 and mat == 'O':
                            continue

                    matkey2 = f'{key2}.{mat2}'
                    if matkey == matkey2:
                        continue
                    score = scores[matkey, matkey2]
                    if score > thresh:
                        relevant_subspaces[matkey][matkey2] = score
        print(f'{matkey}:')
        if relevant_subspaces[matkey]:
            most_relevant = sorted(relevant_subspaces[matkey].items(), key=lambda x: x[1], reverse=True)
            for matkey2, score in most_relevant:
                print(f'\t{matkey2}: {score:.2f}')
        else:
            print('\tNone')

    return relevant_subspaces

def describe_subspace_distinctions(*, model, key, freqs, lens='logit', trials=10, seed=0,
                                    prevent_numbers=True, temperature=2.0):
    """Qualitatively describe the distinctions a subspace is capable of making."""
    device = get_device(model)
    layer, head, mat = key.split('.')
    layer, head = int(layer), int(head)
    assert mat in ['Q', 'K', 'O', 'V']
    subspaces = get_attention_head_subspaces(model)
    subspace = subspaces[key]
    projector = get_projector(subspace)
    nsamples = 1_000
    #most_common_tokens = np.argsort(freqs)[-40_000:]
    best_pairs = []
    if prevent_numbers:
        freqs = freqs.copy()
        for i in range(50257):
            if model.tokenizer.decode([i]).strip().isnumeric():
                freqs[i] = 0
    modified_freqs = freqs ** (1 / temperature)
    modified_freqs = modified_freqs / modified_freqs.sum()
    np.random.seed(seed)
    for trial in tqdm(range(trials)):
        #tokens = np.random.choice(most_common_tokens, nsamples)
        tokens = np.random.choice(50257, nsamples, p=modified_freqs, replace=False)
        tokens = torch.tensor(tokens, device=device)
        if lens == 'logit':
            vecs = model.W_U.T[tokens]
        else:
            assert lens == 'embed'
            vecs = model.W_E[tokens]
        cosims = F.cosine_similarity(vecs.unsqueeze(0), vecs.unsqueeze(1), dim=-1)
        diffs = vecs.unsqueeze(0) - vecs.unsqueeze(1)
        proj_diffs = diffs @ projector
        proj_cosims = F.cosine_similarity(proj_diffs, diffs, dim=-1)
        figure_of_merit = 0.5 * torch.log(cosims + 2) + torch.log(proj_cosims + 1)
        for good in torch.topk(figure_of_merit.flatten(), 200)[1]:
            good1, good2 = good // nsamples, good % nsamples
            if good1 >= good2:
                continue
            token1, token2 = tokens[good1], tokens[good2]
            token1 = model.tokenizer.decode(token1)
            token2 = model.tokenizer.decode(token2)
            if token1.strip().lower() == token2.strip().lower():
                continue
            best_pairs.append((figure_of_merit[good1, good2].item(), 
                                cosims[good1, good2].item(),        
                                proj_cosims[good1, good2].item(),
                                tokens[good1], tokens[good2]))

    best_pairs.sort(reverse=True)
    for fom, cosim, proj_cosim, token1, token2 in best_pairs[:100]:
        print(key, '%.2f' % fom, 
            'sim=%.2f' % cosim,
            'diff=%.2f' % proj_cosim,
            f'"{model.tokenizer.decode(token1)}" "{model.tokenizer.decode(token2)}"')

def describe_attention_io_pairs(*, model, text, freqs, layer, head, temperature=2, prevent_numbers=True):
    tokens = get_tokens(model=model, text=text)
    str_tokens = get_str_tokens(model=model, text=text)
    wu = model.W_U.T
    print(wu.shape)
    wo = model.blocks[layer].attn.W_O[head]
    wv = model.blocks[layer].attn.W_V[head]
    v_projector = get_projector(wv.T)
    o_projector = get_projector(wo)

    if prevent_numbers:
        freqs = freqs.copy()
        for i in range(50257):
            if model.tokenizer.decode([i]).strip().isnumeric():
                freqs[i] = 0
    modified_freqs = freqs ** (1 / temperature)
    modified_freqs = modified_freqs / modified_freqs.sum()

    torch.random.manual_seed(0)
    description_tokens = torch.multinomial(torch.tensor(modified_freqs), 1000, replacement=False)
    description_tokens = sorted(description_tokens.numpy()) #+ list(tokens.numpy())
    description_tokens = description_tokens + list(tokens[0].numpy())
    wu = wu[description_tokens]

    cosims = F.cosine_similarity(wu.unsqueeze(0), wu.unsqueeze(1), dim=2)
    diffs = wu.unsqueeze(0) - wu.unsqueeze(1)
    normdiffs = F.normalize(diffs, dim=2)
    #print('cosims', cosims.shape)
    print('diffs', diffs.shape)

    model.set_use_attn_result(True)
    logits, cache = model.run_with_cache(tokens)
    attn_input = cache[f'blocks.{layer}.ln1.hook_normalized'].squeeze(0)
    print(cache[f'blocks.{layer}.attn.hook_result'].shape)
    attn_output = cache[f'blocks.{layer}.attn.hook_result'].squeeze(0)[:, head, :]
    print(attn_input.shape)
    print(attn_output.shape)
    attn_input = attn_input.unbind(0)
    attn_output = attn_output.unbind(0)
    for i, (input, output) in enumerate(zip(attn_input, attn_output)):
        print(v_projector.shape)
        input = v_projector @ input
        input_mapped = wo @ input
        projdiffs = F.cosine_similarity(diffs, input.unsqueeze(0).unsqueeze(0), dim=2).abs()
        fom = torch.log(projdiffs + 1) + 0.5 * torch.log(cosims + 2)
        fom = fom - fom.diag().diag()
        ibest = fom.argmax()
        ibest1, ibest2 = np.unravel_index(ibest, fom.shape)
        token1 = model.tokenizer.decode([description_tokens[ibest1]])
        token2 = model.tokenizer.decode([description_tokens[ibest2]])
        print(f'V[{str_tokens[i]}] "{token1}" "{token2}"', input.norm().item(), input_mapped.norm().item())

        if i < len(attn_input) - 1:
            projdiffs = F.cosine_similarity(diffs, output.unsqueeze(0).unsqueeze(0), dim=2).abs()
            fom = torch.log(projdiffs + 1) + 0.5 * torch.log(cosims + 2)
            fom = fom - fom.diag().diag()
            ibest = fom.argmax()
            ibest1, ibest2 = np.unravel_index(ibest, fom.shape)
            token1 = model.tokenizer.decode([description_tokens[ibest1]])
            token2 = model.tokenizer.decode([description_tokens[ibest2]])
            print(f'O[{str_tokens[i]} -> {str_tokens[i + 1]}] "{token1}" "{token2}"', output.norm().item())


def test_subspace_relations():
    # sanity checks for the relation score
    subspace = torch.randn(64, 768)
    projector = get_projector(subspace)
    self_score = relation_score2(projector, subspace)
    print(f'Self score: {self_score:.3f}')

    subspace2 = torch.randn(64, 768)
    subspace2 = subspace2 - subspace2 @ projector
    orth_score = relation_score2(projector, subspace2)
    print(f'Orthogonal subspace score: {orth_score:.3f}')


def eval_plot(all_evals, titles):
    fig, axes = plt.subplots(12, 12, subplot_kw=dict(polar=True), figsize=(100, 100))
    for (evals, ax, title) in zip(all_evals, axes.ravel(), titles):
        radius = np.abs(evals)
        theta = np.arctan2(evals.imag, evals.real)
        ax.scatter(theta, radius)
        ax.set_rscale('symlog')
        ax.set_title(title)
        ax.set_rlim(0, 2 * radius.max())

def plot_wov_eigenvalues(model):
    all_evals = []
    titles = []
    for layer in range(model.cfg.n_layers):        
        attn = model.blocks[layer].attn
        wo = attn.W_O
        wv = attn.W_V
        for head in range(model.cfg.n_heads):
            wo_head = wo[head]
            wv_head = wv[head]
            wvo = wo_head @ wv_head
            evals = np.linalg.eigvals(wvo.detach().cpu().numpy())
            all_evals.append(evals)
            titles.append(f'WOV evals, attn.{layer:02d}.{head:02d}')
    eval_plot(all_evals, titles)


if __name__ == '__main__':
    test_subspace_relations()