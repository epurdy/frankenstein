from collections import defaultdict

import torch

from frankenstein.utils import get_attention_head_subspaces, get_projector


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

def get_projector(subspace1):
#    return subspace1 @ torch.pinverse(subspace1.T @ subspace1) @ subspace1.T
    return subspace1.T @ torch.pinverse(subspace1 @ subspace1.T) @ subspace1

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

def analyze_scores(scores):
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

    if 0:
        subgraphs = []
        for key in all_subspace_keys:
            if key not in gf:
                continue
            nbrs = gf.neighbors(key)
            gf2 = nx.Graph()
            for nbr in nbrs:
                gf2.add_edge(key, nbr, weight=gf[key][nbr]['weight'])
            if nbrs:
                subgraphs.append((key, gf2))

        subgraphs = sorted(subgraphs, key=lambda x: x[0])
        nrows = int(np.ceil(np.sqrt(len(subgraphs))))
        fig, axes = plt.subplots(nrows, nrows, figsize=(nrows * 3, nrows * 3))
        for i, ((key, gf2), ax) in enumerate(zip(subgraphs, axes.flatten())):
            pos = nx.fruchterman_reingold_layout(gf2)
            nx.draw(gf2, with_labels=True, pos=pos, node_size=1000, font_size=8, width=0.1, node_color='red', ax=ax)
            ax.set_title(f'{key}')

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
                    if key.startswith('05.09.'):
                        if not (lower_thresh < score < upper_thresh):
                            print(f'{key} {key2} {score:.2f}')
        im = np.concatenate([np.concatenate([ims[mat] for mat in 'KQ'], axis=1),
                                np.concatenate([ims[mat] for mat in 'OV'], axis=1)], axis=0)
        ax.imshow(im, vmin=thresh, vmax=1, cmap='turbo', interpolation='nearest')
        ax.hlines([11.5, 23.5], 0, 24, color='black')
        ax.vlines([11.5, 23.5], 0, 24, color='black')
        ax.hlines([0.5 + i for i in range(24)], 0, 24, color='black', linewidth=0.5)
        ax.vlines([0.5 + i for i in range(24)], 0, 24, color='black', linewidth=0.5)
        ax.colorbar = plt.colorbar(ax.images[0], ax=ax)
    plt.tight_layout()


lower_thresh = 0.243
upper_thresh = 0.332

analyze_scores(scores)


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

def describe_subspace_distinctions(*, model, key, freqs, lens='logit', trials=10, 
                                    prevent_numbers=True, temperature=2.0):
    """Qualitatively describe the distinctions a subspace is capable of making."""
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
    for trial in tqdm(range(trials)):
        #tokens = np.random.choice(most_common_tokens, nsamples)
        tokens = np.random.choice(50257, nsamples, p=modified_freqs)
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



if __name__ == '__main__':
    test_subspace_relations()