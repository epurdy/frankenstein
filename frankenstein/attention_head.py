from collections import defaultdict

from frankenstein.utils import get_attention_head_subspaces, get_projector

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