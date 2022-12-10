from collections import defaultdict
import os
import shutil

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.express as px
import torch
import torch.nn.functional as F
from tqdm import tqdm

from frankenstein.utils import (get_projector, get_device, show_layer_head_image, get_tokens, get_str_tokens,
                                get_frequency, get_model, get_openwebtext_dataset)  


MAT_ORDER = list('QKVO')

def make_key_name(*, layer, head, mat):
    return f'{layer:02d}.{head:02d}.{mat}'

class DossierMaker:
    def __init__(self, model, texts, num_texts=10):
        self.device = get_device(model)
        self.model = model
        self.texts = texts
        self.num_texts = num_texts
        self.preparation_done = False
        self.make_index()
        self.prepare_dossiers()

    def prepare_dossiers(self):
        """Do work that has to be shared between heads"""
        self.freqs = get_frequency(dataset=self.texts, model=self.model)
        self.subspaces = get_attention_head_subspaces(self.model)
        self.projectors = {key: get_projector(subspace) for key, subspace in self.subspaces.items()}
        self.preparation_done = True

    def make_index(self, *, path='dossiers'):
        index_file = os.path.join(path, 'index.html')
        with open(index_file, 'w') as f:
            f.write('''<html>
            <head>
            <style>
            table, th, td {
                border: 1px solid black;
                border-collapse: collapse;
                padding: 30px;
            }
            </style>
            <body>
            <table>
            ''')
            for layer in range(self.model.cfg.n_layers):
                f.write('<tr>\n')
                for head in range(self.model.cfg.n_heads):
                    f.write(f'<td><a href="dossier_{layer:02d}_{head:02d}/index.html">')
                    f.write(f'attn.{layer:02d}.{head:02d}</a></td>\n')
                f.write('</tr>\n')
            f.write('</table></body></html>\n')

    def dossier(self, *, layer, head, path='dossiers'):
        path = os.path.join(path, f'dossier_{layer:02d}_{head:02d}')
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        index_file = os.path.join(path, 'index.html')
        wov_evals_image = os.path.join(path, 'wov_eigenvalues.png')
        plot_wov_eigenvalues(model=self.model, layer=layer, head=head, path=wov_evals_image)
        scores = get_subspace_relation_scores(model=self.model, layer=layer, head=head,
                                              subspaces=self.subspaces,
                                              projectors=self.projectors)
        subspace_relations_image = os.path.join(path, 'subspace_relations.png')
        plot_subspace_relation_scores(model=self.model, scores=scores, layer=layer, head=head, 
                                        path=subspace_relations_image)
        relevant_subspaces = describe_head_subspace_relations(model=self.model, layer=layer, head=head,
                                                                scores=scores)

        attention_pattern_image = os.path.join(path, 'attention_pattern.png')
        positional_score = describe_attention_patterns_positional(model=self.model, layer=layer, head=head, 
                                                                  path=attention_pattern_image)

        embedding_attention_pairs = describe_embedding_attention_patterns(model=self.model, 
                                                                       layer=layer, head=head, freqs=self.freqs)

        distinctions_text = {}
        for mat in MAT_ORDER:
            key = make_key_name(layer=layer, head=head, mat=mat)
            mat_text = describe_subspace_distinctions(model=self.model, key=key, freqs=self.freqs, trials=1)
            distinctions_text[mat] = mat_text

        head_name = f'{layer:02d}.{head:02d}'                                                    

        q_links = []
        for key2, score2 in relevant_subspaces[make_key_name(layer=layer, head=head, mat='Q')].items():
            layer2, head2, mat2 = key2.split('.')
            q_links.append(f'<a href="../dossier_{layer2}_{head2}/index.html">{key2}</a> ({score2:.3f})<br />')
        q_links = '\n'.join(q_links)
        k_links = []
        for key2, score2 in relevant_subspaces[make_key_name(layer=layer, head=head, mat='K')].items():
            layer2, head2, mat2 = key2.split('.')
            k_links.append(f'<a href="../dossier_{layer2}_{head2}/index.html">{key2}</a> ({score2:.3f})<br />')
        k_links = '\n'.join(k_links)
        v_links = []
        for key2, score2 in relevant_subspaces[make_key_name(layer=layer, head=head, mat='V')].items():
            layer2, head2, mat2 = key2.split('.')
            v_links.append(f'<a href="../dossier_{layer2}_{head2}/index.html">{key2}</a> ({score2:.3f})<br />')
        v_links = '\n'.join(v_links)
        o_links = []
        for key2, score2 in relevant_subspaces[make_key_name(layer=layer, head=head, mat='O')].items():
            layer2, head2, mat2 = key2.split('.')
            o_links.append(f'<a href="../dossier_{layer2}_{head2}/index.html">{key2}</a> ({score2:.3f})<br />')
        o_links = '\n'.join(o_links)

        html = f'''
        <html>
        <head>
            <title>Attention Head Dossier - attn.{head_name}</title>
            <style>
                img {{ max-width: 800px; float: left; }}
            </style>
        </head>
        <body>
            <h1>Attention Head Dossier - attn.{head_name}</h1>
            <h2>Positional score {positional_score:.2f} (max ~11K)</h2>
            <img src="{os.path.basename(wov_evals_image)}" />
            <img src="{os.path.basename(subspace_relations_image)}" />
            <img src="{os.path.basename(attention_pattern_image)}" />
            <table>
                <tr>
                    <th>Q</th>
                    <th>K</th>
                    <th>V</th>
                    <th>O</th>
                </tr>
                <tr>
                    <td colspan="4">Subspace Links</td>
                </tr>
                <tr>
                    <td>{q_links}</td>
                    <td>{k_links}</td>
                    <td>{v_links}</td>
                    <td>{o_links}</td>
                </tr>
                <tr>
                    <td colspan="4">Subspace Distinctions</td>
                </tr>
                <tr>
                    <td><textarea rows="100" cols="40">{distinctions_text['Q']}</textarea></td>
                    <td><textarea rows="100" cols="40">{distinctions_text['K']}</textarea></td>
                    <td><textarea rows="100" cols="40">{distinctions_text['V']}</textarea></td>
                    <td><textarea rows="100" cols="40">{distinctions_text['O']}</textarea></td>
                </tr>
                <tr>
                    <td colspan="2">QK Pairs</td>
                    <td colspan="2">OV Pairs</td>
                </tr>
                <tr>
                    <td colspan="2"><textarea rows="100" cols="40">{embedding_attention_pairs}</textarea></td>
                    <td colspan="2"><textarea rows="100" cols="40"></textarea></td>
                </tr>
            </table>
        </body>
        </html>
        '''
        with open(index_file, 'w') as f:
            f.write(html)


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


def relation_score2(*, subspace1_projector, subspace2):
    subspace2_projected = subspace2 @ subspace1_projector
    return torch.cosine_similarity(subspace2, subspace2_projected, dim=-1).abs().mean()


def get_subspace_relation_scores(*, model, layer=None, head=None, subspaces=None, projectors=None):
    retval = {}
    if subspaces is None:
        subspaces = get_attention_head_subspaces(model)
    keys = sorted(subspaces.keys())
    if layer is None and head is None:
        head_keys = keys
    else:
        assert layer is not None and head is not None
        head_keys = [f'{layer:02d}.{head:02d}.{mat}' for mat in MAT_ORDER]
 
    if projectors is None:
        projectors = {key: get_projector(subspaces[key]) for key in tqdm(keys)}

    for key1 in tqdm(head_keys):
        for key2 in keys:
            retval[key1, key2] = relation_score2(subspace1_projector=projectors[key1], 
                                                 subspace2=subspaces[key2])
    return retval


def plot_subspace_relation_scores(*, model, scores, head, layer, path=None, thresh=0.5):
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    
    for imat, mat in enumerate(MAT_ORDER):
        key = f'{layer:02d}.{head:02d}.{mat}'
        for imat2, mat2 in enumerate(MAT_ORDER):
            im = np.zeros((model.cfg.n_layers, model.cfg.n_heads))
            for layer2 in range(model.cfg.n_layers):
                for head2 in range(model.cfg.n_heads):
                    key2 = f'{layer2:02d}.{head2:02d}.{mat2}'
                    score = scores[key, key2]
                    if score >= thresh:
                        im[layer2, head2] = score
                    else:
                        im[layer2, head2] = np.nan
            show_layer_head_image(im=im, ax=axes[imat, imat2])
            axes[imat, imat2].set_title(f'{key} vs XX.XX.{mat2}')

    if path is not None:
        plt.savefig(path)


def get_relation_score_baseline(model):
    n_subspaces = model.cfg.n_layers * model.cfg.n_heads * 4 # 4 matrices per head
    random_subspaces = torch.randn(n_subspaces, model.d_head, model.d_head, device=get_device(model))
    scores = []
    projectors = {i: get_projector(subspace) for i, subspace in enumerate(random_subspaces)}
    for i, ispace in enumerate(tqdm(random_subspaces)):
        for j, jspace in enumerate(random_subspaces):
            if i == j:
                continue
            scores.append(relation_score2(projectors[i], jspace))
    plt.figure(figsize=(10, 10))
    plt.hist(scores, bins=100)
    plt.title('Distribution of relation scores between random subspaces of the same shapes as model')


def analyze_subspace_relation_scores(scores, *, lower_thresh, upper_thresh, show_thresh=0.5):
    scores_by_mat = defaultdict(list)
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

        scores_by_mat[sym2 + sym1].append(score)

    for key, scores2 in sorted(scores_by_mat.items(), key=lambda x: np.mean(x[1]), reverse=True):
        print(f'{key} {np.mean(scores2):.2f} {np.std(scores2):.2f}')
        plt.figure(figsize=(3, 3))
        retval = plt.hist(scores2, label=key, range=(0, 1), bins=30, density=False)
        max_count = np.max(retval[0])
        plt.vlines([lower_thresh, upper_thresh], 0, max_count, color='red', label='random')
        plt.title(f'{key} {np.mean(scores2):.2f}+-{np.std(scores2):.2f}')


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
    assert mat in MAT_ORDER
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

    retval = []
    best_pairs.sort(reverse=True)
    for fom, cosim, proj_cosim, token1, token2 in best_pairs[:100]:
        str1 = model.tokenizer.decode(token1)
        str2 = model.tokenizer.decode(token2)
        retval.append(f'"{str1}" "{str2}"')
        print(key, '%.2f' % fom, 
            'sim=%.2f' % cosim,
            'diff=%.2f' % proj_cosim,
            f'"{str1}" "{str2}"')

    return '\n'.join(retval)


def describe_attention_io_pairs(*, model, text, freqs, layer, head, temperature=2, prevent_numbers=True):
    tokens = get_tokens(model=model, text=text)
    str_tokens = get_str_tokens(model=model, text=text)
    wu = model.W_U.T
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

    model.set_use_attn_result(True)
    logits, cache = model.run_with_cache(tokens)
    attn_input = cache[f'blocks.{layer}.ln1.hook_normalized'].squeeze(0)
    attn_output = cache[f'blocks.{layer}.attn.hook_result'].squeeze(0)[:, head, :]
    attn_input = attn_input.unbind(0)
    attn_output = attn_output.unbind(0)
    for i, (input, output) in enumerate(zip(attn_input, attn_output)):
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
    self_score = relation_score2(subspace1_projector=projector, subspace2=subspace)
    print(f'Self score: {self_score:.3f}')

    subspace2 = torch.randn(64, 768)
    subspace2 = subspace2 - subspace2 @ projector
    orth_score = relation_score2(subspace1_projector=projector, subspace2=subspace2)
    print(f'Orthogonal subspace score: {orth_score:.3f}')


def single_eigenvalue_plot(*, evals, title, ax):
    radius = np.abs(evals)
    theta = np.arctan2(evals.imag, evals.real)
    ax.scatter(theta, radius)
    ax.set_rscale('symlog')
    ax.set_title(title)
    ax.set_rlim(0, 2 * radius.max())


def compute_wov_eigenvalues(*, model, layer, head):
    attn = model.blocks[layer].attn
    wo = attn.W_O
    wv = attn.W_V
    wo_head = wo[head]
    wv_head = wv[head]
    wvo = wo_head @ wv_head
    evals = np.linalg.eigvals(wvo.detach().cpu().numpy())
    return evals

def plot_wov_eigenvalues(*, model, layer, head, path):
    evals = compute_wov_eigenvalues(model=model, layer=layer, head=head)
    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(10, 10))
    single_eigenvalue_plot(evals=evals, title=f'WOV evals, attn.{layer:02d}.{head:02d}', ax=ax)
    if path:
        fig.savefig(path)


def plot_all_wov_eigenvalues(model):
    all_evals = []
    titles = []
    for layer in range(model.cfg.n_layers):        
        for head in range(model.cfg.n_heads):
            evals = compute_wov_eigenvalues(model=model, layer=layer, head=head)
            all_evals.append(evals)
            titles.append(f'WOV evals, attn.{layer:02d}.{head:02d}')
    fig, axes = plt.subplots(model.cfg.n_layers, model.cfg.n_heads, 
                            subplot_kw=dict(polar=True), figsize=(100, 100))
    for (evals, ax, title) in zip(all_evals, axes.ravel(), titles):
        single_eigenvalue_plot(evals=evals, title=title, ax=ax)


def describe_attention_patterns_positional(*, model, layer, head, path):
    wq = model.blocks[layer].attn.W_Q[head].detach().numpy()
    wk = model.blocks[layer].attn.W_K[head].detach().numpy()
    qk = np.matmul(wq, wk.T)
    pos = model.W_pos.detach().numpy()
    pos_qk_pos = np.matmul(np.matmul(pos, qk), pos.T)
    pos_qk_pos = pos_qk_pos - np.tril(pos_qk_pos)
    im = pos_qk_pos.copy()
    im[pos_qk_pos == 0] = np.nan
    fig = plt.figure()
    plt.title(f'Positional preference for attention head {layer:02d}.{head:02d}')
    max_score = np.abs(pos_qk_pos).max()
    plt.imshow(im, origin='lower', cmap='RdBu_r', vmin=-max_score, vmax=max_score)
    plt.xlabel('Query')
    plt.ylabel('Key')
    plt.colorbar()
    if path:
        fig.savefig(path)
    score = np.linalg.norm(pos_qk_pos)
    print(f'Layer {layer:02d}.{head:02d} score: {score:.2f}')
    return score

def describe_embedding_attention_patterns(model, layer, head, freqs):
    wq = model.blocks[layer].attn.W_Q[head].detach().numpy()
    wk = model.blocks[layer].attn.W_K[head].detach().numpy()
    qk = np.matmul(wq, wk.T)
    we = model.W_E.detach().numpy()
    frequent_tokens = (freqs > 1e-5).nonzero()[0]
    print(len(frequent_tokens))
    we = we[frequent_tokens, :]
    weqkwe = np.matmul(we, np.matmul(qk, we.T))
    idxs = np.argpartition(weqkwe.ravel(), -100)[-100:]
    tok1_idxs, tok2_idxs = np.unravel_index(idxs, weqkwe.shape)
    tok1s = frequent_tokens[tok1_idxs]
    tok2s = frequent_tokens[tok2_idxs]
    str_tok1s = [model.tokenizer.decode([tok]) for tok in tok1s]
    str_tok2s = [model.tokenizer.decode([tok]) for tok in tok2s]
    pairs = []
    for tok1, tok2 in zip(str_tok1s, str_tok2s):
        print(tok1, tok2)
        pairs.append(f'"{tok1}" "{tok2}"')
    return '\n'.join(pairs)


if __name__ == '__main__':
    model = get_model(name='gpt2', device='cpu')
    texts = get_openwebtext_dataset()
    dossier_maker = DossierMaker(model=model, texts=texts)
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            dossier_maker.dossier(layer=layer, head=head)