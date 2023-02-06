import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm


# we assume that x2 is the last k elements of x when end=True, otherwise it is the first k elements
def conditional_gaussian(*, mu, cov, x2, end=True, rcond=1e-6):   
    divergence_from_symmetry = torch.norm(cov - cov.T)
    if divergence_from_symmetry > 1e-3:
        print('cov is not symmetric')
        assert False

    n = mu.shape[0]
    k = x2.shape[0]
    if end:
        mu1 = mu[:-k]
        mu2 = mu[-k:]
        cov1 = cov[:-k, :-k]
        cov2 = cov[-k:, -k:]
        cov21 = cov[-k:, :-k]
        cov12 = cov[:-k, -k:]
    else:
        mu1 = mu[k:]
        mu2 = mu[:k]
        cov1 = cov[k:, k:]
        cov2 = cov[:k, :k]
        cov21 = cov[:k, k:]
        cov12 = cov[k:, :k]

    divergence_from_symmetry = torch.norm(cov1 - cov1.T)
    if divergence_from_symmetry > 1e-3:
        print('cov1 is not symmetric')
        assert False

    divergence_from_symmetry = torch.norm(cov2 - cov2.T)
    if divergence_from_symmetry > 1e-3:
        print('cov2 is not symmetric')
        assert False

    cov22_inv = torch.pinverse(cov2, rcond=rcond)
    cov22_inv = (cov22_inv + cov22_inv.T) / 2
    #cov22_inv = torch.inverse(cov2)

    divergence_from_symmetry = torch.norm(cov12 - cov21.T)
    if divergence_from_symmetry > 1e-3:
        print('cov12 is not symmetric with cov21')
        assert False

    divergence_from_symmetry = torch.norm(cov22_inv - cov22_inv.T)
    if divergence_from_symmetry > 1e-3:
        print('cov22_inv is not symmetric')
        assert False

    if 0:
        print(cov22_inv.shape, cov21.shape, cov12.shape, cov1.shape)
        print(mu1.shape, mu2.shape, x2.shape)
    mu1_cond = mu1 + cov12 @ (cov22_inv @ (x2 - mu2))
    cov1_cond = cov1 - cov12 @ (cov22_inv @ cov21)

    # make symmetric
    cov1_cond = (cov1_cond + cov1_cond.T) / 2

    divergence_from_symmetry = torch.norm(cov1_cond - cov1_cond.T)
    if divergence_from_symmetry > 1e-3:
        print('cov1_cond is not symmetric')
        assert False
    return dict(mu=mu1_cond, cov=cov1_cond)

def infer_variables(*, mu, cov, vec, key_classes, values, dim_map):
    conditioning = vec[:768]
    conditional_dist = conditional_gaussian(mu=mu, cov=cov, x2=conditioning, end=False)
    mu_cond = conditional_dist['mu']
    cov_cond = conditional_dist['cov']

    vec_preds = {}
    for key_cls in key_classes:
        max_mu_cond = -np.inf
        max_value = None
        for value_cls in values[key_cls]:
            dim = dim_map[key_cls, value_cls] - 768
            if mu_cond[dim] > max_mu_cond:
                max_mu_cond = mu_cond[dim]
                max_value = value_cls
        vec_preds[key_cls] = max_value

    return vec_preds
    


def build_conditional_classifier(*, dataset, dataset_name, key_vec, key_classes, 
    show_points=False, show_everything=False, 
    title=None, test=True):
    key_classes = tuple(sorted(key_classes.split('+')))
    key_classes = tuple(x for x in key_classes if x)

    #print('build mapping of class/value to dimension')
    values = defaultdict(list)
    for datapoint in dataset:
        for key in key_classes:
            values[key].append(datapoint[key])
    for key in key_classes:
        values[key] = sorted(set(values[key]))
    dim = 768
    dim_map = {}
    for key in key_classes:
        for value in values[key]:
            dim_map[key, value] = dim
            dim += 1

    #print('build vector dataset')
    vecs = []
    for datapoint in dataset:
        vec = datapoint[key_vec]
        vec = torch.cat([vec, torch.zeros(dim - vec.shape[0])])
        for key in key_classes:
            value = datapoint[key]
            vec[dim_map[key, value]] = 1
        vecs.append(vec)
    vecs = torch.stack(vecs)

    ntrain = int(0.8 * len(vecs))
    vecs_train = vecs[:ntrain]
    vecs_test = vecs[ntrain:]

    #print('train classifier')
    mu = vecs_train.mean(dim=0)
    cov = torch.cov(vecs_train.T)
    while True:
        try:
            covinv = torch.linalg.pinv(cov, rcond=1e-2)
            break
        except:
            cov += 1e-3 * torch.eye(cov.shape[0])

    if show_points:
        u, s, vh = torch.linalg.svd(cov[:768, :768])
        dims = u[:, :2]
        points = vecs[:, :768] @ dims
        plt.figure(figsize=(10, 10))
        plt.scatter(points[:, 0], points[:, 1], s=30, c=[datapoint['abb'] for datapoint in dataset])
        plt.title(title + ' (abb)')
        plt.show()

    if show_everything:
        for key_cls in key_classes:
            plt.figure(figsize=(10, 10))
            plt.scatter(points[:, 0], points[:, 1], s=30, c=[datapoint[key_cls] for datapoint in dataset])
            plt.title(title + ' color=%s' % (key_cls))
            plt.show()

            plt.figure(figsize=(10, 10))
            plt.scatter(points[:, 0], [datapoint[key_cls] for datapoint in dataset], s=30, 
                        c=[datapoint['abb'] for datapoint in dataset])
            plt.title(title + ' coord0 vs. (%s)' % (key_cls))
            plt.show()

            plt.figure(figsize=(10, 10))
            plt.scatter(points[:, 1], [datapoint[key_cls] for datapoint in dataset], s=30, 
                        c=[datapoint['abb'] for datapoint in dataset])
            plt.title(title + ' coord1 vs. (%s)' % (key_cls))
            plt.show()


    if test:
        preds = []
        hits = defaultdict(int)
        for vec, datapoint in tqdm(zip(vecs_test, dataset[ntrain:]), total=len(vecs_test)):
            conditioning = vec[:768]
            #conditional_dist = conditional_gaussian(mu=mu[::-1], cov=cov[::-1, ::-1], x2=conditioning[::-1])
            #mu_cond = conditional_dist['mu'][::-1]
            #cov_cond = conditional_dist['cov'][::-1, ::-1]
            conditional_dist = conditional_gaussian(mu=mu, cov=cov, x2=conditioning, end=False)
            mu_cond = conditional_dist['mu']
            cov_cond = conditional_dist['cov']

            vec_preds = {}
            for key_cls in key_classes:
                max_mu_cond = -np.inf
                max_value = None
                for value_cls in values[key_cls]:
                    dim = dim_map[key_cls, value_cls] - 768
                    if mu_cond[dim] > max_mu_cond:
                        max_mu_cond = mu_cond[dim]
                        max_value = value_cls
                vec_preds[key_cls] = max_value
                if max_value == datapoint[key_cls]:
                    hits[key_cls] += 1
            preds.append(vec_preds)

        acc = {key: hits[key] / len(preds) for key in key_classes}
        rand = {key: np.bincount([datapoint[key] for datapoint in dataset[:ntrain]]).max() / ntrain for key in key_classes}
        for key in acc:
            print(f'{dataset_name}  {key}: {acc[key]:.3f} (rand={rand[key]:.3f})')
    else:
        acc = None
        rand = None

    return dict(acc=acc, rand=rand, model=dict(mu=mu, cov=cov, covinv=covinv), dim_map=dim_map, values=values)

def parse_head_number(x):
    if x == 'mlp':
        return x
    return int(x)

def parse_head_string(x):
    return [tuple(map(parse_head_number, x.split('.'))) for x in x.split()]

def remember_head_hook(result, hook, layer, head, idx, cache):
    cache[layer, head] = result[0, idx, head, :].clone()

def remember_mlp_hook(result, hook, layer, idx, cache):
    cache[layer, 'mlp'] = result[0, idx, :].clone()

def get_log_probs(*, model, vec, key_cls, temperature=0.01):
    print
    vec = torch.tensor(vec)
    conditioning = vec[:768]
    conditional_dist = conditional_gaussian(mu=model['model']['mu'], cov=model['model']['cov'], x2=conditioning, end=False)
    mu_cond = conditional_dist['mu']
    cov_cond = conditional_dist['cov']

    vals = []
    for value_cls in sorted(model['values'][key_cls]):
        dim = model['dim_map'][key_cls, value_cls] - 768
        vals.append(mu_cond[dim])
    vals = torch.stack(vals) / temperature
    vals = vals.log_softmax(dim=0)

    return {value_cls: val for value_cls, val in zip(sorted(model['values'][key_cls]), vals)}


def intervene(*, model, ioi, i, datasets, classifiers, classifiers2, intervention_heads, measure_heads,
            method='class'):
            
    datapoint = datasets[0, 0][i]
    augmented_prompt = datapoint['augmented_prompt']
    tokens = augmented_prompt['tokens']
    model.reset_hooks()        
    endpos = datapoint['endpos']
    name1pos = datapoint['name1pos']
    name2pos = datapoint['name2pos']
    iopos = datapoint['iopos']
    s2pos = datapoint['s2pos']
    s1pos = datapoint['s1pos']
    ioname = datapoint['ioname']
    sname = datapoint['sname']
    name1 = datapoint['name1']
    name2 = datapoint['name2']
    abb = datapoint['abb']

    # clean
    model.reset_hooks()
    clean_logits, tmpcache = model.run_with_cache(tokens[:1 + endpos].long())
    clean_logits = clean_logits[0, -1, :]
    clean_logit_diff = clean_logits[ioname] - clean_logits[sname]
    clean_logit_diff = clean_logit_diff.item()
    clean_cache = {}
    for layer, head in measure_heads:
        clean_cache[layer, head] = tmpcache[f'blocks.{layer}.attn.hook_result'][0, endpos, head, :].clone()

    # s1pos = s1pos
    hooks = []
    for layer, head in intervention_heads:
        x2 = torch.zeros(len(classifiers[layer, head]['dim_map']))
        dim = classifiers[layer, head]['dim_map']['abb', datapoint['abb']]
        x2[dim - 768] = 1                
        dim = classifiers[layer, head]['dim_map']['sname', sname]
        x2[dim - 768] = 1
        swap_value = conditional_gaussian(mu=classifiers[layer, head]['model']['mu'],
                                            cov=classifiers[layer, head]['model']['cov'],
                                            x2=x2)['mu']

        if head == 'mlp':
            hooks.append(
                (f'blocks.{layer}.hook_mlp_out',
                    partial(swap_mlp_hook, swap_value=swap_value, idx=s2pos)))
        else:
            hooks.append(
                (f'blocks.{layer}.attn.hook_result', 
                    partial(swap_head_hook, head=head, swap_value=swap_value, idx=s2pos)))

    s1pos_s1pos_cache = {}
    for layer, head in measure_heads:
        hooks.append(
            (f'blocks.{layer}.attn.hook_result', 
                partial(remember_head_hook, layer=layer, head=head, idx=endpos, cache=s1pos_s1pos_cache)))
    model.reset_hooks()
    s1pos_s1pos_ablated_logits = model.run_with_hooks(
        tokens[:1 + endpos].long(), fwd_hooks=hooks)[0, -1, :]
    s1pos_s1pos_ablated_logit_diff = s1pos_s1pos_ablated_logits[ioname] - s1pos_s1pos_ablated_logits[sname]
    s1pos_s1pos_ablated_logit_diff = s1pos_s1pos_ablated_logit_diff.item()

    # s1pos = iopos
    hooks = []
    for layer, head in intervention_heads:
        x2 = torch.zeros(len(classifiers[layer, head]['dim_map']))
        dim = classifiers[layer, head]['dim_map']['abb', not datapoint['abb']]
        x2[dim - 768] = 1                
        dim = classifiers[layer, head]['dim_map']['sname', sname]
        x2[dim - 768] = 1
        swap_value = conditional_gaussian(mu=classifiers[layer, head]['model']['mu'],
                                            cov=classifiers[layer, head]['model']['cov'],
                                            x2=x2)['mu']

        if head == 'mlp':
            hooks.append(
                (f'blocks.{layer}.hook_mlp_out',
                    partial(swap_mlp_hook, swap_value=swap_value, idx=s2pos)))
        else:
            hooks.append(
                (f'blocks.{layer}.attn.hook_result', 
                    partial(swap_head_hook, head=head, swap_value=swap_value, idx=s2pos)))
    s1pos_fakepos_cache = {}
    for layer, head in measure_heads:
        hooks.append(
            (f'blocks.{layer}.attn.hook_result', 
                partial(remember_head_hook, layer=layer, head=head, idx=endpos, cache=s1pos_fakepos_cache)))
    model.reset_hooks()
    s1pos_fakepos_ablated_logits = model.run_with_hooks(
        tokens[:1 + endpos].long(), fwd_hooks=hooks)[0, -1, :]
    s1pos_fakepos_ablated_logit_diff = s1pos_fakepos_ablated_logits[ioname] - s1pos_fakepos_ablated_logits[sname]
    s1pos_fakepos_ablated_logit_diff = s1pos_fakepos_ablated_logit_diff.item()

    # probe measure_heads about s1pos at s2
    clean_probs_per_head = {}
    s1pos_s1pos_probs_per_head = {}
    s1pos_fakepos_probs_per_head = {}
    for layer, head in measure_heads:
        clean_offset = clean_cache[layer, head]
        s1pos_s1pos_offset = s1pos_s1pos_cache[layer, head]
        s1pos_fakepos_offset = s1pos_fakepos_cache[layer, head]

        # calculate probs
        clean_logprobs = get_log_probs(
            model=classifiers2[layer, head],
            key_cls='sname',
            vec=clean_offset)
        clean_probs = {k: np.exp(v) for k, v in clean_logprobs.items()}
        #clean_probs = np.exp(clean_logprobs)

        s1pos_s1pos_logprobs = get_log_probs(
            model=classifiers2[layer, head],
            key_cls='sname',
            vec=s1pos_s1pos_offset)
        s1pos_s1pos_probs = {k: np.exp(v) for k, v in s1pos_s1pos_logprobs.items()}
        #s1pos_s1pos_probs = np.exp(s1pos_s1pos_logprobs)

        s1pos_fakepos_logprobs = get_log_probs(
            model=classifiers2[layer, head],
            key_cls='sname',
            vec=s1pos_fakepos_offset)
        s1pos_fakepos_probs = {k: np.exp(v) for k, v in s1pos_fakepos_logprobs.items()}
        #s1pos_fakepos_probs = np.exp(s1pos_fakepos_logprobs)

        clean_probs_per_head[layer, head] = (clean_probs[sname], clean_probs[ioname])
        s1pos_s1pos_probs_per_head[layer, head] = (s1pos_s1pos_probs[sname], s1pos_s1pos_probs[ioname])
        s1pos_fakepos_probs_per_head[layer, head] = (s1pos_fakepos_probs[sname], s1pos_fakepos_probs[ioname])

    return dict(
        clean_logit_diff=clean_logit_diff,
        s1pos_s1pos_ablated_logit_diff=s1pos_s1pos_ablated_logit_diff,
        s1pos_fakepos_ablated_logit_diff=s1pos_fakepos_ablated_logit_diff,
        clean_cache=clean_cache,
        s1pos_s1pos_cache=s1pos_s1pos_cache,
        s1pos_fakepos_cache=s1pos_fakepos_cache,
        clean_probs_per_head=clean_probs_per_head,
        s1pos_s1pos_probs_per_head=s1pos_s1pos_probs_per_head,
        s1pos_fakepos_probs_per_head=s1pos_fakepos_probs_per_head,
    )


def plot_relevant_stuff(model, ioi, name_mover_heads, sinhibition_heads, early_heads, datasets,
                        method, n=10):
    name_mover_heads = parse_head_string(name_mover_heads)
    sinhibition_heads = parse_head_string(sinhibition_heads)
    early_heads = parse_head_string(early_heads)

    classifiers = dict()
    classifiers2 = dict()
    for layer, head in tqdm(early_heads + sinhibition_heads):
        dataset = datasets[layer, head]
        classifiers[layer, head] = build_conditional_classifier(dataset=dataset, dataset_name=f'L{layer}_H{head}', 
            key_vec='vecs2', key_classes='sname+abb', test=False)

    for layer_head in tqdm(sinhibition_heads):
        dataset = datasets[layer_head]
        classifiers2[layer_head] = build_conditional_classifier(dataset=dataset, dataset_name=f'L{layer_head[0]}_H{layer_head[1]}', 
            key_vec='vecend', key_classes='sname+abb', test=False)

    clean_logit_diffs = []
    s1pos_s1pos_ablated_logit_diffs = []
    s1pos_fakepos_ablated_logit_diffs = []

    clean_xs = defaultdict(list)
    clean_ys = defaultdict(list)
    s1pos_xs = defaultdict(list)
    s1pos_ys = defaultdict(list)
    fakepos_xs = defaultdict(list)
    fakepos_ys = defaultdict(list)
    for i in tqdm(range(n)):
        stuff = intervene(model=model, ioi=ioi, i=i, datasets=datasets,
                          classifiers=classifiers, classifiers2=classifiers2,
                          measure_heads=sinhibition_heads,
                          intervention_heads=early_heads, method=method)

        clean_logit_diffs.append(stuff['clean_logit_diff'])
        s1pos_s1pos_ablated_logit_diffs.append(stuff['s1pos_s1pos_ablated_logit_diff'])
        s1pos_fakepos_ablated_logit_diffs.append(stuff['s1pos_fakepos_ablated_logit_diff'])

        for layer, head in sinhibition_heads:
            if 0:
                print(layer, head)
                print('clean', stuff['clean_probs_per_head'][layer, head])
                print('s1pos_s1pos', stuff['s1pos_s1pos_probs_per_head'][layer, head])
                print('s1pos_fakepos', stuff['s1pos_fakepos_probs_per_head'][layer, head])
                print() 
            clean_xs[layer, head].append(stuff['clean_probs_per_head'][layer, head][0])
            clean_ys[layer, head].append(stuff['clean_probs_per_head'][layer, head][1])
            s1pos_xs[layer, head].append(stuff['s1pos_s1pos_probs_per_head'][layer, head][0])
            s1pos_ys[layer, head].append(stuff['s1pos_s1pos_probs_per_head'][layer, head][1])
            fakepos_xs[layer, head].append(stuff['s1pos_fakepos_probs_per_head'][layer, head][0])
            fakepos_ys[layer, head].append(stuff['s1pos_fakepos_probs_per_head'][layer, head][1])

    for layer, head in sinhibition_heads:
        if 0:
            plt.figure(figsize=(10, 10))
            plt.scatter(clean_xs[layer, head], clean_ys[layer, head], label='clean')
            plt.scatter(s1pos_xs[layer, head], s1pos_ys[layer, head], label='s1pos_s1pos')
            plt.scatter(fakepos_xs[layer, head], fakepos_ys[layer, head], label='s1pos_fakepos')
            plt.legend()
            plt.title(f'L{layer}.H{head}')
            plt.xlabel('P(s1pos=s1pos)')
            plt.ylabel('P(s1pos=fakepos)')         

        plt.figure(figsize=(10, 10))
        fig, ax = plt.subplots(3, 1, figsize=(20, 10))
        ax[0].hist(clean_xs[layer, head], label='clean', alpha=0.5, range=(0, 1), bins=(n // 10))
        ax[1].hist(s1pos_xs[layer, head], label='s1pos=s1pos', alpha=0.5, range=(0, 1), bins=(n // 10))
        ax[2].hist(fakepos_xs[layer, head], label='s1pos=fakepos', alpha=0.5, range=(0, 1), bins=(n // 10))
        ax[0].set_title(f'L{layer}.H{head} P(s1pos=s1pos) clean')
        ax[1].set_title(f'L{layer}.H{head} P(s1pos=s1pos) s1pos=s1pos')
        ax[2].set_title(f'L{layer}.H{head} P(s1pos=s1pos) s1pos=fakepos')
        plt.xlabel('P(sname=sname)')
        plt.ylabel('Frequency')

    print('Flipped answers:', sum([1 for v in s1pos_fakepos_ablated_logit_diffs if v < 0]), 'out of', n)
    print('Accidental flipped answers:', sum([1 for v in s1pos_s1pos_ablated_logit_diffs if v > 0]), 'out of', n)

    plt.figure(figsize=(10, 10))
    plt.title('Ablations according to probe')
    plt.scatter(clean_logit_diffs, s1pos_s1pos_ablated_logit_diffs, color='blue', label='s1pos = s1pos, iopos=iopos')
    plt.scatter(clean_logit_diffs, s1pos_fakepos_ablated_logit_diffs, color='green', label='s1pos = iopos, iopos=s1pos')
    minval = min(min(clean_logit_diffs), min(s1pos_s1pos_ablated_logit_diffs), min(s1pos_fakepos_ablated_logit_diffs))
    maxval = max(max(clean_logit_diffs), max(s1pos_s1pos_ablated_logit_diffs), max(s1pos_fakepos_ablated_logit_diffs))
    plt.plot([minval, maxval], [minval, maxval], color='red')
    plt.plot([minval, maxval], [-minval, -maxval], color='red')
    plt.plot([minval, maxval], [0, 0], color='black')
    plt.plot([0, 0], [minval, maxval], color='black')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.title('Ablations according to probe')
    plt.hist(clean_logit_diffs, color='blue', label='clean', alpha=0.5, bins=(n // 10))
    plt.hist(s1pos_s1pos_ablated_logit_diffs, color='green', label='s1pos=s1pos,iopos=iopos', alpha=0.5, bins=(n // 10))
    plt.hist(s1pos_fakepos_ablated_logit_diffs, color='red', label='s1pos=iopos,iopos=s1pos', alpha=0.5, bins=(n // 10))
    plt.vlines(0, 0, (n // 10), color='black')
    plt.legend()
    plt.xlabel('logit difference: IO - S')
    plt.ylabel('count')
    plt.show()

        
