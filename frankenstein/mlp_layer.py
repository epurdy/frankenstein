from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import gaussian_kde

from tqdm import tqdm

from frankenstein.utils import get_device, get_tokens

def cluster_mlp_neurons(model, num_clusters):
    device = get_device(model)
    for layer in tqdm(range(model.cfg.n_layers)):
        sims = np.zeros((num_clusters, num_clusters))
        mlp_win = model.blocks[layer].mlp.W_in.T.detach().cpu().numpy()
        clustering = AgglomerativeClustering(n_clusters=num_clusters, affinity='cosine', 
            linkage='complete').fit(mlp_win)
        print(np.bincount(clustering.labels_))
        for i in range(num_clusters):
            for j in range(i, num_clusters):
                cosine_similarities = F.cosine_similarity(
                    torch.tensor(mlp_win[clustering.labels_ == i], device=device).unsqueeze(0),
                    torch.tensor(mlp_win[clustering.labels_ == j], device=device).unsqueeze(1),
                    dim=-1).detach().cpu().numpy().ravel()
                sims[i, j] = cosine_similarities.mean()
                sims[j, i] = sims[i, j]
        plt.figure(figsize=(10, 10))
        plt.title(f'Cosine similarity of mlp clusters for layer {layer}')
        plt.imshow(sims)
        plt.colorbar()

def measure_mlp_mask_values(model, texts):
    device = get_device(model)
    mask_values_by_layer = defaultdict(list)
    for text in tqdm(texts):
        tokens = model.tokenizer.encode(text)
        tokens = [50256] + tokens + [50256]
        tokens = tokens[:model.cfg.n_ctx]
        tokens = torch.tensor(tokens, device = device).unsqueeze(0)
        logits, cache = model.run_with_cache(tokens)
        for i in range(12):
            mask_values = cache[f'blocks.{i}.mlp.hook_post'] / cache[f'blocks.{i}.mlp.hook_pre']
            mask_values_by_layer[i].extend(mask_values.cpu().numpy().ravel())

    fig, subplots = plt.subplots(3, 4, figsize=(20, 20))
    for i, ax in enumerate(subplots.ravel()):
        subsample = np.random.choice(mask_values_by_layer[i], 100_000)
        print(i, len(mask_values_by_layer[i]))
        ax.set_title(f'Mask values of mlp.{i}')
        ax.hist(subsample, bins=300)


def eval_plot(all_evals, titles):
    fig, axes = plt.subplots(3, 4, subplot_kw=dict(polar=True), figsize=(100, 100))
    for (evals, ax, title) in zip(all_evals, axes.ravel(), titles):
        radius = np.abs(evals)
        theta = np.arctan2(evals.imag, evals.real)
        xy = np.vstack([evals.real, evals.imag])
        z = gaussian_kde(xy)(xy)
        ax.scatter(theta, radius, c=z, s=1000, cmap='viridis', linewidth=0)
        ax.set_rscale('symlog')
        ax.set_title(title)
        ax.set_rlim(0, 2 * radius.max())

def plot_mlp_eigenvalues(model):
    all_evals = []
    titles = []
    for layer in range(model.cfg.n_layers):  
        mlp = model.blocks[layer].mlp      
        win = mlp.W_in.detach().cpu().numpy()
        wout = mlp.W_out.detach().cpu().numpy()
        winout = win @ wout
        evals = np.linalg.eigvals(winout)
        all_evals.append(evals)
        titles.append(f'MLP evals, mlp.{layer:02d}')
    eval_plot(all_evals, titles)

def plot_mlp_eigenvalues_dataset(model, dataset):
    all_evals = []
    titles = []
    for layer in range(model.cfg.n_layers):  
        mlp = model.blocks[layer].mlp      
        win = mlp.W_in
        wout = mlp.W_out
        layer_evals = []
        for text in dataset:
            tokens = get_tokens(model=model, text=text)
            logits, cache = model.run_with_cache(tokens)
            pre = cache[f'blocks.{layer}.mlp.hook_pre']
            post = cache[f'blocks.{layer}.mlp.hook_post']
            mask = post / pre
            masks = mask.squeeze(0).unbind(0)
            for mask in tqdm(masks):
                winmaskout = win @ (mask.unsqueeze(1) * wout)
                evals = np.linalg.eigvals(winmaskout.detach().cpu().numpy())
                layer_evals.extend(evals)
        all_evals.append(np.array(layer_evals))
        titles.append(f'MLP evals, mlp.{layer:02d}')
    eval_plot(all_evals, titles)

def find_mechanical_deletions(model):
    fig, subplots = plt.subplots(3, 4, figsize=(20, 20))
    fig2, subplots2 = plt.subplots(3, 4, figsize=(20, 20))
    for layer, ax, ax2 in zip(range(model.cfg.n_layers), subplots.ravel(), subplots2.ravel()):
        mlp_win = model.blocks[layer].mlp.W_in
        mlp_wout = model.blocks[layer].mlp.W_out.T
        cosines = F.cosine_similarity(mlp_win, mlp_wout, dim=-1).detach().cpu().numpy().ravel()
        dotprods = (mlp_win * mlp_wout).sum(-1).detach().cpu().numpy().ravel()
        ax.set_title(f'Cosine sim mlp.{layer} in/out')
        ax.hist(cosines, bins=100, histtype='bar')
        ax2.set_title(f'Dotprod mlp.{layer} in/out')
        ax2.hist(dotprods, bins=100, histtype='bar')

def evaluate_erase_trigger(model, texts):
    device = get_device(model)
    values_by_component = defaultdict(list)
    loudness_by_component = defaultdict(list)
    for text in tqdm(texts):
        tokens = model.tokenizer.encode(text)
        tokens = [50256] + tokens + [50256]
        tokens = tokens[:model.cfg.n_ctx]
        tokens = torch.tensor(tokens, device = device).unsqueeze(0)
        logits, cache = model.run_with_cache(tokens)
        for layer in range(model.cfg.n_layers):
            input_to_mlp = cache[f'blocks.{layer}.hook_resid_mid']
            mlp_pre = cache[f'blocks.{layer}.mlp.hook_pre']
            mlp_post = cache[f'blocks.{layer}.mlp.hook_post']
            mlp_mask = (mlp_post / mlp_pre).squeeze(0)
            dotprods = (model.blocks[layer].mlp.W_in.T * model.blocks[layer].mlp.W_out).sum(-1)
            scale = cache[f'blocks.{layer}.ln2.hook_scale'].squeeze(0).squeeze(-1)            
            values = (mlp_mask * dotprods.unsqueeze(0) / scale.unsqueeze(-1)).detach().cpu().numpy().ravel()
            values_by_component[f'mlp.{layer}'].append(values)
            norm_win = model.blocks[layer].mlp.W_in / model.blocks[layer].mlp.W_in.norm(dim=0, keepdim=True)
            #print(norm_input.norm(dim=0).min(), norm_input.norm(dim=0).max())
            loudness = (input_to_mlp.squeeze(0) @ norm_win) / scale.unsqueeze(-1)
            #print(loudness.min(), loudness.max())
            loudness_by_component[f'mlp.{layer}'].append(loudness.detach().cpu().numpy().ravel())
    
    nsamples = 1_000
    fig, axes = plt.subplots(3, 4, figsize=(20, 10))
    fig2, axes2 = plt.subplots(3, 4, figsize=(20, 10))
    for layer, ax, ax2 in zip(tqdm(range(model.cfg.n_layers)), axes.ravel(), axes2.ravel()):
        k = f'mlp.{layer}'
        values = np.concatenate(values_by_component[k])
        loudness = np.concatenate(loudness_by_component[k])
        indices = np.random.choice(np.nonzero(loudness > 0)[0], nsamples, replace=False)
        values_sampled = values[indices]
        loudness_sampled = loudness[indices]
        xy_sampled = np.vstack([loudness_sampled, values_sampled])
        #xy = np.vstack([loudness, values])
        z = gaussian_kde(xy_sampled)(xy_sampled)
        ax.set_title(f'Erasure coefficients for {k}')
        retval = ax.hist(values, bins=100, histtype='bar', range=(-1, 1))
        max_height = retval[0].max()
        ax.vlines(-1, 0, max_height, color='red')
        ax.vlines(0, 0, max_height, color='red')
        ax.vlines(1, 0, max_height, color='red')
        ax2.set_title(f'Loudness vs Erasure for {k}')
        ax2.scatter(loudness_sampled, values_sampled, c=z, s=1, zorder=2)
        ax2.set_xlim(0, 2)
        ax2.hlines(-1, 0, 1, color='red', zorder=1)
        ax2.hlines(0, 0, 1, color='red', zorder=1)                     
        ax2.vlines(0, -1, 1, color='green', zorder=1)
            